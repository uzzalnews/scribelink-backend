require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const { execFile } = require("child_process");
const fetch = require("node-fetch");
const FormData = require("form-data");

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;
const PROVIDER = (process.env.TRANSCRIBE_PROVIDER || "openai").trim().toLowerCase();
const TMP_DIR = path.join(os.tmpdir(), "scribelink");
if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR, { recursive: true });

const upload = multer({ dest: TMP_DIR, limits: { fileSize: 200 * 1024 * 1024 } });

// ---------- transcription providers ----------

async function transcribeWithOpenAI(filePath) {
  const key = (process.env.OPENAI_API_KEY || "").trim();
  if (!key) throw new Error("OPENAI_API_KEY is not set on the server");

  const form = new FormData();
  form.append("file", fs.createReadStream(filePath));
  form.append("model", "whisper-1");

  const res = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, ...form.getHeaders() },
    body: form,
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`OpenAI transcription failed: ${errText}`);
  }
  const data = await res.json();
  return data.text;
}

async function transcribeWithAssemblyAI(filePath) {
  const key = (process.env.ASSEMBLYAI_API_KEY || "").trim();
  if (!key) throw new Error("ASSEMBLYAI_API_KEY is not set on the server");

  const languageCode = (process.env.LANGUAGE_CODE || "").trim(); // e.g. "bn" for Bengali

  // 1. upload audio
  const uploadRes = await fetch("https://api.assemblyai.com/v2/upload", {
    method: "POST",
    headers: { authorization: key },
    body: fs.createReadStream(filePath),
  });
  if (!uploadRes.ok) {
    const errBody = await uploadRes.text();
    throw new Error(`AssemblyAI upload failed: ${errBody}`);
  }
  const { upload_url } = await uploadRes.json();

  // 2. request transcript. The audio mixes languages (code-switching), so instead of
  // forcing a single language (which garbles mixed-language speech), let the model
  // detect language per-segment. If LANGUAGE_CODE is set, it's used as a hint via
  // expected_languages rather than a hard lock.
  const transcriptBody = {
    audio_url: upload_url,
    speaker_labels: true,
    speech_model: "universal", // Universal-2 supports code-switching across 99 languages, including Bengali
    language_detection: true,
    language_detection_options: {
      code_switching: true,
      ...(languageCode ? { expected_languages: [languageCode] } : {}),
    },
  };

  const transcriptRes = await fetch("https://api.assemblyai.com/v2/transcript", {
    method: "POST",
    headers: { authorization: key, "content-type": "application/json" },
    body: JSON.stringify(transcriptBody),
  });
  if (!transcriptRes.ok) {
    const errBody = await transcriptRes.text();
    throw new Error(`AssemblyAI transcript request failed: ${errBody}`);
  }
  const { id } = await transcriptRes.json();

  // 3. poll until done
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 3000));
    const pollRes = await fetch(`https://api.assemblyai.com/v2/transcript/${id}`, {
      headers: { authorization: key },
    });
    const pollData = await pollRes.json();
    if (pollData.status === "completed") {
      // If speaker labels were returned, format as an interview-style transcript
      if (Array.isArray(pollData.utterances) && pollData.utterances.length > 0) {
        return pollData.utterances
          .map((u) => `Speaker ${u.speaker}: ${u.text}`)
          .join("\n\n");
      }
      return pollData.text;
    }
    if (pollData.status === "error") throw new Error(pollData.error || "AssemblyAI transcription error");
  }
  throw new Error("AssemblyAI transcription timed out");
}

const GEMINI_MIME_TYPES = {
  ".mp3": "audio/mp3",
  ".wav": "audio/wav",
  ".m4a": "audio/mp4",
  ".aac": "audio/aac",
  ".ogg": "audio/ogg",
  ".flac": "audio/flac",
  ".mp4": "audio/mp4",
};

async function uploadFileToGemini(filePath, mimeType, key) {
  const numBytes = fs.statSync(filePath).size;
  const displayName = path.basename(filePath);

  // Step 1: start a resumable upload session
  const startRes = await fetch("https://generativelanguage.googleapis.com/upload/v1beta/files", {
    method: "POST",
    headers: {
      "x-goog-api-key": key,
      "X-Goog-Upload-Protocol": "resumable",
      "X-Goog-Upload-Command": "start",
      "X-Goog-Upload-Header-Content-Length": String(numBytes),
      "X-Goog-Upload-Header-Content-Type": mimeType,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ file: { display_name: displayName } }),
  });
  if (!startRes.ok) throw new Error(`Gemini upload (start) failed: ${await startRes.text()}`);
  const uploadUrl = startRes.headers.get("x-goog-upload-url");
  if (!uploadUrl) throw new Error("Gemini did not return an upload URL");

  // Step 2: upload the actual bytes
  const fileBuffer = fs.readFileSync(filePath);
  const uploadRes = await fetch(uploadUrl, {
    method: "POST",
    headers: {
      "Content-Length": String(numBytes),
      "X-Goog-Upload-Offset": "0",
      "X-Goog-Upload-Command": "upload, finalize",
    },
    body: fileBuffer,
  });
  if (!uploadRes.ok) throw new Error(`Gemini upload (bytes) failed: ${await uploadRes.text()}`);
  const info = await uploadRes.json();
  let file = info.file;

  // Step 3: wait until the file is ACTIVE (usually instant for audio, but be safe)
  for (let i = 0; i < 20 && file.state === "PROCESSING"; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    const checkRes = await fetch(`https://generativelanguage.googleapis.com/v1beta/${file.name}`, {
      headers: { "x-goog-api-key": key },
    });
    file = await checkRes.json();
  }
  if (file.state === "FAILED") throw new Error("Gemini file processing failed");
  return file; // { uri, mimeType, ... }
}

async function transcribeWithGemini(filePath) {
  const key = (process.env.GEMINI_API_KEY || "").trim();
  if (!key) throw new Error("GEMINI_API_KEY is not set on the server");

  const model = process.env.GEMINI_MODEL || "gemini-2.0-flash";
  const ext = path.extname(filePath).toLowerCase();
  const mimeType = GEMINI_MIME_TYPES[ext] || "audio/mp3";
  const numBytes = fs.statSync(filePath).size;

  const promptText =
    "Transcribe this audio verbatim, exactly as spoken. Keep the original spoken language and script " +
    "(do not translate or transliterate into a different script or language). Do not add commentary, " +
    "summaries, or explanations. Do not repeat any phrase or sentence more than once. If a section is " +
    "unclear, write [inaudible] instead of guessing or inventing words. Return only the transcript text.";
  let audioPart;

  if (numBytes > 19 * 1024 * 1024) {
    // Large file: upload via the Files API first, then reference by URI
    const uploadedFile = await uploadFileToGemini(filePath, mimeType, key);
    audioPart = { file_data: { mime_type: uploadedFile.mimeType || mimeType, file_uri: uploadedFile.uri } };
  } else {
    // Small file: send inline as base64
    const base64Audio = fs.readFileSync(filePath).toString("base64");
    audioPart = { inline_data: { mime_type: mimeType, data: base64Audio } };
  }

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": key,
      },
      body: JSON.stringify({
        contents: [{ parts: [{ text: promptText }, audioPart] }],
        generationConfig: { temperature: 0 },
      }),
    }
  );

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Gemini transcription failed: ${errText}`);
  }
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "";
  return text.trim();
}

async function transcribeFile(filePath) {
  if (PROVIDER === "assemblyai") return transcribeWithAssemblyAI(filePath);
  if (PROVIDER === "gemini") return transcribeWithGemini(filePath);
  return transcribeWithOpenAI(filePath);
}

// ---------- yt-dlp helpers ----------
// NOTE: requires yt-dlp installed on the server (pip install yt-dlp)
// Only pass the url as a discrete execFile argument (never shell-interpolated)
// to avoid command injection.

function runYtDlpJson(url) {
  return new Promise((resolve, reject) => {
    execFile(
      "yt-dlp",
      ["--dump-json", "--no-warnings", "--no-playlist", url],
      { maxBuffer: 1024 * 1024 * 10 },
      (err, stdout) => {
        if (err) return reject(new Error("Could not read that link. Is yt-dlp installed and is the link supported?"));
        try {
          resolve(JSON.parse(stdout));
        } catch (e) {
          reject(new Error("Could not parse video info"));
        }
      }
    );
  });
}

function runYtDlpDownloadAudio(url, outBaseName) {
  const outputTemplate = path.join(TMP_DIR, `${outBaseName}.%(ext)s`);
  return new Promise((resolve, reject) => {
    execFile(
      "yt-dlp",
      ["-x", "--audio-format", "mp3", "--no-playlist", "-o", outputTemplate, url],
      { maxBuffer: 1024 * 1024 * 10 },
      (err) => {
        if (err) return reject(new Error("Failed to download audio from that link"));
        resolve(path.join(TMP_DIR, `${outBaseName}.mp3`));
      }
    );
  });
}

function formatDuration(seconds) {
  if (!seconds) return null;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---------- routes ----------

app.get("/api/link-info", async (req, res) => {
  const url = req.query.url;
  if (!url) return res.status(400).json({ error: "Missing url" });
  try {
    const info = await runYtDlpJson(url);
    res.json({
      title: info.title,
      thumbnail: info.thumbnail,
      duration: formatDuration(info.duration),
    });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.post("/api/transcribe-file", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });
  try {
    const text = await transcribeFile(req.file.path);
    res.json({ text });
  } catch (err) {
    res.status(500).json({ error: err.message });
  } finally {
    fs.unlink(req.file.path, () => {});
  }
});

app.post("/api/transcribe-link", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "Missing url" });

  const jobId = crypto.randomBytes(8).toString("hex");
  let audioPath;
  try {
    audioPath = await runYtDlpDownloadAudio(url, jobId);
    const text = await transcribeFile(audioPath);
    res.json({ text });
  } catch (err) {
    res.status(500).json({ error: err.message });
  } finally {
    if (audioPath && fs.existsSync(audioPath)) fs.unlink(audioPath, () => {});
  }
});

app.get("/health", (req, res) => res.json({ ok: true, provider: PROVIDER }));

app.get("/api/debug", (req, res) => {
  const geminiKey = (process.env.GEMINI_API_KEY || "").trim();
  res.json({
    providerResolved: PROVIDER,
    providerRawEnv: JSON.stringify(process.env.TRANSCRIBE_PROVIDER),
    languageCode: JSON.stringify((process.env.LANGUAGE_CODE || "").trim()),
    hasOpenAIKey: Boolean((process.env.OPENAI_API_KEY || "").trim()),
    hasGeminiKey: Boolean(geminiKey),
    geminiKeyPreview: geminiKey ? `${geminiKey.slice(0, 6)}...${geminiKey.slice(-4)} (length ${geminiKey.length})` : null,
    hasAssemblyAIKey: Boolean((process.env.ASSEMBLYAI_API_KEY || "").trim()),
  });
});

app.listen(PORT, () => {
  console.log(`ScribeLink backend running on http://localhost:${PORT} (provider: ${PROVIDER})`);
});
