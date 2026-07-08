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

  // 1. upload audio
  const uploadRes = await fetch("https://api.assemblyai.com/v2/upload", {
    method: "POST",
    headers: { authorization: key },
    body: fs.createReadStream(filePath),
  });
  if (!uploadRes.ok) throw new Error("AssemblyAI upload failed");
  const { upload_url } = await uploadRes.json();

  // 2. request transcript
  const transcriptRes = await fetch("https://api.assemblyai.com/v2/transcript", {
    method: "POST",
    headers: { authorization: key, "content-type": "application/json" },
    body: JSON.stringify({ audio_url: upload_url }),
  });
  if (!transcriptRes.ok) throw new Error("AssemblyAI transcript request failed");
  const { id } = await transcriptRes.json();

  // 3. poll until done
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 3000));
    const pollRes = await fetch(`https://api.assemblyai.com/v2/transcript/${id}`, {
      headers: { authorization: key },
    });
    const pollData = await pollRes.json();
    if (pollData.status === "completed") return pollData.text;
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

async function transcribeWithGemini(filePath) {
  const key = (process.env.GEMINI_API_KEY || "").trim();
  if (!key) throw new Error("GEMINI_API_KEY is not set on the server");

  const model = process.env.GEMINI_MODEL || "gemini-2.0-flash";
  const ext = path.extname(filePath).toLowerCase();
  const mimeType = GEMINI_MIME_TYPES[ext] || "audio/mp3";

  const audioBytes = fs.readFileSync(filePath);
  // Gemini's inline_data path supports requests up to ~20MB; larger files
  // would need the Files API (upload first, then reference by URI).
  if (audioBytes.length > 19 * 1024 * 1024) {
    throw new Error("File is too large for inline Gemini transcription (limit ~19MB). Try a shorter clip.");
  }
  const base64Audio = audioBytes.toString("base64");

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": key,
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { text: "Transcribe this audio in full. Return only the transcript text, no extra commentary." },
              { inline_data: { mime_type: mimeType, data: base64Audio } },
            ],
          },
        ],
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
  res.json({
    providerResolved: PROVIDER,
    providerRawEnv: JSON.stringify(process.env.TRANSCRIBE_PROVIDER),
    hasOpenAIKey: Boolean((process.env.OPENAI_API_KEY || "").trim()),
    hasGeminiKey: Boolean((process.env.GEMINI_API_KEY || "").trim()),
    hasAssemblyAIKey: Boolean((process.env.ASSEMBLYAI_API_KEY || "").trim()),
  });
});

app.listen(PORT, () => {
  console.log(`ScribeLink backend running on http://localhost:${PORT} (provider: ${PROVIDER})`);
});
