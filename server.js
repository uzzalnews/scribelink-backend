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
const PROVIDER = (process.env.TRANSCRIBE_PROVIDER || "openai").toLowerCase();
const TMP_DIR = path.join(os.tmpdir(), "scribelink");
if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR, { recursive: true });

const upload = multer({ dest: TMP_DIR, limits: { fileSize: 200 * 1024 * 1024 } });

// ---------- transcription providers ----------

async function transcribeWithOpenAI(filePath) {
  const key = process.env.OPENAI_API_KEY;
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
  const key = process.env.ASSEMBLYAI_API_KEY;
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

async function transcribeFile(filePath) {
  return PROVIDER === "assemblyai"
    ? transcribeWithAssemblyAI(filePath)
    : transcribeWithOpenAI(filePath);
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

app.listen(PORT, () => {
  console.log(`ScribeLink backend running on http://localhost:${PORT} (provider: ${PROVIDER})`);
});
