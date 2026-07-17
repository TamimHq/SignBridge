"""
SignBridge FastAPI Server
=========================
Serves sign recognition (BdSL model) and Whisper transcription to the Flutter app.

Endpoints:
  GET  /health                  → connectivity check
  POST /api/recognize           → keypoint sequence → predicted sign word
  POST /api/transcribe          → audio file → text (Whisper)
  GET  /api/keypoints/{word}    → fetch avatar keypoints (optional, for streaming)

Run:
  pip install fastapi uvicorn torch numpy python-multipart openai-whisper
  python server.py
  # or: uvicorn server:app --host 0.0.0.0 --port 8000

The --host 0.0.0.0 makes it reachable from your phone on the same WiFi.
Find your PC's local IP (ipconfig on Windows) and point the Flutter app at
  http://<your-pc-ip>:8000
"""

import os
import json
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# Config — point these at where you put the model files
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("SIGNBRIDGE_MODEL_DIR", "./models")

BDSL_MODEL_PATH = f"{MODEL_DIR}/bdsl_bilstm_scripted.pt"
BDSL_LABELS_PATH = f"{MODEL_DIR}/bdsl_idx_to_gloss.json"

# ASL model — train later using the same pipeline; server handles its absence
ASL_MODEL_PATH = f"{MODEL_DIR}/asl_bilstm_scripted.pt"
ASL_LABELS_PATH = f"{MODEL_DIR}/asl_idx_to_gloss.json"

TARGET_FRAMES = 30
FEATURE_DIM = 144
CONFIDENCE_THRESHOLD = 0.35  # below this, return "uncertain"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
class SignModel:
    def __init__(self, model_path: str, labels_path: str, name: str):
        self.name = name
        self.available = False
        self.model = None
        self.idx_to_gloss = {}

        if not os.path.exists(model_path):
            print(f"[{name}] Model not found at {model_path} — endpoint will return unavailable")
            return
        if not os.path.exists(labels_path):
            print(f"[{name}] Labels not found at {labels_path}")
            return

        try:
            self.model = torch.jit.load(model_path, map_location=device)
            self.model.eval()
            with open(labels_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Keys may be strings; normalize to int → gloss
            self.idx_to_gloss = {int(k): v for k, v in raw.items()}
            self.available = True
            print(f"[{name}] Loaded — {len(self.idx_to_gloss)} classes")
        except Exception as e:
            print(f"[{name}] Failed to load: {e}")

    def predict(self, keypoints: np.ndarray) -> tuple[str, float]:
        """keypoints: (T, 144) array. Returns (gloss, confidence)."""
        x = self._prepare(keypoints)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            conf = conf.item()
            idx = idx.item()
        gloss = self.idx_to_gloss.get(idx, "unknown")
        return gloss, conf

    def _prepare(self, keypoints: np.ndarray) -> torch.Tensor:
        """Resample to TARGET_FRAMES, ensure shape (1, 30, 144)."""
        kps = np.asarray(keypoints, dtype=np.float32)
        if kps.ndim != 2 or kps.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected (T, {FEATURE_DIM}), got {kps.shape}")

        # Resample time dimension to exactly TARGET_FRAMES
        T = kps.shape[0]
        if T != TARGET_FRAMES:
            idx = np.linspace(0, T - 1, TARGET_FRAMES).astype(int)
            kps = kps[idx]

        return torch.tensor(kps, dtype=torch.float32).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Whisper (lazy-loaded — only when first transcription request arrives)
# ─────────────────────────────────────────────────────────────────────────────
class WhisperEngine:
    def __init__(self):
        self.model = None
        self.model_size = os.environ.get("WHISPER_SIZE", "base")

    def _ensure_loaded(self):
        if self.model is None:
            import whisper
            print(f"[Whisper] Loading '{self.model_size}' model (first request)...")
            self.model = whisper.load_model(self.model_size, device=device)
            print("[Whisper] Loaded")

    def transcribe(self, audio_path: str, language: str) -> str:
        self._ensure_loaded()
        # language: 'en' or 'bn'
        result = self.model.transcribe(
            audio_path,
            language=language,
            fp16=(device.type == "cuda"),
        )
        return result.get("text", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SignBridge Server")

# Allow the Flutter app (any origin) to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
bdsl_model = SignModel(BDSL_MODEL_PATH, BDSL_LABELS_PATH, "BdSL")
asl_model = SignModel(ASL_MODEL_PATH, ASL_LABELS_PATH, "ASL")
whisper_engine = WhisperEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Request/response schemas
# ─────────────────────────────────────────────────────────────────────────────
class RecognizeRequest(BaseModel):
    keypoints: list[list[float]]  # (T, 144)
    language: str                  # "asl_en" or "bdsl_bn"


class RecognizeResponse(BaseModel):
    word: str
    confidence: float
    language: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "models": {
            "bdsl": bdsl_model.available,
            "asl": asl_model.available,
        },
    }


@app.post("/api/recognize", response_model=RecognizeResponse)
def recognize(req: RecognizeRequest):
    # Pick the model for the requested language
    if req.language == "bdsl_bn":
        model = bdsl_model
    elif req.language == "asl_en":
        model = asl_model
    else:
        raise HTTPException(400, f"Unknown language: {req.language}")

    if not model.available:
        raise HTTPException(
            503,
            f"{model.name} model not loaded on server. "
            f"Train it and place the .pt file in {MODEL_DIR}/",
        )

    try:
        kps = np.array(req.keypoints, dtype=np.float32)
        gloss, conf = model.predict(kps)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")

    # Below threshold → signal uncertainty rather than a wrong guess
    word = gloss if conf >= CONFIDENCE_THRESHOLD else "…"

    return RecognizeResponse(
        word=word,
        confidence=round(conf, 3),
        language=req.language,
    )


@app.post("/api/recognize_video")
async def recognize_video(
    video: UploadFile = File(...),
    language: str = Form("bdsl_bn"),
):
    """
    Accept a short recorded video clip of one sign, extract keypoints with
    MediaPipe server-side, and return the predicted word.

    This is the primary recognition path for the mobile app — the phone just
    records a clip and uploads it; all ML happens here.
    """
    # Pick model
    if language == "bdsl_bn":
        model = bdsl_model
    elif language == "asl_en":
        model = asl_model
    else:
        raise HTTPException(400, f"Unknown language: {language}")

    if not model.available:
        raise HTTPException(
            503, f"{model.name} model not loaded. Place the .pt in {MODEL_DIR}/"
        )

    # Save uploaded video to a temp file
    import tempfile
    suffix = os.path.splitext(video.filename or "clip.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from keypoint_extractor import extract_from_video
        keypoints = extract_from_video(tmp_path)   # (30, 144)
        gloss, conf = model.predict(keypoints)
    except Exception as e:
        raise HTTPException(500, f"Video recognition failed: {e}")
    finally:
        os.unlink(tmp_path)

    word = gloss if conf >= CONFIDENCE_THRESHOLD else "…"
    return RecognizeResponse(
        word=word,
        confidence=round(conf, 3),
        language=language,
    )


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("en")):
    # Save uploaded audio to a temp file
    import tempfile
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = whisper_engine.transcribe(tmp_path, language)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)

    return {"text": text, "language": language}


@app.get("/")
def root():
    return {
        "service": "SignBridge Server",
        "endpoints": ["/health", "/api/recognize", "/api/transcribe"],
        "bdsl_ready": bdsl_model.available,
        "asl_ready": asl_model.available,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("SignBridge Server starting")
    print(f"  Device      : {device}")
    print(f"  BdSL model  : {'✓' if bdsl_model.available else '✗ (not found)'}")
    print(f"  ASL model   : {'✓' if asl_model.available else '✗ (not found)'}")
    print(f"  Model dir   : {os.path.abspath(MODEL_DIR)}")
    print("=" * 50)
    print("\nReachable at http://0.0.0.0:8000")
    print("From your phone, use http://<your-pc-ip>:8000")
    print("Find your PC IP with: ipconfig (Windows) / ifconfig (Mac/Linux)\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)