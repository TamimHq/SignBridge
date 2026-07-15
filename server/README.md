# SignBridge Server

FastAPI server that serves sign recognition and Whisper transcription to the Flutter app.

## Setup

1. Create a folder for the server and put `server.py` in it.

2. Create a `models/` subfolder and copy these from your Google Drive
   (`SignBD/output/`):
   ```
   models/
     bdsl_bilstm_scripted.pt
     bdsl_idx_to_gloss.json
   ```
   (ASL model files go here too once you train that model — the server
    runs fine without them, it just returns "unavailable" for ASL recognition.)

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: `openai-whisper` is only needed for the mic/transcribe feature.
   If you only want sign recognition for now, you can skip it and remove
   the whisper import.

## Run

```bash
python server.py
```

You'll see:
```
SignBridge Server starting
  Device      : cpu   (or cuda if you have a GPU)
  BdSL model  : ✓
  ASL model   : ✗ (not found)
Reachable at http://0.0.0.0:8000
```

## Connect the Flutter app

1. Find your PC's local IP:
   - Windows: `ipconfig` → look for IPv4 Address (e.g. 192.168.1.42)
   - Mac/Linux: `ifconfig` or `ip addr`

2. Make sure your phone and PC are on the **same WiFi network**.

3. In the Flutter app, set the server URL to `http://<your-pc-ip>:8000`
   (the ApiClient defaults to localhost — you'll add a settings field or
    hardcode your IP for testing).

4. Test connectivity: open `http://<your-pc-ip>:8000/health` in your phone's
   browser. You should see JSON with `"status": "ok"`.

## Endpoints

- `GET /health` — connectivity + which models are loaded
- `POST /api/recognize` — body: `{"keypoints": [[...144...], ...], "language": "bdsl_bn"}`
  → `{"word": "মা", "confidence": 0.87, "language": "bdsl_bn"}`
- `POST /api/transcribe` — multipart audio file + language field → `{"text": "..."}`

## Troubleshooting

- **Phone can't reach server**: Windows Firewall often blocks incoming
  connections. Allow Python through the firewall, or temporarily disable it
  for testing on a trusted network.
- **BdSL model not found**: check the `models/` folder has the two BdSL files
  with exact names.
- **Whisper slow first request**: the model loads lazily on the first
  transcription call (can take 10-30s to download+load). Subsequent calls are fast.