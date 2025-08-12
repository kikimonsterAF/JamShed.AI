# JamTab FastAPI Server

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

Health check: `GET http://localhost:8080/health`

## Endpoints
- `GET /usage` — returns entitlement/quota
- `POST /analyze` — upload audio/video file; returns chords, form, and scale suggestions

## Dependencies

This service uses FFmpeg for robust media decoding and librosa/scipy for audio analysis.

- FFmpeg (CLI) must be installed and available on PATH.
  - Windows (PowerShell):
    - `winget install -e --id Gyan.FFmpeg`
    - Then open a new terminal and verify: `ffmpeg -version`
  - macOS: `brew install ffmpeg`
  - Linux (Debian/Ubuntu): `sudo apt update; sudo apt install -y ffmpeg`

- Python packages are listed in `requirements.txt` (includes `librosa`, `soundfile`, `scipy`, `numpy`).

## Architecture Note

MVP is server-side only for analysis. The mobile app always uploads media to `/analyze`; any on-device processing is deferred.

