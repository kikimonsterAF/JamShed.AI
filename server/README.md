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
- `POST /analyze` — upload audio/video file; returns chords, form, scale suggestions (stub)

