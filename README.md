# Auction / Race AI Analyzer v2.2.1 (Fixed 422 + Better Errors)

This package is designed to avoid Render import errors by providing a stable `app.py` entrypoint.

## Files
- app.py (entrypoint) -> imports `app` from app_v22.py
- app_v22.py (FastAPI app + /api/extract + /api/analyze)
- requirements.txt
- static_v22/index.html

## Run locally
pip install -r requirements.txt
export OPENAI_API_KEY="..."
uvicorn app:app --host 0.0.0.0 --port 8000

## Render
Build Command:
pip install -r requirements.txt

Start Command:
uvicorn app:app --host 0.0.0.0 --port $PORT

Env Vars:
OPENAI_API_KEY=...
(optional) OPENAI_MODEL=gpt-5.2

Root Directory:
(blank) repository root
