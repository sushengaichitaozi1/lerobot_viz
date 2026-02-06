# LeRobot Dataset API v2.0 (Local Backend)

This backend follows `project/ARCHITECTURE.md` and powers `project/111.html`.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r project/backend/requirements.txt
```

## Environment Variables
```
DATABASE_URL=sqlite:///lerobot_dataset_v2.db
DATA_ROOT=project/data
JWT_SECRET=dev-secret
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin
CORS_ORIGINS=http://localhost:5090
PRIMARY_CAMERA_KEY=top
```

## Run
```bash
uvicorn app.main:app --reload --port 8000
```

## Notes
- Episodes are indexed from the LeRobot-style folder layout under `DATA_ROOT` (e.g. `data/robots/...`).
- Video is streamed on demand from image sequences (no pre-generated `.mp4` files).
- `.rrd` files are saved under `project/downloads/rrd` and served from `/downloads/rrd`.
