# Backend Service

This directory contains the FastAPI backend for Dataset Viz.

## Key Paths
- App entry: `project/backend/app/main.py`
- API schemas: `project/backend/app/schemas.py`
- Models: `project/backend/app/models.py`
- API reference: `project/backend/API.md`

## Install

```powershell
cd project/backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

```text
DATABASE_URL=sqlite:///lerobot_dataset_v2.db
DATA_ROOT=project/data
JWT_SECRET=dev-secret
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin
CORS_ORIGINS=http://localhost:5090
PRIMARY_CAMERA_KEY=top
```

## Run

```powershell
cd project/backend
.\.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8899 --reload
```

or from repo root:

```powershell
python start_backend.py
```

## Validation
- Swagger docs: `http://127.0.0.1:8899/docs`
- OpenAPI JSON: `http://127.0.0.1:8899/openapi.json`

## Notes
- LeRobot v3 dataset name validation supports `{robot}_{task}_{YYYY-MM-DD}`.
- Dataset and item update/delete endpoints support optional local filesystem synchronization.
- Rerun outputs are served under `/downloads/rrd`.
