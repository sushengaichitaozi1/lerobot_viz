# Dataset Viz

Dataset Viz is a local-first LeRobot dataset management and visualization system.

## What Is Included
- Frontend SPA (Vue 3): `project/frontend/index.html`
- Backend API (FastAPI): `project/backend/app/main.py`
- API documentation: `project/backend/API.md`

## Frontend/Backend Separation
- Frontend files are under `project/frontend/`.
- Backend code is under `project/backend/`.
- Compatibility page `project/111.html` redirects to `project/frontend/index.html`.

## Main Capabilities
- Login and JWT-based session flow.
- Register local dataset paths and run async indexing.
- Browse robots, task types, datasets, and episodes.
- Episode detail visualization (video + timeseries).
- Update dataset/episode metadata and path fields.
- Delete dataset/episode with optional local filesystem sync.
- Generate/serve Rerun visualization outputs.

## Quick Start

### 1) Start backend
From repository root:

```powershell
python start_backend.py
```

Backend URLs:
- `http://127.0.0.1:8899/docs`
- `http://127.0.0.1:8899/openapi.json`

### 2) Start frontend

```powershell
cd project
python -m http.server 5090
```

Frontend URLs:
- `http://127.0.0.1:5090/frontend/index.html`
- `http://127.0.0.1:5090/111.html` (compatibility redirect)

## Naming Rule (LeRobot v3)
Dataset folders should follow:

```text
{robot_type}_{task_name}_{YYYY-MM-DD}
```

Example:

```text
so101_cleandesk_2000-01-01
```

## Leadership Review Checklist
- Dataset registration and indexing complete successfully.
- Dataset/episode lists and details load correctly.
- Update endpoints change DB/local state as expected.
- Delete endpoints synchronize DB/local state as expected.
- Stream/thumbnail/timeseries/Rerun endpoints are available.

## Related Docs
- Frontend guide: `project/frontend/README.md`
- Backend guide: `project/backend/README.md`
- Backend API spec (human-readable): `project/backend/API.md`
- Full architecture notes: `project/ARCHITECTURE.md`
- Framework report: `project/FRAMEWORK.md`
