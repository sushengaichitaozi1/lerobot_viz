# Dataset Viz Framework and Feature Report

Last updated: 2026-02-08
Purpose: management review document for architecture, module boundaries, implemented capabilities, and recent delivery details.

---

## 1. Project Scope

Dataset Viz is a local-first LeRobot dataset management and visualization system.

Main use cases:
- Register local dataset folders.
- Index metadata into a relational database.
- Browse robots, task types, datasets, and episodes.
- Play media and inspect timeseries data.
- Update and delete dataset/episode records with DB synchronization.
- Generate and serve Rerun artifacts.

---

## 2. High-Level Architecture

```text
[Frontend SPA: Vue3 + Tailwind]
             |
             | HTTP + JWT
             v
[Backend API: FastAPI + SQLAlchemy]
      |                 |                 |
      |                 |                 +--> Rerun export/service
      |                 +--> DB (SQLite/MySQL/PostgreSQL)
      +--> Local filesystem datasets (LeRobot layout)
```

Layer responsibilities:
- Frontend layer (`project/frontend/index.html`): user interactions and visualization.
- API layer (`project/backend/app/main.py`): routing, auth, validation, response contracts.
- Service layer (`project/backend/app/services/*`): dataset parse/edit/stream/thumb/rerun.
- Data layer (`project/backend/app/models.py`, `project/backend/app/db.py`): persistence and query models.

---

## 3. Frontend/Backend Separation (Implemented)

Current separation status:
- Frontend entry moved to `project/frontend/index.html`.
- Compatibility page `project/111.html` now redirects to frontend entry.
- Backend remains isolated in `project/backend/`.
- Backend API reference is now centralized in `project/backend/API.md`.

Result:
- Cleaner ownership boundaries.
- Easier future frontend modularization.
- API contract easier to review and test.

---

## 4. Repository Structure (Key Paths)

```text
project/
  frontend/
    index.html
    README.md
  backend/
    app/
      main.py
      auth.py
      config.py
      db.py
      models.py
      schemas.py
      indexer.py
      services/
        parser_service.py
        lerobot_v3.py
        dataset_editor.py
        video_streamer.py
        thumbnail.py
        rerun_viz.py
    API.md
    README.md
  111.html
  README.md
  FRAMEWORK.md
```

---

## 5. Functional Capabilities (Detailed)

### 5.1 Authentication
- `POST /auth/login`
- `POST /auth/refresh`
- `GET /auth/me`
- JWT bearer flow integrated in frontend requests.

### 5.2 Registration and Indexing
- `POST /datasets/register`
- `POST /index/scan`
- `GET /index/status`
- Dataset naming validation for LeRobot v3:
  - `{robot_type}_{task_name}_{YYYY-MM-DD}`

### 5.3 Browsing and Query
- `GET /robots`
- `GET /task-types`
- `GET /datasets`
- `GET /datasets/{datasetId}/items`
- `GET /items`
- `GET /items/{itemId}`

### 5.4 Media and Timeseries
- `GET /items/{itemId}/stream`
- `GET /items/{itemId}/thumbnail`
- `GET /items/{itemId}/timeseries`
- Multi-camera + action/state chart support in UI.

### 5.5 Update Operations
- Dataset update: `PUT /datasets/{datasetId}`
  - Fields: `robot`, `task_type`, `storage_path`
  - Options: `update_local`, `force_local`
- Item update: `PUT /items/{itemId}`
  - Fields: `episode_id`, `robot`, `task_type`, `file_path`, `storage_path`
  - Options: `update_local`, `force_local`

### 5.6 Delete Operations
- Dataset delete: `DELETE /datasets/{datasetId}` (DB records)
- Item delete: `DELETE /items/{itemId}`
  - Options: `delete_local`, `force_local`
  - LeRobot v3 path uses local delete + reindex sync strategy.

### 5.7 Rerun Visualization
- `POST /items/{itemId}/visualize/rrd`
- `POST /items/{itemId}/visualize/server`
- Download path: `/downloads/rrd/{filename}`

---

## 6. Database Model Summary

Core tables:
- `robots`
- `task_types`
- `items`
- `camera_infos`
- `index_tasks`
- `index_logs`
- `users`

Critical field:
- `items.storage_path`
  - Used as dataset root anchor for grouping, local sync, and reindex consistency.

Relationships:
- `Robot 1-N TaskType`
- `Robot 1-N Item`
- `TaskType 1-N Item`
- `Item 1-N CameraInfo`

---

## 7. API Documentation Standardization

A unified API reference has been delivered at:
- `project/backend/API.md`

Format includes:
- Endpoint (method + path)
- Auth requirement
- Query parameters
- Body examples
- Response examples
- Behavior notes for local-sync flags

---

## 8. What Was Delivered in This Iteration

### 8.1 Frontend cleanup and naming consistency
- Unified update naming (`updateDataset`, `updateItem`, `updateSelectedItem`).
- Refactored registration state naming (`datasetForm`).
- Split loading flags (`loading.datasets` vs `loading.items`).
- Added reusable request helper (`buildJsonRequestOptions`).

### 8.2 Backend readability and maintainability
- Standardized variable naming in update/delete flow:
  - `apply_local_changes`
  - `force_local_unlock`
  - `delete_local_files`
  - `requested_task_type`
  - `requested_storage_path`
- Preserved query aliases for compatibility:
  - `update_local`, `force_local`, `delete_local`

### 8.3 Project cleanup
Removed unused files:
- `project/backend/app/etl.py`
- `project/backend/app/storage.py`
- `project/backend/parser_service.py`
- `project/test_vue.html`
- `__pycache__` folders

### 8.4 Documentation set completed
- Project summary: `project/README.md`
- Frontend doc: `project/frontend/README.md`
- Backend doc: `project/backend/README.md`
- Backend API doc: `project/backend/API.md`
- This framework report: `project/FRAMEWORK.md`

---

## 9. Startup and Validation

Backend:

```powershell
python start_backend.py
```

Frontend:

```powershell
cd project
python -m http.server 5090
```

Validation URLs:
- `http://127.0.0.1:8899/docs`
- `http://127.0.0.1:8899/openapi.json`
- `http://127.0.0.1:5090/frontend/index.html`

---

## 10. Suggested Next Steps

- Split frontend single file into component-based structure.
- Add automated API regression tests (update/delete edge cases first).
- Add role-based permission tiers and operation audit logs.
- Add bulk update/delete operations with safety controls.

---

This file can be used directly as a management review artifact.
