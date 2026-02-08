# Backend API Reference

Base URL (default local): `http://127.0.0.1:8899`

Auth mode:
- `Bearer <access_token>` for all protected endpoints.
- Public/static endpoints are listed explicitly.

Response style:
- Success returns JSON model.
- Errors use FastAPI standard:

```json
{"detail": "..."}
```

---

## 1) Auth

### POST `/auth/login`
- Auth: none
- Body:

```json
{
  "username": "admin",
  "password": "admin"
}
```

- 200 response:

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "expires_in": 86400,
  "user": {"id": 1, "username": "admin", "displayName": "admin", "roles": ["admin"]}
}
```

### POST `/auth/refresh`
- Auth: none
- Body:

```json
{"refresh_token": "..."}
```

### GET `/auth/me`
- Auth: required
- Response: current user profile

---

## 2) Taxonomy and Listing

### GET `/robots`
- Auth: required
- Query: none
- Response: robot summaries

### GET `/robots/{name}/types`
- Auth: required
- Path: `name` robot name
- Response: task types under this robot

### GET `/task-types`
- Auth: required
- Query:
  - `robot` (optional)
- Response: task type summaries

### GET `/datasets`
- Auth: required
- Query:
  - `robot` (optional)
  - `type` (optional)
- Response: dataset summaries

### GET `/datasets/{datasetId}/items`
- Auth: required
- Path: `datasetId` (base64-url encoded dataset path)
- Query:
  - `page` (default `1`)
  - `page_size` (default from server config)
- Response: paged items for one dataset

### GET `/items`
- Auth: required
- Query:
  - `robot` (optional)
  - `type` (optional)
  - `page` (default `1`)
  - `page_size`
- Response: paged global items

### GET `/items/{itemId}`
- Auth: required
- Response: item detail with cameras

---

## 3) Dataset and Item Update/Delete

### PUT `/datasets/{datasetId}`
- Auth: required
- Path:
  - `datasetId`: encoded dataset path
- Query:
  - `update_local` (bool, default `true`)
  - `force_local` (bool, default `true`)
- Body (all optional):

```json
{
  "robot": "so101",
  "task_type": "cleandesk",
  "storage_path": "D:/so101_cleandesk_2000-01-01"
}
```

- Notes:
  - With `update_local=true`, backend updates local dataset metadata/path and reindexes DB.
  - With `update_local=false`, only DB fields are updated.

### PUT `/items/{itemId}`
- Auth: required
- Query:
  - `update_local` (bool, default `false`)
  - `force_local` (bool, default `true`)
- Body (all optional):

```json
{
  "episode_id": "episode_000123",
  "robot": "so101",
  "task_type": "cleandesk",
  "file_path": "D:/.../episode-000123.mp4",
  "storage_path": "D:/so101_cleandesk_2000-01-01"
}
```

### DELETE `/datasets/{datasetId}`
- Auth: required
- Behavior:
  - Deletes dataset records from DB by dataset path key.
  - Does not remove full local dataset root directory.

### DELETE `/items/{itemId}`
- Auth: required
- Query:
  - `delete_local` (bool, default `true`)
  - `force_local` (bool, default `true`)
- Behavior:
  - Deletes one episode from DB.
  - If local delete enabled, attempts to remove local episode data and resync DB.

---

## 4) Media and Timeseries

### GET `/items/{itemId}/stream`
- Auth: none (current implementation)
- Query:
  - `camera` (default primary camera)
  - `resolution` (optional profile key: `360p|480p|720p|1080p`)
- Response: video stream

### GET `/items/{itemId}/thumbnail`
- Auth: none (current implementation)
- Query:
  - `camera` (default primary camera)
- Response: image bytes

### GET `/items/{itemId}/timeseries`
- Auth: none (current implementation)
- Query:
  - `max_points` (optional, default by backend)
- Response: timestamps/action/state arrays

---

## 5) Index and Registration

### POST `/index/scan`
- Auth: required
- Body:

```json
{"path": "D:/dataset_root_or_parent"}
```

- Response:

```json
{"taskId": 1, "status": "running", "message": "Index task started"}
```

### GET `/index/status`
- Auth: required
- Query:
  - `task_id` (int)
- Response: task progress/status

### POST `/datasets/register`
- Auth: required
- Body:

```json
{
  "path": "D:/so101_cleandesk_2000-01-01",
  "dataset_name": "so101_cleandesk_2000-01-01",
  "robot": "so101",
  "task_type": "cleandesk",
  "materialize_assets": false,
  "overwrite_assets": false
}
```

- Notes:
  - Supports LeRobot v3 naming validation.
  - Creates index task after mapping and optional media materialization.

---

## 6) LeRobot Parse/Create Helpers

### POST `/datasets/parse-lerobot`
- Auth: required
- Parses a LeRobot dataset folder and returns structured metadata preview.

### POST `/datasets/create-lerobot`
- Auth: required
- Creates/registers a dataset using parse + optional indexing flow.

### POST `/datasets/upload-lerobot`
- Auth: required
- Alias flow similar to create endpoint for upload workflows.

---

## 7) Rerun Visualization

### POST `/items/{itemId}/visualize/rrd`
- Auth: required
- Body:

```json
{"format": "rrd"}
```

- Response:

```json
{
  "status": "completed",
  "downloadUrl": "http://127.0.0.1:8899/downloads/rrd/xxx.rrd",
  "filePath": "/downloads/rrd/xxx.rrd",
  "fileSize": 12345,
  "duration": 12.34
}
```

### POST `/items/{itemId}/visualize/server`
- Auth: required
- Body:

```json
{"mode": "distant", "ws_port": 9087, "web_port": 9090}
```

- Response includes `wsUrl` and `webUrl`.

---

## 8) Static Downloads

### GET `/downloads/rrd/{filename}`
- Auth: none
- Serves generated `.rrd` files from configured download directory.

### GET `/static/{filename}`
- Auth: none
- Serves backend static assets.

---

## 9) Source of Truth

- Runtime docs: `http://127.0.0.1:8899/docs`
- OpenAPI JSON: `http://127.0.0.1:8899/openapi.json`
- Backend implementation: `project/backend/app/main.py`
