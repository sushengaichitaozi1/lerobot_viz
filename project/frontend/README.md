# Frontend Guide

## Location
- Entry page: `project/frontend/index.html`
- Compatibility redirect: `project/111.html` -> `project/frontend/index.html`

## Stack
- Vue 3 (CDN)
- Tailwind CSS (CDN)
- Font Awesome (CDN)

## Responsibilities
- Auth login and token handling.
- Dataset registration and index progress polling.
- Dataset and episode list browsing.
- Episode detail panel (video + timeseries).
- Dataset and episode update/delete actions.
- Rerun visualization actions.

## API Base
- Default: `http://localhost:8899`
- Configurable from login dialog.

## Start Frontend
From repository root:

```powershell
cd project
python -m http.server 5090
```

Open:
- `http://127.0.0.1:5090/frontend/index.html`
- (or compatibility URL) `http://127.0.0.1:5090/111.html`

## Main Frontend Conventions
- `datasetForm` stores registration modal state.
- `loading.datasets` and `loading.items` are separated.
- Update actions use unified naming:
  - `updateDataset(...)`
  - `updateItem(...)`
  - `updateSelectedItem(...)`
- Reusable helper for JSON requests:
  - `buildJsonRequestOptions(payload)`
- Dataset request fallback helper:
  - `datasetRequestWithFallback(...)`

## Notes
- This is currently a single-file SPA for delivery speed.
- If needed, it can be split into `frontend/src` modules later (components, stores, api client).
