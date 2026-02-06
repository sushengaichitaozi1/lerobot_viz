import os
from pathlib import Path
from typing import IO

from fastapi import HTTPException

from .config import settings


def ensure_data_root() -> Path:
    settings.data_root.mkdir(parents=True, exist_ok=True)
    return settings.data_root


def ensure_dataset_dir(dataset_id: int) -> Path:
    root = ensure_data_root()
    dataset_dir = root / f"dataset-{dataset_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def safe_relative_path(path: Path) -> str:
    root = ensure_data_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")


def safe_absolute_path(relative_path: str) -> Path:
    root = ensure_data_root()
    candidate = (root / relative_path).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return candidate


def save_stream_to_path(source: IO[bytes], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with destination.open("wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            target.write(chunk)
    return size


def guess_content_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext in {"mp4", "mov", "webm"}:
        return f"video/{ext if ext != 'mov' else 'quicktime'}"
    if ext in {"png", "jpg", "jpeg"}:
        return f"image/{'jpeg' if ext == 'jpg' else ext}"
    return "application/octet-stream"
