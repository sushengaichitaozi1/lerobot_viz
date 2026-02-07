from __future__ import annotations

import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from shutil import which
from typing import Any, Optional

import pandas as pd

from . import lerobot_v3
from ..config import REPO_ROOT, settings


DATASET_NAME_PATTERN = re.compile(
    r"^(?P<robot>[^_]+)_(?P<task>.+)_(?P<date>\d{4}-\d{2}-\d{2})$"
)

ROBOT_CATALOG: dict[str, dict[str, str]] = {
    "so100": {
        "name": "SO100 Robot",
        "model": "SO100",
        "description": "SO100 manipulation robot",
    },
    "so101": {
        "name": "SO101 Robot",
        "model": "SO101",
        "description": "SO101 manipulation robot",
    },
    "so101_follower": {
        "name": "SO101 Follower Robot",
        "model": "SO101",
        "description": "SO101 follower robot",
    },
    "aloha": {
        "name": "ALOHA Robot",
        "model": "ALOHA",
        "description": "ALOHA bimanual robot",
    },
    "aloha_mobile": {
        "name": "ALOHA Mobile Robot",
        "model": "ALOHA Mobile",
        "description": "ALOHA mobile manipulator",
    },
}


def _to_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            value = value.tolist()
        except Exception:
            return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _to_scalar(value[0])
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    value = _to_scalar(value)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    value = _to_scalar(value)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _to_relative_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _sum_directory_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def infer_dataset_name(dataset_path: str | Path, dataset_name: Optional[str] = None) -> str:
    candidate = (dataset_name or "").strip()
    if candidate:
        return candidate
    inferred = Path(dataset_path).expanduser().name.strip()
    if inferred:
        return inferred
    raise ValueError("dataset_name is required when dataset path does not contain a folder name.")


def _resolve_ffmpeg() -> str:
    if settings.ffmpeg_path:
        configured = Path(settings.ffmpeg_path)
        if configured.is_file():
            return str(configured)

    system_ffmpeg = which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    tools_dir = REPO_ROOT / "project" / "tools" / "ffmpeg"
    if tools_dir.exists():
        for candidate in tools_dir.glob("**/bin/ffmpeg.exe"):
            if candidate.is_file():
                return str(candidate)
        for candidate in tools_dir.glob("**/ffmpeg.exe"):
            if candidate.is_file():
                return str(candidate)

    raise RuntimeError("FFmpeg not found. Install ffmpeg or set FFMPEG_PATH.")


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "").strip()
    raise RuntimeError(detail or "Failed to run ffmpeg command.")


def _materialize_video_assets(
    video_path: Path,
    cut_file_path: Path,
    thumbnail_path: Path,
    from_seconds: float,
    duration_seconds: float,
    overwrite_assets: bool,
) -> None:
    if not video_path.exists():
        return

    ffmpeg = _resolve_ffmpeg()
    cut_file_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite_assets:
        if cut_file_path.exists():
            cut_file_path.unlink()
        if thumbnail_path.exists():
            thumbnail_path.unlink()

    if not cut_file_path.exists():
        encoding = settings.video_encoding or {}
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(from_seconds, 0.0):.6f}",
            "-i",
            str(video_path),
        ]
        if duration_seconds > 0:
            cmd += ["-t", f"{duration_seconds:.6f}"]
        cmd += ["-c:v", str(encoding.get("codec", "libx264"))]
        if encoding.get("preset"):
            cmd += ["-preset", str(encoding["preset"])]
        if encoding.get("crf") is not None:
            cmd += ["-crf", str(encoding["crf"])]
        if encoding.get("pix_fmt"):
            cmd += ["-pix_fmt", str(encoding["pix_fmt"])]
        cmd += ["-an", str(cut_file_path)]
        _run_ffmpeg(cmd)

    if not thumbnail_path.exists():
        source = cut_file_path if cut_file_path.exists() else video_path
        start = 0.0 if source == cut_file_path else max(from_seconds, 0.0)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(thumbnail_path),
        ]
        _run_ffmpeg(cmd)


def _parse_task_from_dataset_name(dataset_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    match = DATASET_NAME_PATTERN.match(dataset_name)
    if not match:
        return None, None, None
    return (
        match.group("robot").strip(),
        match.group("task").strip(),
        match.group("date").strip(),
    )


def validate_dataset_name(dataset_name: str) -> tuple[bool, str | None]:
    robot_name, task_name, date_str = _parse_task_from_dataset_name(dataset_name)
    if not robot_name or not task_name or not date_str:
        return False, "dataset_name must match {robot_type}_{task_name}_{YYYY-MM-DD}."
    try:
        date.fromisoformat(date_str)
    except ValueError:
        return False, "dataset_name contains an invalid date. Expected YYYY-MM-DD."
    return True, None


def parse_dataset_name(dataset_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    return _parse_task_from_dataset_name(dataset_name)


def _extract_task_description(value: Any) -> Optional[str]:
    value = _to_scalar(value)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _extract_task_description(item)
            if text:
                return text
    return None


def _read_tasks(root: Path, episodes_df: pd.DataFrame) -> list[dict[str, Any]]:
    tasks_path = root / "meta" / "tasks.parquet"
    task_map: dict[int, Optional[str]] = {}

    if tasks_path.exists():
        try:
            tasks_df = pd.read_parquet(tasks_path)
            for idx, row in tasks_df.iterrows():
                task_index = _safe_int(row.get("task_index"), idx)
                description = _extract_task_description(
                    row.get("task_description") or row.get("task") or row.get("description")
                )
                task_map[task_index] = description
        except Exception:
            pass

    if not episodes_df.empty and "tasks" in episodes_df.columns:
        for _, row in episodes_df.iterrows():
            task_index = _safe_int(row.get("task_index"), _safe_int(row.get("stats/task_index/min"), 0))
            description = _extract_task_description(row.get("tasks"))
            if task_index not in task_map or not task_map[task_index]:
                if description:
                    task_map[task_index] = description

    if not task_map and not episodes_df.empty:
        for task_index in sorted(
            {
                _safe_int(row.get("task_index"), _safe_int(row.get("stats/task_index/min"), 0))
                for _, row in episodes_df.iterrows()
            }
        ):
            task_map[task_index] = None

    tasks: list[dict[str, Any]] = []
    for task_index in sorted(task_map):
        description = task_map[task_index] or f"task_{task_index}"
        tasks.append(
            {
                "task_index": task_index,
                "task_description": description,
            }
        )
    return tasks


def _dataset_data_path(root: Path, info: dict[str, Any], chunk_index: int, file_index: int) -> Path:
    template = info.get("data_path")
    if isinstance(template, str) and template:
        rel = template.format(chunk_index=chunk_index, file_index=file_index)
    else:
        rel = f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    return root / rel


def _dataset_video_path(
    root: Path,
    info: dict[str, Any],
    camera_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    template = info.get("video_path")
    if isinstance(template, str) and template:
        rel = template.format(video_key=camera_key, chunk_index=chunk_index, file_index=file_index)
    else:
        rel = f"videos/{camera_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    return root / rel


def _read_parquet_metadata(parquet_path: Path, fallback_rows: int) -> dict[str, Any]:
    if not parquet_path.exists():
        return {
            "num_rows": fallback_rows,
            "num_columns": 0,
            "columns": [],
        }

    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        columns = list(parquet_file.schema.names)
        return {
            "num_rows": int(parquet_file.metadata.num_rows) if parquet_file.metadata else fallback_rows,
            "num_columns": len(columns),
            "columns": columns,
        }
    except Exception:
        try:
            df = pd.read_parquet(parquet_path)
            return {
                "num_rows": int(len(df.index)),
                "num_columns": int(len(df.columns)),
                "columns": [str(col) for col in df.columns],
            }
        except Exception:
            return {
                "num_rows": fallback_rows,
                "num_columns": 0,
                "columns": [],
            }


def _format_capture_time(value: Any) -> Optional[str]:
    value = _to_scalar(value)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _camera_alias(camera_key: str) -> str:
    return camera_key.split(".")[-1].replace(" ", "_").lower()


def _resolve_task_index(row: pd.Series) -> int:
    if "task_index" in row:
        return _safe_int(row.get("task_index"), 0)
    if "stats/task_index/min" in row:
        return _safe_int(row.get("stats/task_index/min"), 0)
    return 0


def _build_episode_entry(
    root: Path,
    info: dict[str, Any],
    row: pd.Series,
    camera_keys: list[str],
    materialize_assets: bool,
    overwrite_assets: bool,
) -> dict[str, Any]:
    episode_index = _safe_int(row.get("episode_index"), 0)
    task_index = _resolve_task_index(row)
    length = _safe_int(row.get("length"), 0)
    data_chunk_index = _safe_int(row.get("data/chunk_index"), _safe_int(row.get("data_chunk_index"), 0))
    data_file_index = _safe_int(row.get("data/file_index"), _safe_int(row.get("data_file_index"), 0))

    fps = _safe_float(info.get("fps"), 30.0)
    capture_time = _format_capture_time(row.get("capture_time"))

    parquet_path = _dataset_data_path(root, info, data_chunk_index, data_file_index)
    parquet_rel = _to_relative_path(parquet_path, root)
    parquet_size = parquet_path.stat().st_size if parquet_path.exists() else 0
    parquet_meta = _read_parquet_metadata(parquet_path, length)

    video_entries: list[dict[str, Any]] = []
    image_entries: list[dict[str, Any]] = []

    for camera_key in camera_keys:
        chunk_col = f"videos/{camera_key}/chunk_index"
        file_col = f"videos/{camera_key}/file_index"
        if chunk_col not in row or file_col not in row:
            continue

        video_chunk_index = _safe_int(row.get(chunk_col), data_chunk_index)
        video_file_index = _safe_int(row.get(file_col), data_file_index)

        video_path = _dataset_video_path(root, info, camera_key, video_chunk_index, video_file_index)
        video_rel = _to_relative_path(video_path, root)
        size_bytes = video_path.stat().st_size if video_path.exists() else 0

        from_ts = _safe_float(row.get(f"videos/{camera_key}/from_timestamp"), 0.0)
        to_ts = _safe_float(row.get(f"videos/{camera_key}/to_timestamp"), 0.0)
        duration_seconds = max(to_ts - from_ts, 0.0)
        if duration_seconds <= 0 and fps > 0:
            duration_seconds = length / fps

        width, height = lerobot_v3.get_camera_shape(info, camera_key)
        feature_info = info.get("features", {}).get(camera_key, {}).get("info", {})
        codec = feature_info.get("video.codec") if isinstance(feature_info, dict) else None

        camera_alias = _camera_alias(camera_key)
        cut_file_path = f"videos_cut/episode_{episode_index:03d}_{camera_alias}.mp4"
        thumbnail_path = f"thumbnails/episode_{episode_index:03d}_{camera_alias}.jpg"
        cut_file_abs = root / cut_file_path
        thumbnail_abs = root / thumbnail_path

        if materialize_assets:
            _materialize_video_assets(
                video_path=video_path,
                cut_file_path=cut_file_abs,
                thumbnail_path=thumbnail_abs,
                from_seconds=from_ts,
                duration_seconds=duration_seconds,
                overwrite_assets=overwrite_assets,
            )

        effective_video_path = cut_file_abs if cut_file_abs.exists() else video_path

        video_entries.append(
            {
                "type": "video",
                "feature_name": camera_key,
                "original_file_path": video_rel,
                "cut_file_path": cut_file_path,
                "thumbnail_path": thumbnail_path,
                "size_bytes": effective_video_path.stat().st_size if effective_video_path.exists() else size_bytes,
                "mime_type": "video/mp4",
                "metadata": {
                    "width": width,
                    "height": height,
                    "duration_seconds": duration_seconds,
                    "fps": fps,
                    "codec": codec,
                    "frame_count": length,
                },
            }
        )

        image_entries.append(
            {
                "type": "image",
                "feature_name": camera_key,
                "file_path": thumbnail_path,
                "size_bytes": thumbnail_abs.stat().st_size if thumbnail_abs.exists() else 0,
                "mime_type": "image/jpeg",
                "metadata": {
                    "width": width,
                    "height": height,
                    "format": "jpeg",
                },
            }
        )

    return {
        "episode_index": episode_index,
        "task_index": task_index,
        "length": length,
        "data_chunk_index": data_chunk_index,
        "data_file_index": data_file_index,
        "capture_time": capture_time,
        "data_files": {
            "video": video_entries,
            "images": image_entries,
            "parquet": [
                {
                    "type": "parquet",
                    "feature_name": "data",
                    "file_path": parquet_rel,
                    "size_bytes": parquet_size,
                    "mime_type": "application/octet-stream",
                    "metadata": parquet_meta,
                }
            ],
        },
    }


def _build_robot_info(robot_type: str) -> dict[str, Any]:
    info = ROBOT_CATALOG.get(robot_type, {})
    return {
        "robot_id": robot_type,
        "robot_type": robot_type,
        "name": info.get("name", robot_type.replace("_", " ").title()),
        "model": info.get("model", robot_type.split("_")[0].upper()),
        "description": info.get("description", f"{robot_type} robot"),
        "status": "active",
    }


def parse_dataset(
    dataset_path: str | Path,
    dataset_name: Optional[str] = None,
    materialize_assets: bool = False,
    overwrite_assets: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    if not lerobot_v3.is_v3_dataset_root(root):
        raise ValueError("Only LeRobot v3.0 dataset roots are supported (missing meta/info.json or data/).")

    dataset_name = infer_dataset_name(root, dataset_name)
    is_valid_name, error_message = validate_dataset_name(dataset_name)
    if not is_valid_name:
        raise ValueError(error_message)

    parsed_robot, parsed_task, _ = parse_dataset_name(dataset_name)

    info = lerobot_v3.load_info(root)
    episodes_df = lerobot_v3.load_episodes(root)
    camera_keys = lerobot_v3.get_camera_keys(info)

    tasks = _read_tasks(root, episodes_df)
    episodes = [
        _build_episode_entry(
            root,
            info,
            row,
            camera_keys,
            materialize_assets=materialize_assets,
            overwrite_assets=overwrite_assets,
        )
        for _, row in episodes_df.iterrows()
    ]

    robot_type = str(info.get("robot_type") or parsed_robot or root.name)
    fps = _safe_float(info.get("fps"), 30.0)
    total_frames = sum(_safe_int(ep.get("length"), 0) for ep in episodes)

    dataset_info = {
        "codebase_version": info.get("codebase_version", "v3.0"),
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_size_bytes": _sum_directory_size(root),
    }

    if parsed_task and tasks and all(str(task.get("task_description", "")).startswith("task_") for task in tasks):
        tasks[0]["task_description"] = parsed_task

    message = "Parse succeeded"
    if materialize_assets:
        message = "Parse succeeded and media assets generated"

    return {
        "success": True,
        "message": message,
        "dataset_info": dataset_info,
        "robot_info": _build_robot_info(robot_type),
        "tasks": tasks,
        "episodes": episodes,
    }
