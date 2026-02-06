from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


def is_v3_dataset_root(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file() and (root / "data").is_dir()


@lru_cache(maxsize=16)
def _load_info_cached(root_str: str) -> dict:
    root = Path(root_str)
    info_path = root / "meta" / "info.json"
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_info(root: Path) -> dict:
    return _load_info_cached(str(root))


@lru_cache(maxsize=16)
def _load_episodes_cached(root_str: str) -> pd.DataFrame:
    root = Path(root_str)
    episodes_dir = root / "meta" / "episodes"
    frames = []
    if episodes_dir.is_dir():
        for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
            frames.append(pd.read_parquet(parquet_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_episodes(root: Path) -> pd.DataFrame:
    return _load_episodes_cached(str(root))


def parse_episode_index(episode_id: Optional[str]) -> Optional[int]:
    if not episode_id:
        return None
    match = re.search(r"(\d+)$", episode_id)
    if match:
        return int(match.group(1))
    return None


def get_episode_row(root: Path, episode_index: int) -> Optional[dict]:
    df = load_episodes(root)
    if df.empty:
        return None
    row = df.loc[df["episode_index"] == episode_index]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_camera_keys(info: dict) -> list[str]:
    features = info.get("features", {})
    keys = []
    for key, meta in features.items():
        if isinstance(meta, dict) and meta.get("dtype") == "video":
            keys.append(key)
    return keys


def get_camera_shape(info: dict, camera_key: str) -> tuple[Optional[int], Optional[int]]:
    feature = info.get("features", {}).get(camera_key, {})
    info_block = feature.get("info", {}) if isinstance(feature, dict) else {}
    width = info_block.get("video.width")
    height = info_block.get("video.height")
    return width, height


def _format_path(template: str, **kwargs) -> str:
    return template.format(**kwargs)


def build_data_path(root: Path, info: dict, chunk_index: int, file_index: int) -> Path:
    template = info.get("data_path")
    rel = _format_path(template, chunk_index=int(chunk_index), file_index=int(file_index))
    return root / rel


def build_video_path(root: Path, info: dict, camera_key: str, chunk_index: int, file_index: int) -> Path:
    template = info.get("video_path")
    rel = _format_path(
        template,
        video_key=camera_key,
        chunk_index=int(chunk_index),
        file_index=int(file_index),
    )
    return root / rel


def load_timeseries(
    root: Path,
    episode_index: int,
    max_points: int = 500,
) -> Optional[dict]:
    info = load_info(root)
    row = get_episode_row(root, episode_index)
    if not row:
        return None
    data_path = build_data_path(root, info, row["data/chunk_index"], row["data/file_index"])
    if not data_path.exists():
        return None
    df = pd.read_parquet(data_path)
    start = int(row.get("dataset_from_index", 0))
    end = int(row.get("dataset_to_index", len(df)))
    df = df.iloc[start:end]
    if df.empty:
        return None

    timestamps = df["timestamp"].tolist() if "timestamp" in df.columns else list(range(len(df)))
    action = df["action"].tolist() if "action" in df.columns else []
    state = df["observation.state"].tolist() if "observation.state" in df.columns else []

    if max_points and len(timestamps) > max_points:
        stride = max(1, int(len(timestamps) / max_points))
        timestamps = timestamps[::stride]
        action = action[::stride]
        state = state[::stride]

    action_names = info.get("features", {}).get("action", {}).get("names") or []
    state_names = info.get("features", {}).get("observation.state", {}).get("names") or []

    return {
        "timestamps": timestamps,
        "action": [arr.tolist() for arr in action],
        "state": [arr.tolist() for arr in state],
        "actionNames": action_names,
        "stateNames": state_names,
    }


def get_video_clip(
    root: Path,
    episode_index: int,
    camera_key: str,
) -> Optional[dict]:
    info = load_info(root)
    row = get_episode_row(root, episode_index)
    if not row:
        return None
    chunk_key = f"videos/{camera_key}/chunk_index"
    file_key = f"videos/{camera_key}/file_index"
    start_key = f"videos/{camera_key}/from_timestamp"
    end_key = f"videos/{camera_key}/to_timestamp"
    if chunk_key not in row or file_key not in row:
        return None
    video_path = build_video_path(root, info, camera_key, row[chunk_key], row[file_key])
    start = float(row.get(start_key, 0.0) or 0.0)
    end = float(row.get(end_key, 0.0) or 0.0)
    duration = max(end - start, 0.0)
    return {
        "video_path": video_path,
        "start_time": start,
        "duration": duration,
        "fps": info.get("fps", 30.0),
    }
