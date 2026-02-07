from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import which
from typing import Optional, Callable

from sqlalchemy.orm import Session

from .config import REPO_ROOT, settings
from . import models
from .services import lerobot_v3


logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
DATASET_NAME_PATTERN = re.compile(
    r"^(?P<robot>[^_]+)_(?P<task>.+)_(?P<date>\d{4}-\d{2}-\d{2})$"
)


@dataclass(frozen=True)
class EpisodeLocation:
    robot_name: str
    task_type: str
    episode_path: Path


@dataclass
class EpisodeMeta:
    episode_id: str
    file_path: Path
    total_frames: int
    fps: float
    duration_s: float
    width: Optional[int]
    height: Optional[int]
    image_count: int
    total_size_bytes: int
    cameras: list[dict]


def resolve_robots_root(root: Path) -> Path:
    if (root / "robots").is_dir():
        return root / "robots"
    return root


def _discover_from_task(task_dir: Path) -> list[EpisodeLocation]:
    episodes: list[EpisodeLocation] = []
    episodes_dir = task_dir / "episodes"
    if not episodes_dir.is_dir():
        return episodes
    if task_dir.parent.name == "tasks":
        robot_dir = task_dir.parent.parent
        task_type = task_dir.name
    else:
        robot_dir = task_dir
        task_type = "default"
    for episode_entry in episodes_dir.iterdir():
        if episode_entry.is_dir() or (
            episode_entry.is_file() and episode_entry.suffix.lower() in VIDEO_EXTENSIONS
        ):
            episodes.append(
                EpisodeLocation(
                    robot_name=robot_dir.name,
                    task_type=task_type,
                    episode_path=episode_entry,
                )
            )
    return episodes


def _discover_from_robot(robot_dir: Path) -> list[EpisodeLocation]:
    episodes: list[EpisodeLocation] = []
    tasks_dir = robot_dir / "tasks"
    if not tasks_dir.is_dir():
        return episodes
    for task_dir in tasks_dir.iterdir():
        if task_dir.is_dir():
            episodes.extend(_discover_from_task(task_dir))
    return episodes


def discover_episodes(root: Path) -> list[EpisodeLocation]:
    if (root / "tasks").is_dir():
        return _discover_from_robot(root)
    if (root / "episodes").is_dir():
        return _discover_from_task(root)

    episodes: list[EpisodeLocation] = []
    robots_root = resolve_robots_root(root)
    if not robots_root.exists():
        return episodes
    for robot_dir in robots_root.iterdir():
        if robot_dir.is_dir():
            if (robot_dir / "tasks").is_dir():
                episodes.extend(_discover_from_robot(robot_dir))
            elif (robot_dir / "episodes").is_dir():
                episodes.extend(_discover_from_task(robot_dir))
    return episodes


def _read_episode_info(episode_path: Path) -> dict:
    info_path = episode_path / "info.json"
    if not info_path.exists():
        info_path = episode_path / "meta" / "info.json"
    if info_path.exists():
        try:
            return json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse info.json: %s", info_path)
    return {}


def _load_category_map() -> dict:
    path = settings.category_map_path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read category map: %s", path)
        return {}
    datasets = data.get("datasets") if isinstance(data, dict) else None
    return datasets if isinstance(datasets, dict) else {}


def _parse_dataset_name(dataset_name: str) -> tuple[Optional[str], Optional[str]]:
    match = DATASET_NAME_PATTERN.match(dataset_name)
    if not match:
        return None, None

    robot_name = match.group("robot").strip()
    task_name = match.group("task").strip()
    date_str = match.group("date")
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None, None

    if not robot_name or not task_name:
        return None, None
    return robot_name, task_name


def _infer_storage_path(file_path: Path) -> Path:
    path = file_path if file_path.is_dir() else file_path.parent
    if lerobot_v3.is_v3_dataset_root(path):
        return path

    if "episodes" in path.parts:
        idx = path.parts.index("episodes")
        if idx > 0:
            return Path(*path.parts[:idx])
    return path


def _pick_primary_camera(cameras: list[dict]) -> Optional[dict]:
    if not cameras:
        return None
    preferred = settings.primary_camera_key
    for camera in cameras:
        if camera["camera_key"] == preferred:
            return camera
    for camera in cameras:
        if "top" in camera["camera_key"].lower():
            return camera
    return cameras[0]


def _build_file_pattern(filename: str) -> str:
    stem, ext = Path(filename).stem, Path(filename).suffix
    match = re.search(r"(\d+)", stem)
    if match:
        prefix = stem[: match.start()]
        suffix = stem[match.end() :]
        width = len(match.group(1))
        return f"{prefix}%0{width}d{suffix}{ext}"
    return f"%06d{ext}"


def _iter_images(camera_dir: Path) -> list[Path]:
    return sorted(
        [path for path in camera_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def _read_image_size(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image
    except Exception:
        return None, None
    try:
        with Image.open(path) as img:
            width, height = img.size
            return width, height
    except Exception:
        return None, None


def _resolve_ffprobe() -> Optional[str]:
    if settings.ffmpeg_path:
        candidate = Path(settings.ffmpeg_path).with_name("ffprobe.exe")
        if candidate.is_file():
            return str(candidate)
    system_ffprobe = which("ffprobe")
    if system_ffprobe:
        return system_ffprobe
    tools_dir = REPO_ROOT / "project" / "tools" / "ffmpeg"
    if tools_dir.exists():
        for candidate in tools_dir.glob("**/bin/ffprobe.exe"):
            if candidate.is_file():
                return str(candidate)
        for candidate in tools_dir.glob("**/ffprobe.exe"):
            if candidate.is_file():
                return str(candidate)
    return None


def _probe_video(path: Path) -> tuple[Optional[int], Optional[int], Optional[float], Optional[int], Optional[float]]:
    ffprobe = _resolve_ffprobe()
    if not ffprobe:
        return None, None, None, None, None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams:
            return None, None, None, None, None
        stream = streams[0]
        width = stream.get("width")
        height = stream.get("height")
        fps = None
        avg_rate = stream.get("avg_frame_rate")
        if isinstance(avg_rate, str) and "/" in avg_rate:
            num, den = avg_rate.split("/", 1)
            try:
                fps = float(num) / float(den) if float(den) != 0 else None
            except Exception:
                fps = None
        elif avg_rate:
            try:
                fps = float(avg_rate)
            except Exception:
                fps = None
        frame_count = None
        if stream.get("nb_frames"):
            try:
                frame_count = int(float(stream["nb_frames"]))
            except Exception:
                frame_count = None
        duration = None
        if stream.get("duration"):
            try:
                duration = float(stream["duration"])
            except Exception:
                duration = None
        return width, height, fps, frame_count, duration
    except Exception:
        return None, None, None, None, None


def _parse_video_episode(episode_path: Path) -> EpisodeMeta:
    width, height, fps, frame_count, duration = _probe_video(episode_path)
    if fps is None:
        fps = 30.0
    if duration is None and frame_count is not None and fps:
        duration = frame_count / fps
    if frame_count is None and duration is not None and fps:
        try:
            frame_count = int(duration * fps)
        except Exception:
            frame_count = None

    size_bytes = episode_path.stat().st_size if episode_path.exists() else 0
    camera_key = settings.primary_camera_key
    cameras = [
        {
            "camera_key": camera_key,
            "display_name": camera_key.replace("_", " ").title(),
            "image_path": str(episode_path),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "file_pattern": None,
        }
    ]
    total_frames = frame_count or 0
    duration_s = duration if duration is not None else (total_frames / fps if fps else 0.0)
    return EpisodeMeta(
        episode_id=episode_path.stem,
        file_path=episode_path,
        total_frames=total_frames,
        fps=fps,
        duration_s=duration_s,
        width=width,
        height=height,
        image_count=total_frames,
        total_size_bytes=size_bytes,
        cameras=cameras,
    )


def parse_episode(episode_path: Path) -> EpisodeMeta:
    if episode_path.is_file() and episode_path.suffix.lower() in VIDEO_EXTENSIONS:
        return _parse_video_episode(episode_path)

    info = _read_episode_info(episode_path)
    fps = info.get("fps") or info.get("metadata", {}).get("fps") or 30.0
    try:
        fps = float(fps)
    except Exception:
        fps = 30.0

    cameras_dir = episode_path / "observation_images"
    cameras: list[dict] = []
    total_size = 0
    total_images = 0

    if cameras_dir.is_dir():
        for camera_dir in cameras_dir.iterdir():
            if not camera_dir.is_dir():
                continue
            images = _iter_images(camera_dir)
            if not images:
                continue
            frame_count = len(images)
            width, height = _read_image_size(images[0])
            file_pattern = _build_file_pattern(images[0].name)
            size_sum = sum(img.stat().st_size for img in images)
            total_size += size_sum
            total_images += frame_count
            cameras.append(
                {
                    "camera_key": camera_dir.name,
                    "display_name": camera_dir.name.replace("_", " ").title(),
                    "image_path": str(camera_dir),
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "file_pattern": file_pattern,
                }
            )

    primary_camera = _pick_primary_camera(cameras)
    total_frames = primary_camera["frame_count"] if primary_camera else 0
    duration_s = total_frames / fps if fps else 0.0
    width = primary_camera.get("width") if primary_camera else None
    height = primary_camera.get("height") if primary_camera else None

    return EpisodeMeta(
        episode_id=episode_path.name,
        file_path=episode_path,
        total_frames=total_frames,
        fps=fps,
        duration_s=duration_s,
        width=width,
        height=height,
        image_count=total_images,
        total_size_bytes=total_size,
        cameras=cameras,
    )


def _upsert_robot(db: Session, name: str) -> models.Robot:
    robot = db.query(models.Robot).filter(models.Robot.name == name).one_or_none()
    if robot:
        return robot
    robot = models.Robot(name=name, display_name=name)
    db.add(robot)
    db.flush()
    return robot


def _upsert_task_type(db: Session, robot: models.Robot, name: str) -> models.TaskType:
    task_type = (
        db.query(models.TaskType)
        .filter(models.TaskType.robot_id == robot.id, models.TaskType.name == name)
        .one_or_none()
    )
    if task_type:
        return task_type
    task_type = models.TaskType(name=name, display_name=name.replace("_", " ").title(), robot_id=robot.id)
    db.add(task_type)
    db.flush()
    return task_type


def _upsert_item(
    db: Session,
    robot: models.Robot,
    task_type: models.TaskType,
    meta: EpisodeMeta,
) -> models.Item:
    items_with_path = (
        db.query(models.Item)
        .filter(
            models.Item.file_path == str(meta.file_path),
            models.Item.episode_id == meta.episode_id,
        )
        .all()
    )
    item = items_with_path[0] if items_with_path else None
    if item:
        for extra in items_with_path[1:]:
            db.delete(extra)
    else:
        item = (
            db.query(models.Item)
            .filter(
                models.Item.robot_id == robot.id,
                models.Item.task_type_id == task_type.id,
                models.Item.episode_id == meta.episode_id,
            )
            .one_or_none()
        )
    if item is None:
        item = models.Item(
            robot_id=robot.id,
            task_type_id=task_type.id,
            episode_id=meta.episode_id,
            file_path=str(meta.file_path),
            storage_path=str(_infer_storage_path(meta.file_path)),
        )
        db.add(item)
        db.flush()
    else:
        item.robot_id = robot.id
        item.task_type_id = task_type.id
        item.episode_id = meta.episode_id

    item.file_path = str(meta.file_path)
    item.storage_path = str(_infer_storage_path(meta.file_path))
    item.total_frames = meta.total_frames
    item.fps = meta.fps
    item.duration_s = meta.duration_s
    item.width = meta.width
    item.height = meta.height
    item.image_count = meta.image_count
    item.total_size_bytes = meta.total_size_bytes
    item.camera_count = len(meta.cameras)
    item.index_status = "completed"
    item.index_error = None
    item.indexed_at = datetime.utcnow()
    item.deleted_at = None
    return item


def _refresh_cameras(db: Session, item: models.Item, cameras: list[dict]) -> None:
    existing = db.query(models.CameraInfo).filter(models.CameraInfo.item_id == item.id).all()
    for cam in existing:
        db.delete(cam)
    if existing:
        db.flush()

    seen_camera_keys: set[str] = set()
    for camera in cameras:
        camera_key = camera["camera_key"]
        if camera_key in seen_camera_keys:
            continue
        seen_camera_keys.add(camera_key)
        db.add(
            models.CameraInfo(
                item_id=item.id,
                camera_key=camera_key,
                display_name=camera["display_name"],
                image_path=camera["image_path"],
                frame_count=camera["frame_count"],
                width=camera["width"],
                height=camera["height"],
                file_pattern=camera["file_pattern"],
            )
        )


def index_dataset(
    db: Session,
    root: Path,
    task: Optional[models.IndexTask] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> int:
    category_map = _load_category_map()
    v3_roots: list[Path] = []
    if lerobot_v3.is_v3_dataset_root(root):
        v3_roots = [root]
    elif root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and lerobot_v3.is_v3_dataset_root(child):
                v3_roots.append(child)
    if v3_roots:
        total = 0
        for v3_root in v3_roots:
            episodes_df = lerobot_v3.load_episodes(v3_root)
            total += len(episodes_df.index)
        if task:
            task.total_episodes = total
            task.status = "running"
            task.started_at = datetime.utcnow()
            db.commit()

        processed = 0
        for v3_root in v3_roots:
            info = lerobot_v3.load_info(v3_root)
            episodes_df = lerobot_v3.load_episodes(v3_root)
            camera_keys = lerobot_v3.get_camera_keys(info)
            mapped = category_map.get(v3_root.name, {}) if category_map else {}
            parsed_robot, parsed_task = _parse_dataset_name(v3_root.name)
            robot_name = mapped.get("robot") or info.get("robot_type") or parsed_robot or v3_root.name
            task_name = mapped.get("task_type") or info.get("task_type") or parsed_task or "default"
            robot = _upsert_robot(db, robot_name)
            task_type = _upsert_task_type(db, robot, task_name)

            for _, row in episodes_df.iterrows():
                try:
                    episode_index = int(row.get("episode_index", 0))
                    length = int(row.get("length", 0))
                    fps = float(info.get("fps", 30.0))
                    duration_s = length / fps if fps else 0.0
                    width = height = None
                    if camera_keys:
                        width, height = lerobot_v3.get_camera_shape(info, camera_keys[0])
                    meta = EpisodeMeta(
                        episode_id=f"episode_{episode_index:06d}",
                        file_path=v3_root,
                        total_frames=length,
                        fps=fps,
                        duration_s=duration_s,
                        width=width,
                        height=height,
                        image_count=length * max(len(camera_keys), 1),
                        total_size_bytes=0,
                        cameras=[
                            {
                                "camera_key": key,
                                "display_name": key.replace("_", " "),
                                "image_path": str(v3_root),
                                "frame_count": length,
                                "width": lerobot_v3.get_camera_shape(info, key)[0],
                                "height": lerobot_v3.get_camera_shape(info, key)[1],
                                "file_pattern": None,
                            }
                            for key in camera_keys
                        ],
                    )
                    item = _upsert_item(db, robot, task_type, meta)
                    _refresh_cameras(db, item, meta.cameras)
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.exception("Failed to index v3 episode %s: %s", v3_root, exc)
                    if task:
                        db.add(
                            models.IndexLog(
                                task_id=task.id,
                                level="error",
                                message=f"{v3_root}: {exc}",
                            )
                        )
                        task.status = "running"
                        task.error_message = str(exc)
                        db.commit()

                processed += 1
                if task:
                    task.processed_episodes = processed
                    task.progress = int(processed / total * 100) if total else 100
                    db.commit()
                if progress_cb:
                    progress_cb(processed, total)

        if task:
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.progress = 100
            db.commit()
        return total

    episodes = discover_episodes(root)
    total = len(episodes)
    if task:
        task.total_episodes = total
        task.status = "running"
        task.started_at = datetime.utcnow()
        db.commit()

    processed = 0
    for location in episodes:
        try:
            mapped = category_map.get(location.robot_name, {}) if category_map else {}
            parsed_robot, parsed_task = _parse_dataset_name(location.robot_name)
            robot_name = mapped.get("robot") or parsed_robot or location.robot_name
            inferred_task = parsed_task if location.task_type == "default" else None
            task_name = mapped.get("task_type") or inferred_task or location.task_type
            robot = _upsert_robot(db, robot_name)
            task_type = _upsert_task_type(db, robot, task_name)
            meta = parse_episode(location.episode_path)
            item = _upsert_item(db, robot, task_type, meta)
            _refresh_cameras(db, item, meta.cameras)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.exception("Failed to index episode %s: %s", location.episode_path, exc)
            if task:
                db.add(
                    models.IndexLog(
                        task_id=task.id,
                        level="error",
                        message=f"{location.episode_path}: {exc}",
                    )
                )
                task.status = "running"
                task.error_message = str(exc)
                db.commit()
        processed += 1
        if task:
            task.processed_episodes = processed
            task.progress = int(processed / total * 100) if total else 100
            db.commit()
        if progress_cb:
            progress_cb(processed, total)

    if task:
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        task.progress = 100
        db.commit()
    return total
