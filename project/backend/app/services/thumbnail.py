from pathlib import Path
import subprocess
from shutil import which

from fastapi import HTTPException, Response

from . import lerobot_v3
from ..config import REPO_ROOT, settings


def resolve_thumbnail_path(image_dir: Path, file_pattern: str, frame: int) -> Path:
    if image_dir.is_file():
        thumb_dir = image_dir.parent.parent / "thumbnails"
        stem = image_dir.stem
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = thumb_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise HTTPException(status_code=404, detail="Thumbnail not available for video source")
    if frame < 0:
        frame = 0
    try:
        filename = file_pattern % frame
    except Exception:
        filename = file_pattern.replace("%d", str(frame)).replace("%06d", f"{frame:06d}")
    path = image_dir / filename
    if not path.exists() and frame != 0:
        try:
            filename = file_pattern % 0
        except Exception:
            filename = file_pattern.replace("%d", "0").replace("%06d", "000000")
        path = image_dir / filename
    if not path.exists():
        candidates = sorted(
            [
                p
                for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        if candidates:
            return candidates[0]
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return path


def _resolve_ffmpeg() -> str:
    if settings.ffmpeg_path:
        candidate = Path(settings.ffmpeg_path)
        if candidate.is_file():
            return str(candidate)
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
    raise HTTPException(status_code=500, detail="FFmpeg not found")


def render_video_thumbnail(video_path: Path, timestamp: float) -> Response:
    ffmpeg = _resolve_ffmpeg()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        output = subprocess.check_output(cmd)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
    return Response(content=output, media_type="image/jpeg")
