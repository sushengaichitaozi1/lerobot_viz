import subprocess
import threading
from pathlib import Path
from shutil import which
from typing import Generator, Optional

from fastapi.responses import StreamingResponse

from ..config import REPO_ROOT, settings


_STREAM_SEMAPHORE = threading.BoundedSemaphore(settings.stream_config.get("max_concurrent_streams", 5))


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
    raise FileNotFoundError(
        "FFmpeg not found. Install ffmpeg or set FFMPEG_PATH to the ffmpeg executable."
    )


def _build_ffmpeg_command(
    input_path: str,
    is_video: bool,
    fps: Optional[float],
    resolution: Optional[dict] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> list[str]:
    enc = settings.video_encoding
    ffmpeg = _resolve_ffmpeg()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start_time is not None:
        cmd += ["-ss", f"{start_time:.3f}"]
    if fps and not is_video:
        cmd += ["-framerate", str(fps)]
    cmd += ["-i", input_path]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    if resolution:
        cmd += ["-vf", f"scale={resolution['width']}:{resolution['height']}"]
    cmd += ["-c:v", enc.get("codec", "libx264")]
    if enc.get("preset"):
        cmd += ["-preset", enc["preset"]]
    if enc.get("crf") is not None:
        cmd += ["-crf", str(enc["crf"])]
    if enc.get("pix_fmt"):
        cmd += ["-pix_fmt", enc["pix_fmt"]]
    if enc.get("movflags"):
        cmd += ["-movflags", enc["movflags"]]
    if enc.get("tune"):
        cmd += ["-tune", enc["tune"]]
    if enc.get("g") is not None:
        cmd += ["-g", str(enc["g"])]
    if enc.get("bframes") is not None:
        cmd += ["-bf", str(enc["bframes"])]
    cmd += ["-f", "mp4", "pipe:1"]
    return cmd


def stream_video(
    source_path: Path,
    file_pattern: Optional[str],
    fps: Optional[float],
    resolution: Optional[dict] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    status_code: int = 200,
    headers: Optional[dict] = None,
) -> StreamingResponse:
    is_video = source_path.is_file()
    input_path = str(source_path if is_video else source_path / (file_pattern or "%06d.png"))
    cmd = _build_ffmpeg_command(input_path, is_video, fps, resolution, start_time, duration)

    def iterator() -> Generator[bytes, None, None]:
        _STREAM_SEMAPHORE.acquire()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while True:
                chunk = process.stdout.read(8192)
                if not chunk:
                    break
                yield chunk
        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except Exception:
                process.kill()
            _STREAM_SEMAPHORE.release()

    return StreamingResponse(iterator(), media_type="video/mp4", status_code=status_code, headers=headers or {})
