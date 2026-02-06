import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..config import settings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _script_path() -> Path:
    return _repo_root() / "standalone_dataset_viz.py"


def _parse_episode_index(episode_id: Optional[str]) -> int:
    if not episode_id:
        return 0
    match = re.findall(r"(\d+)", episode_id)
    if not match:
        return 0
    return int(match[-1])


def resolve_dataset_root(item_path: Path) -> Path:
    for parent in [item_path] + list(item_path.parents):
        if parent.name == "episodes":
            return parent.parent
    return item_path.parent


def cleanup_old_rrd_files() -> None:
    output_dir = settings.rerun_downloads_dir
    if not output_dir.exists():
        return
    ttl_days = settings.rerun_config.get("auto_cleanup_days", 7)
    cutoff = datetime.utcnow() - timedelta(days=ttl_days)
    for file in output_dir.glob("*.rrd"):
        try:
            if datetime.utcfromtimestamp(file.stat().st_mtime) < cutoff:
                file.unlink(missing_ok=True)
        except Exception:
            continue


def generate_rrd(robot_name: str, episode_id: str, item_path: Path) -> Path:
    if not item_path.exists():
        raise RuntimeError(f"Episode path not found: {item_path}")
    output_dir = settings.rerun_downloads_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_index = _parse_episode_index(episode_id)
    dataset_root = resolve_dataset_root(item_path)
    script = _script_path()
    if not script.exists():
        raise RuntimeError(f"standalone_dataset_viz.py not found at {script}")

    cmd = [
        sys.executable,
        str(script),
        "--repo-id",
        robot_name,
        "--episode-index",
        str(episode_index),
        "--root",
        str(dataset_root),
        "--local",
        "--save",
        "1",
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(settings.rerun_config.get("batch_size", 32)),
        "--num-workers",
        str(settings.rerun_config.get("num_workers", 4)),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Failed to generate .rrd")

    filename = f"{robot_name.replace('/', '_')}_episode_{episode_index}.rrd"
    return output_dir / filename


def start_rerun_server(robot_name: str, episode_id: str, item_path: Path, ws_port: int, web_port: int) -> None:
    if not item_path.exists():
        raise RuntimeError(f"Episode path not found: {item_path}")
    dataset_root = resolve_dataset_root(item_path)
    script = _script_path()
    if not script.exists():
        raise RuntimeError(f"standalone_dataset_viz.py not found at {script}")

    cmd = [
        sys.executable,
        str(script),
        "--repo-id",
        robot_name,
        "--episode-index",
        str(_parse_episode_index(episode_id)),
        "--root",
        str(dataset_root),
        "--local",
        "--mode",
        "distant",
        "--ws-port",
        str(ws_port),
        "--web-port",
        str(web_port),
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
