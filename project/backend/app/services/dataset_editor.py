from __future__ import annotations

import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import psutil

from . import lerobot_v3


CRITICAL_PROCESS_NAMES = {
    "system",
    "registry",
    "smss.exe",
    "csrss.exe",
    "wininit.exe",
    "services.exe",
    "lsass.exe",
    "winlogon.exe",
}


def _clear_v3_caches() -> None:
    for cache_name in ("_load_info_cached", "_load_episodes_cached"):
        cache_fn = getattr(lerobot_v3, cache_name, None)
        if cache_fn is not None and hasattr(cache_fn, "cache_clear"):
            cache_fn.cache_clear()


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _normalize_path_str(path: str | Path) -> str:
    return str(path).replace("/", "\\").rstrip("\\").lower()


def _path_matches(candidate: str | None, root_norm: str) -> bool:
    if not candidate:
        return False
    candidate_norm = _normalize_path_str(candidate)
    return candidate_norm == root_norm or candidate_norm.startswith(root_norm + "\\")


def _process_holds_dataset(proc: psutil.Process, root_norm: str) -> bool:
    try:
        cwd = proc.cwd()
        if _path_matches(cwd, root_norm):
            return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, OSError):
        pass

    try:
        cmdline = " ".join(proc.cmdline())
        if _path_matches(cmdline, root_norm) or root_norm in cmdline.lower().replace("/", "\\"):
            return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, OSError):
        pass

    try:
        for opened in proc.open_files() or []:
            if _path_matches(opened.path, root_norm):
                return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, OSError):
        pass

    return False


def _terminate_process(proc: psutil.Process) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=1.2)
    except psutil.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.2)


def _force_release_dataset_locks(dataset_root: Path) -> tuple[list[int], list[int]]:
    root_norm = _normalize_path_str(dataset_root)
    current_pid = os.getpid()
    parent_pid = os.getppid()

    terminated: list[int] = []
    blocked: list[int] = []

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pid = proc.info.get("pid")
            if pid in (current_pid, parent_pid):
                continue

            name = (proc.info.get("name") or "").lower()
            if name in CRITICAL_PROCESS_NAMES:
                continue

            if not _process_holds_dataset(proc, root_norm):
                continue

            _terminate_process(proc)
            terminated.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, PermissionError):
            blocked.append(proc.pid)

    if terminated:
        time.sleep(0.4)

    return terminated, blocked


def _rename_with_retries(src: Path, dst: Path, retries: int = 10, delay_s: float = 0.25) -> None:
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            src.rename(dst)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay_s)
    if last_exc:
        raise last_exc


def _move_with_retries(src: Path, dst: Path, retries: int = 10, delay_s: float = 0.25) -> None:
    if dst.exists() and src.resolve() != dst.resolve():
        raise ValueError(f"Destination already exists: {dst}")

    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay_s)
    if last_exc:
        raise last_exc


def _busy_dataset_error(
    dataset_root: Path,
    force_unlock: bool,
    terminated: list[int] | None = None,
    blocked: list[int] | None = None,
) -> PermissionError:
    if force_unlock:
        details = []
        if terminated:
            details.append(f"terminated={terminated}")
        if blocked:
            details.append(f"blocked={blocked}")
        suffix = f" ({', '.join(details)})" if details else ""
        return PermissionError(
            f"Dataset is still busy after forced unlock: {dataset_root}.{suffix}"
        )
    return PermissionError(
        f"Dataset is busy: {dataset_root}. Close preview/tools or retry with force_local=true."
    )


def move_dataset_root(src_root: Path, dst_root: Path, force_unlock: bool = False) -> Path:
    src = Path(src_root).expanduser().resolve()
    dst = Path(dst_root).expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    if src == dst:
        return src

    terminated: list[int] = []
    blocked: list[int] = []

    try:
        _move_with_retries(src, dst)
    except ValueError:
        raise
    except PermissionError as exc:
        if not force_unlock:
            raise _busy_dataset_error(src, force_unlock=False) from exc
        terminated, blocked = _force_release_dataset_locks(src)
        try:
            _move_with_retries(src, dst)
        except PermissionError as retry_exc:
            raise _busy_dataset_error(
                src,
                force_unlock=True,
                terminated=terminated,
                blocked=blocked,
            ) from retry_exc

    _clear_v3_caches()
    return dst


def update_v3_dataset_metadata(
    dataset_root: Path,
    robot: Optional[str] = None,
    task_type: Optional[str] = None,
) -> bool:
    root = Path(dataset_root).expanduser().resolve()
    if not lerobot_v3.is_v3_dataset_root(root):
        return False

    changed = False

    if robot is not None:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing info.json: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("robot_type") != robot:
            info["robot_type"] = robot
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            changed = True

    if task_type is not None:
        tasks_path = root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            import pandas as pd

            tasks_df = pd.read_parquet(tasks_path)
            target_col = None
            for col in ("task_description", "task", "description"):
                if col in tasks_df.columns:
                    target_col = col
                    break

            if target_col is None:
                target_col = "task_description"
                tasks_df[target_col] = task_type
                changed = True
            else:
                old_values = tasks_df[target_col].astype(str).tolist() if not tasks_df.empty else []
                tasks_df[target_col] = task_type
                if not old_values or any(v != task_type for v in old_values):
                    changed = True

            tasks_df.to_parquet(tasks_path, index=False)

    if changed:
        _clear_v3_caches()

    return changed


def delete_v3_episode(dataset_root: Path, episode_index: int, force_unlock: bool = False) -> None:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not lerobot_v3.is_v3_dataset_root(dataset_root):
        raise ValueError("Local episode deletion only supports LeRobot v3 datasets")

    try:
        from lerobot.datasets.dataset_tools import delete_episodes
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError(f"Failed to import LeRobot dataset tools: {exc}") from exc

    dataset_name = dataset_root.name
    stamp = int(time.time() * 1000)
    parent = dataset_root.parent
    temp_root = parent / f".{dataset_name}.tmp-delete-{stamp}"
    backup_root = parent / f".{dataset_name}.bak-delete-{stamp}"

    _safe_rmtree(temp_root)
    _safe_rmtree(backup_root)

    terminated: list[int] = []
    blocked: list[int] = []

    try:
        _rename_with_retries(dataset_root, backup_root)
    except PermissionError as exc:
        if not force_unlock:
            raise _busy_dataset_error(dataset_root, force_unlock=False) from exc
        terminated, blocked = _force_release_dataset_locks(dataset_root)
        try:
            _rename_with_retries(dataset_root, backup_root)
        except PermissionError as retry_exc:
            raise _busy_dataset_error(
                dataset_root,
                force_unlock=True,
                terminated=terminated,
                blocked=blocked,
            ) from retry_exc

    try:
        dataset = LeRobotDataset(dataset_name, root=backup_root)
        total_episodes = int(getattr(dataset.meta, "total_episodes", 0) or 0)

        if episode_index < 0 or episode_index >= total_episodes:
            raise ValueError(
                f"Episode index {episode_index} is out of range for dataset with {total_episodes} episodes"
            )
        if total_episodes <= 1:
            raise ValueError("Cannot delete the last episode. Delete the dataset instead.")

        new_dataset = delete_episodes(
            dataset=dataset,
            episode_indices=[int(episode_index)],
            output_dir=temp_root,
            repo_id=dataset_name,
        )

        # Release file handles before replacing the dataset directory on Windows.
        del new_dataset
        del dataset
        gc.collect()

        if not lerobot_v3.is_v3_dataset_root(temp_root):
            raise RuntimeError("Failed to materialize updated dataset while deleting episode")

        _rename_with_retries(temp_root, dataset_root)

    except Exception:
        _safe_rmtree(temp_root)
        if backup_root.exists() and not dataset_root.exists():
            _rename_with_retries(backup_root, dataset_root)
        raise
    finally:
        _safe_rmtree(backup_root)
        _clear_v3_caches()


def delete_episode_file(file_path: Path) -> bool:
    target = Path(file_path)
    if target.is_file():
        target.unlink(missing_ok=True)
        return True
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        return True
    return False
