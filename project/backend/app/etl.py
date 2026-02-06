from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Optional
from uuid import uuid4


@dataclass
class EtlTask:
    id: str
    name: str
    enabled: bool
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    total_runs: int = 0


@dataclass
class EtlRun:
    id: str
    task_id: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    records_processed: Optional[int] = None
    error_message: Optional[str] = None


_LOCK = Lock()
_TASKS: dict[str, EtlTask] = {}
_RUNS: dict[str, list[EtlRun]] = {}


def _now() -> datetime:
    return datetime.utcnow()


def _next_id() -> str:
    return uuid4().hex


def list_tasks(page: int, page_size: int, enabled: Optional[bool] = None) -> tuple[int, list[EtlTask]]:
    with _LOCK:
        tasks = list(_TASKS.values())
    if enabled is not None:
        tasks = [task for task in tasks if task.enabled == enabled]
    tasks.sort(key=lambda task: task.created_at, reverse=True)
    total = len(tasks)
    start = (page - 1) * page_size
    end = start + page_size
    return total, tasks[start:end]


def get_task(task_id: str) -> Optional[EtlTask]:
    with _LOCK:
        return _TASKS.get(task_id)


def find_task_by_dataset(dataset_id: int) -> Optional[EtlTask]:
    with _LOCK:
        for task in _TASKS.values():
            if task.config.get("dataset_id") == dataset_id:
                return task
    return None


def create_task(name: str, config: dict[str, Any], enabled: bool = True) -> EtlTask:
    task = EtlTask(
        id=_next_id(),
        name=name,
        enabled=enabled,
        config=config,
        created_at=_now(),
        updated_at=_now(),
    )
    with _LOCK:
        _TASKS[task.id] = task
        _RUNS.setdefault(task.id, [])
    return task


def ensure_task_for_dataset(dataset_id: int, name: str, config: dict[str, Any]) -> EtlTask:
    task = find_task_by_dataset(dataset_id)
    if task:
        with _LOCK:
            task.name = name
            task.config = config
            task.updated_at = _now()
        return task
    return create_task(name=name, config=config)


def update_task(task_id: str, name: Optional[str], enabled: Optional[bool], config: Optional[dict[str, Any]]) -> Optional[EtlTask]:
    with _LOCK:
        task = _TASKS.get(task_id)
        if not task:
            return None
        if name is not None:
            task.name = name
        if enabled is not None:
            task.enabled = enabled
        if config is not None:
            task.config = config
        task.updated_at = _now()
        return task


def delete_task(task_id: str) -> bool:
    with _LOCK:
        if task_id not in _TASKS:
            return False
        _TASKS.pop(task_id)
        _RUNS.pop(task_id, None)
        return True


def start_run(task_id: str) -> EtlRun:
    run = EtlRun(
        id=_next_id(),
        task_id=task_id,
        status="running",
        start_time=_now(),
        records_processed=0,
    )
    with _LOCK:
        _RUNS.setdefault(task_id, []).append(run)
        task = _TASKS.get(task_id)
        if task:
            task.total_runs += 1
            task.updated_at = _now()
    return run


def list_runs(task_id: str, page: int, page_size: int) -> tuple[int, list[EtlRun]]:
    with _LOCK:
        runs = list(_RUNS.get(task_id, []))
    runs.sort(key=lambda run: run.start_time or _now(), reverse=True)
    total = len(runs)
    start = (page - 1) * page_size
    end = start + page_size
    return total, runs[start:end]


def update_run(task_id: str, run_id: str, records_processed: Optional[int] = None, status: Optional[str] = None, error_message: Optional[str] = None) -> Optional[EtlRun]:
    with _LOCK:
        runs = _RUNS.get(task_id, [])
        for run in runs:
            if run.id == run_id:
                if records_processed is not None:
                    run.records_processed = records_processed
                if status is not None:
                    run.status = status
                if error_message is not None:
                    run.error_message = error_message
                return run
    return None


def finish_run(task_id: str, run_id: str, status: str, records_processed: Optional[int] = None, error_message: Optional[str] = None) -> Optional[EtlRun]:
    run = update_run(task_id, run_id, records_processed=records_processed, status=status, error_message=error_message)
    if run:
        run.end_time = _now()
    return run
