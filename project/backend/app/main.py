from __future__ import annotations

from datetime import datetime
import json
import base64
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from . import auth, indexer, models, schemas
from .config import settings
from .db import SessionLocal, engine, get_db
from .services import lerobot_v3, rerun_viz, thumbnail, video_streamer


app = FastAPI(title=settings.api_title)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.data_root.mkdir(parents=True, exist_ok=True)
settings.rerun_downloads_dir.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def on_startup() -> None:
    models.Base.metadata.create_all(bind=engine)
    rerun_viz.cleanup_old_rrd_files()


static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir), html=False), name="static")
app.mount(
    "/downloads/rrd",
    StaticFiles(directory=str(settings.rerun_downloads_dir), html=False),
    name="rrd",
)


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _download_url(request: Request, filename: str) -> str:
    return f"{_base_url(request)}/downloads/rrd/{filename}"


def _thumbnail_url(request: Request, item_id: int, camera: str) -> str:
    return f"{_base_url(request)}/items/{item_id}/thumbnail?camera={camera}"


def _parse_range(range_header: str, total_frames: Optional[int]) -> tuple[int, int]:
    raw = range_header.replace("bytes=", "").strip()
    if "-" not in raw:
        return 0, max((total_frames or 1) - 1, 0)
    start_str, end_str = raw.split("-", 1)
    start = int(start_str) if start_str else 0
    if end_str:
        end = int(end_str)
    else:
        end = max((total_frames or start + 1) - 1, start)
    return max(start, 0), max(end, start)


def _encode_dataset_id(path: str) -> str:
    encoded = base64.urlsafe_b64encode(path.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _decode_dataset_id(dataset_id: str) -> str:
    padding = "=" * (-len(dataset_id) % 4)
    return base64.urlsafe_b64decode(dataset_id + padding).decode("utf-8")


def _dataset_root_for_item(item: models.Item) -> Path:
    path = Path(item.file_path)
    if lerobot_v3.is_v3_dataset_root(path):
        return path
    probe = path if path.is_dir() else path.parent
    if "episodes" in probe.parts:
        idx = probe.parts.index("episodes")
        return Path(*probe.parts[:idx])
    return probe


def _build_robot_summary(robot: models.Robot, type_count: int, episode_count: int) -> schemas.RobotSummary:
    return schemas.RobotSummary(
        id=robot.id,
        name=robot.name,
        displayName=robot.display_name or robot.name,
        description=robot.description,
        typeCount=type_count,
        episodeCount=episode_count,
    )


def _build_task_type_summary(task_type: models.TaskType, episode_count: int) -> schemas.TaskTypeSummary:
    return schemas.TaskTypeSummary(
        id=task_type.id,
        name=task_type.name,
        displayName=task_type.display_name or task_type.name,
        episodeCount=episode_count,
    )


def _build_item_summary(request: Request, item: models.Item) -> schemas.ItemSummary:
    resolution = None
    if item.width and item.height:
        resolution = f"{item.width}x{item.height}"
    camera_key = settings.primary_camera_key
    if item.cameras:
        camera_key = next(
            (
                c.camera_key
                for c in item.cameras
                if c.camera_key == settings.primary_camera_key
                or settings.primary_camera_key in c.camera_key
            ),
            item.cameras[0].camera_key,
        )
    return schemas.ItemSummary(
        id=item.id,
        episodeId=item.episode_id,
        robot=item.robot.name if item.robot else None,
        taskType=item.task_type.name if item.task_type else None,
        thumbnailUrl=_thumbnail_url(request, item.id, camera_key),
        totalFrames=item.total_frames,
        duration=item.duration_s,
        fps=item.fps,
        resolution=resolution,
        cameraCount=item.camera_count,
    )


@app.post("/auth/login", response_model=schemas.AuthTokenResponse)
def login(payload: schemas.LoginRequest):
    user = auth.authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return schemas.AuthTokenResponse(
        access_token=auth.create_access_token(user),
        refresh_token=auth.create_refresh_token(user),
        expires_in=settings.jwt_expire_seconds,
        user=user,
    )


@app.post("/auth/refresh", response_model=schemas.AuthTokenResponse)
def refresh(payload: schemas.RefreshTokenRequest):
    try:
        decoded = auth.decode_token(payload.refresh_token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    if decoded.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    user = auth._build_user(decoded.get("username", "user"))
    return schemas.AuthTokenResponse(
        access_token=auth.create_access_token(user),
        refresh_token=auth.create_refresh_token(user),
        expires_in=settings.jwt_expire_seconds,
        user=user,
    )


@app.get("/auth/me", response_model=schemas.User)
def me(current_user: schemas.User = Depends(auth.get_current_user)):
    return current_user


@app.get("/robots", response_model=schemas.RobotListResponse)
def list_robots(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    robots = db.query(models.Robot).order_by(models.Robot.name.asc()).all()
    type_counts = dict(
        db.query(models.TaskType.robot_id, func.count(models.TaskType.id))
        .group_by(models.TaskType.robot_id)
        .all()
    )
    episode_counts = dict(
        db.query(models.Item.robot_id, func.count(models.Item.id))
        .filter(models.Item.deleted_at.is_(None))
        .group_by(models.Item.robot_id)
        .all()
    )
    return schemas.RobotListResponse(
        robots=[
            _build_robot_summary(
                robot,
                type_counts.get(robot.id, 0),
                episode_counts.get(robot.id, 0),
            )
            for robot in robots
        ]
    )


@app.get("/robots/{name}/types", response_model=schemas.TaskTypeListResponse)
def list_task_types(
    name: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    robot = db.query(models.Robot).filter(models.Robot.name == name).one_or_none()
    if not robot:
        raise HTTPException(status_code=404, detail="Robot not found")
    task_types = (
        db.query(models.TaskType)
        .filter(models.TaskType.robot_id == robot.id)
        .order_by(models.TaskType.name.asc())
        .all()
    )
    episode_counts = dict(
        db.query(models.Item.task_type_id, func.count(models.Item.id))
        .filter(models.Item.robot_id == robot.id, models.Item.deleted_at.is_(None))
        .group_by(models.Item.task_type_id)
        .all()
    )
    return schemas.TaskTypeListResponse(
        taskTypes=[
            _build_task_type_summary(task_type, episode_counts.get(task_type.id, 0))
            for task_type in task_types
        ]
    )


@app.get("/task-types", response_model=schemas.TaskTypeListResponse)
def list_all_task_types(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    task_types = db.query(models.TaskType).order_by(models.TaskType.name.asc()).all()
    episode_counts = dict(
        db.query(models.Item.task_type_id, func.count(models.Item.id))
        .filter(models.Item.deleted_at.is_(None))
        .group_by(models.Item.task_type_id)
        .all()
    )
    return schemas.TaskTypeListResponse(
        taskTypes=[
            _build_task_type_summary(task_type, episode_counts.get(task_type.id, 0))
            for task_type in task_types
        ]
    )


@app.get("/datasets", response_model=schemas.DatasetListResponse)
def list_datasets(
    robot: Optional[str] = None,
    task_type: Optional[str] = Query(default=None, alias="type"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    query = (
        db.query(models.Item)
        .join(models.Robot)
        .join(models.TaskType)
        .filter(models.Item.deleted_at.is_(None))
    )
    if robot:
        query = query.filter(models.Robot.name == robot)
    if task_type:
        query = query.filter(models.TaskType.name == task_type)

    datasets: dict[str, schemas.DatasetSummary] = {}
    for item in query.all():
        root = _dataset_root_for_item(item)
        key = str(root)
        if key not in datasets:
            datasets[key] = schemas.DatasetSummary(
                id=_encode_dataset_id(key),
                name=root.name,
                path=key,
                robot=item.robot.name if item.robot else None,
                taskType=item.task_type.name if item.task_type else None,
                episodeCount=0,
            )
        datasets[key].episodeCount += 1

    return schemas.DatasetListResponse(datasets=list(datasets.values()))


@app.get("/datasets/{datasetId}/items", response_model=schemas.PagedItems)
def list_dataset_items(
    datasetId: str,
    request: Request,
    page: int = 1,
    page_size: int = Query(default=settings.default_page_size, alias="page_size"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    page = max(page, 1)
    page_size = min(max(page_size, 1), settings.max_page_size)
    dataset_path = _decode_dataset_id(datasetId)

    query = (
        db.query(models.Item)
        .options(
            joinedload(models.Item.robot),
            joinedload(models.Item.task_type),
            joinedload(models.Item.cameras),
        )
        .filter(models.Item.deleted_at.is_(None))
        .filter(models.Item.file_path.startswith(dataset_path))
    )

    total = query.count()
    items = (
        query.order_by(models.Item.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return schemas.PagedItems(
        items=[_build_item_summary(request, item) for item in items],
        total=total,
        page=page,
        pageSize=page_size,
    )


@app.get("/items", response_model=schemas.PagedItems)
def list_items(
    request: Request,
    robot: Optional[str] = None,
    task_type: Optional[str] = Query(default=None, alias="type"),
    page: int = 1,
    page_size: int = Query(default=settings.default_page_size, alias="page_size"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    page = max(page, 1)
    page_size = min(max(page_size, 1), settings.max_page_size)

    query = (
        db.query(models.Item)
        .options(
            joinedload(models.Item.robot),
            joinedload(models.Item.task_type),
            joinedload(models.Item.cameras),
        )
        .join(models.Robot)
        .join(models.TaskType)
        .filter(models.Item.deleted_at.is_(None))
    )
    if robot:
        query = query.filter(models.Robot.name == robot)
    if task_type:
        query = query.filter(models.TaskType.name == task_type)

    total = query.count()
    items = (
        query.order_by(models.Item.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return schemas.PagedItems(
        items=[_build_item_summary(request, item) for item in items],
        total=total,
        page=page,
        pageSize=page_size,
    )


@app.get("/items/{itemId}", response_model=schemas.ItemDetail)
def get_item(
    itemId: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    item = db.get(models.Item, itemId)
    if not item or item.deleted_at:
        raise HTTPException(status_code=404, detail="Item not found")

    cameras = [
        schemas.CameraInfo(
            cameraKey=camera.camera_key,
            displayName=camera.display_name or camera.camera_key,
            frameCount=camera.frame_count,
            width=camera.width,
            height=camera.height,
        )
        for camera in item.cameras
    ]

    return schemas.ItemDetail(
        id=item.id,
        episodeId=item.episode_id,
        robot=schemas.RobotInfo(
            id=item.robot.id,
            name=item.robot.name,
            displayName=item.robot.display_name or item.robot.name,
        ),
        taskType=schemas.TaskTypeInfo(
            id=item.task_type.id,
            name=item.task_type.name,
            displayName=item.task_type.display_name or item.task_type.name,
        ),
        filePath=item.file_path,
        totalFrames=item.total_frames,
        fps=item.fps,
        duration=item.duration_s,
        width=item.width,
        height=item.height,
        imageCount=item.image_count,
        totalSizeBytes=item.total_size_bytes,
        cameraCount=item.camera_count,
        cameras=cameras,
    )


@app.get("/items/{itemId}/stream")
def stream_item(
    itemId: int,
    request: Request,
    camera: str = Query(default=settings.primary_camera_key),
    resolution: Optional[str] = None,
    db: Session = Depends(get_db),
):
    item = db.get(models.Item, itemId)
    if not item or item.deleted_at:
        raise HTTPException(status_code=404, detail="Item not found")

    camera_info = (
        db.query(models.CameraInfo)
        .filter(models.CameraInfo.item_id == item.id, models.CameraInfo.camera_key == camera)
        .one_or_none()
    )
    if not camera_info:
        raise HTTPException(status_code=404, detail="Camera not found")

    source_path = Path(camera_info.image_path)
    dataset_root = Path(item.file_path)
    if lerobot_v3.is_v3_dataset_root(dataset_root):
        episode_index = lerobot_v3.parse_episode_index(item.episode_id)
        if episode_index is None:
            raise HTTPException(status_code=404, detail="Episode index not found")
        clip = lerobot_v3.get_video_clip(dataset_root, episode_index, camera)
        if not clip:
            raise HTTPException(status_code=404, detail="Video clip not found")
        resolution_config = settings.resolutions.get(resolution) if resolution else None
        return video_streamer.stream_video(
            clip["video_path"],
            None,
            clip.get("fps") or item.fps,
            resolution_config,
            clip.get("start_time"),
            clip.get("duration"),
            status_code=200,
            headers={},
        )
    if source_path.is_file() and source_path.suffix.lower() == ".mp4":
        return FileResponse(source_path, media_type="video/mp4")

    range_header = request.headers.get("range")
    start_time = None
    duration = None
    if range_header:
        start, end = _parse_range(range_header, item.total_frames)
        if item.total_frames and item.duration_s:
            start_time = (start / item.total_frames) * item.duration_s
            end_time = (end / item.total_frames) * item.duration_s
            duration = max(end_time - start_time, 0.0)

    resolution_config = settings.resolutions.get(resolution) if resolution else None
    try:
        return video_streamer.stream_video(
            source_path,
            camera_info.file_pattern,
            item.fps,
            resolution_config,
            start_time,
            duration,
            status_code=200,
            headers={},
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/items/{itemId}/thumbnail")
def get_thumbnail(
    itemId: int,
    camera: str = Query(default=settings.primary_camera_key),
    frame: int = 0,
    db: Session = Depends(get_db),
):
    item = db.get(models.Item, itemId)
    if not item or item.deleted_at:
        raise HTTPException(status_code=404, detail="Item not found")

    camera_info = (
        db.query(models.CameraInfo)
        .filter(models.CameraInfo.item_id == item.id, models.CameraInfo.camera_key == camera)
        .one_or_none()
    )
    if not camera_info:
        raise HTTPException(status_code=404, detail="Camera not found")

    dataset_root = Path(item.file_path)
    if lerobot_v3.is_v3_dataset_root(dataset_root):
        episode_index = lerobot_v3.parse_episode_index(item.episode_id)
        if episode_index is None:
            raise HTTPException(status_code=404, detail="Episode index not found")
        clip = lerobot_v3.get_video_clip(dataset_root, episode_index, camera)
        if not clip:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        return thumbnail.render_video_thumbnail(clip["video_path"], clip.get("start_time", 0.0))

    image_path = thumbnail.resolve_thumbnail_path(
        Path(camera_info.image_path),
        camera_info.file_pattern or "%06d.png",
        frame,
    )
    return FileResponse(image_path)


@app.get("/items/{itemId}/timeseries")
def get_timeseries(
    itemId: int,
    max_points: int = 500,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    item = db.get(models.Item, itemId)
    if not item or item.deleted_at:
        raise HTTPException(status_code=404, detail="Item not found")
    dataset_root = Path(item.file_path)
    if not lerobot_v3.is_v3_dataset_root(dataset_root):
        raise HTTPException(status_code=404, detail="Timeseries not available")
    episode_index = lerobot_v3.parse_episode_index(item.episode_id)
    if episode_index is None:
        raise HTTPException(status_code=404, detail="Episode index not found")
    series = lerobot_v3.load_timeseries(dataset_root, episode_index, max_points=max_points)
    if not series:
        raise HTTPException(status_code=404, detail="Timeseries not found")
    return series


def _run_index_task(task_id: int, path: str) -> None:
    with SessionLocal() as db:
        task = db.get(models.IndexTask, task_id)
        if not task:
            return
        try:
            indexer.index_dataset(db, Path(path), task=task)
        except Exception as exc:
            task.status = "failed"
            task.error_message = str(exc)
            task.completed_at = datetime.utcnow()
            db.commit()


@app.post("/index/scan", response_model=schemas.IndexScanResponse)
def scan_index(
    payload: schemas.IndexScanRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    root = Path(payload.path)
    if not root.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    task = models.IndexTask(
        task_type="scan",
        target_path=str(root),
        status="pending",
        progress=0,
        created_at=datetime.utcnow(),
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    background_tasks.add_task(_run_index_task, task.id, str(root))
    return schemas.IndexScanResponse(taskId=task.id, status="running", message="Index task started")


@app.get("/index/status", response_model=schemas.IndexStatusResponse)
def index_status(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    task = db.get(models.IndexTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Index task not found")

    return schemas.IndexStatusResponse(
        taskId=task.id,
        status=task.status,
        progress=task.progress or 0,
        totalEpisodes=task.total_episodes,
        processedEpisodes=task.processed_episodes,
        currentPath=None,
        errorMessage=task.error_message,
    )


def _looks_like_dataset_dir(path: Path) -> bool:
    return (
        lerobot_v3.is_v3_dataset_root(path)
        or (path / "episodes").is_dir()
        or (path / "robots").is_dir()
        or (path / "tasks").is_dir()
    )


def _find_dataset_dirs(root: Path) -> list[Path]:
    if _looks_like_dataset_dir(root):
        return [root]
    if not root.is_dir():
        return []
    candidates = []
    for child in root.iterdir():
        if child.is_dir() and _looks_like_dataset_dir(child):
            candidates.append(child)
    return candidates


@app.post("/datasets/register", response_model=schemas.DatasetRegisterResponse)
def register_dataset(
    payload: schemas.DatasetRegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    root = Path(payload.path)
    if not root.is_absolute():
        root = (settings.data_root / root).resolve()
    if not root.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    mapped_robot = None
    mapped_task = None
    if payload.robot or payload.task_type:
        dataset_dirs = _find_dataset_dirs(root)
        if not dataset_dirs:
            raise HTTPException(
                status_code=400,
                detail="Please provide a dataset folder path (or a folder that contains datasets).",
            )
        map_path = settings.category_map_path
        map_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"datasets": {}}
        if map_path.exists():
            try:
                data = json.loads(map_path.read_text(encoding="utf-8"))
            except Exception:
                data = {"datasets": {}}
        datasets = data.get("datasets")
        if not isinstance(datasets, dict):
            datasets = {}
            data["datasets"] = datasets
        for dataset_dir in dataset_dirs:
            entry = datasets.get(dataset_dir.name, {})
            if payload.robot:
                entry["robot"] = payload.robot
                mapped_robot = payload.robot
            if payload.task_type:
                entry["task_type"] = payload.task_type
                mapped_task = payload.task_type
            datasets[dataset_dir.name] = entry
        map_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    task = models.IndexTask(
        task_type="scan",
        target_path=str(root),
        status="pending",
        progress=0,
        created_at=datetime.utcnow(),
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    background_tasks.add_task(_run_index_task, task.id, str(root))
    return schemas.DatasetRegisterResponse(
        taskId=task.id,
        status="running",
        message="Index task started",
        mappedRobot=mapped_robot,
        mappedTaskType=mapped_task,
    )


@app.post("/items/{itemId}/visualize/rrd", response_model=schemas.RerunGenerateResponse)
def generate_rrd(
    itemId: int,
    payload: schemas.RerunGenerateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    item = db.get(models.Item, itemId)
    if not item or item.deleted_at:
        raise HTTPException(status_code=404, detail="Item not found")

    try:
        rrd_path = rerun_viz.generate_rrd(item.robot.name, item.episode_id or str(item.id), Path(item.file_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    file_size = rrd_path.stat().st_size if rrd_path.exists() else 0
    download_url = _download_url(request, rrd_path.name)
    return schemas.RerunGenerateResponse(
        status="completed",
        downloadUrl=download_url,
        filePath=f"/downloads/rrd/{rrd_path.name}",
        fileSize=file_size,
        duration=item.duration_s,
    )


@app.post("/items/{itemId}/visualize/server", response_model=schemas.RerunServerResponse)
def start_rerun_server(
    itemId: int,
    payload: schemas.RerunServerRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user),
):
    item = db.get(models.Item, itemId)
    if not item or item.deleted_at:
        raise HTTPException(status_code=404, detail="Item not found")

    try:
        rerun_viz.start_rerun_server(
            item.robot.name,
            item.episode_id or str(item.id),
            Path(item.file_path),
            payload.ws_port,
            payload.web_port,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    host = request.url.hostname or "localhost"
    ws_url = f"ws://{host}:{payload.ws_port}"
    web_url = f"http://{host}:{payload.web_port}"
    return schemas.RerunServerResponse(
        status="running",
        wsUrl=ws_url,
        webUrl=web_url,
        instructions=f"Use rerun {ws_url} or visit {web_url}",
    )
