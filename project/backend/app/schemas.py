from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class User(BaseModel):
    id: int
    username: str
    displayName: Optional[str] = None
    roles: list[str] = []


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int
    user: User


class RobotSummary(BaseModel):
    id: int
    name: str
    displayName: Optional[str] = None
    description: Optional[str] = None
    typeCount: int = 0
    episodeCount: int = 0


class RobotListResponse(BaseModel):
    robots: list[RobotSummary]


class TaskTypeSummary(BaseModel):
    id: int
    name: str
    displayName: Optional[str] = None
    episodeCount: int = 0


class TaskTypeListResponse(BaseModel):
    taskTypes: list[TaskTypeSummary]


class DatasetSummary(BaseModel):
    id: str
    name: str
    path: str
    robot: Optional[str] = None
    taskType: Optional[str] = None
    episodeCount: int = 0


class DatasetListResponse(BaseModel):
    datasets: list[DatasetSummary]


class ItemSummary(BaseModel):
    id: int
    episodeId: Optional[str] = None
    robot: Optional[str] = None
    taskType: Optional[str] = None
    thumbnailUrl: Optional[str] = None
    totalFrames: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[str] = None
    cameraCount: Optional[int] = None


class PagedItems(BaseModel):
    items: list[ItemSummary]
    total: int
    page: int
    pageSize: int


class RobotInfo(BaseModel):
    id: int
    name: str
    displayName: Optional[str] = None


class TaskTypeInfo(BaseModel):
    id: int
    name: str
    displayName: Optional[str] = None


class CameraInfo(BaseModel):
    cameraKey: str
    displayName: Optional[str] = None
    frameCount: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class ItemDetail(BaseModel):
    id: int
    episodeId: Optional[str] = None
    robot: RobotInfo
    taskType: TaskTypeInfo
    filePath: str
    totalFrames: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    imageCount: Optional[int] = None
    totalSizeBytes: Optional[int] = None
    cameraCount: Optional[int] = None
    cameras: list[CameraInfo] = []


class IndexScanRequest(BaseModel):
    path: str


class IndexScanResponse(BaseModel):
    taskId: int
    status: str
    message: str


class IndexStatusResponse(BaseModel):
    taskId: int
    status: str
    progress: int
    totalEpisodes: Optional[int] = None
    processedEpisodes: Optional[int] = None
    currentPath: Optional[str] = None
    errorMessage: Optional[str] = None


class DatasetRegisterRequest(BaseModel):
    path: str
    robot: Optional[str] = None
    task_type: Optional[str] = Field(default=None, alias="task_type")


class DatasetRegisterResponse(BaseModel):
    taskId: int
    status: str
    message: str
    mappedRobot: Optional[str] = None
    mappedTaskType: Optional[str] = None


class RerunGenerateRequest(BaseModel):
    format: str = "rrd"


class RerunGenerateResponse(BaseModel):
    status: str
    downloadUrl: str
    filePath: str
    fileSize: int
    duration: Optional[float] = None


class RerunServerRequest(BaseModel):
    mode: str = "distant"
    ws_port: int = 9087
    web_port: int = 9090


class RerunServerResponse(BaseModel):
    status: str
    wsUrl: str
    webUrl: str
    instructions: Optional[str] = None
