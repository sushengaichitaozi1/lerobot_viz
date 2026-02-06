from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    BigInteger,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from .db import Base


class Robot(Base):
    __tablename__ = "robots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255))
    description = Column(Text)
    metadata_json = Column("metadata", JSON)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    task_types = relationship("TaskType", back_populates="robot", cascade="all, delete-orphan")
    items = relationship("Item", back_populates="robot", cascade="all, delete-orphan")


class TaskType(Base):
    __tablename__ = "task_types"
    __table_args__ = (UniqueConstraint("robot_id", "name", name="uk_task_types_robot_name"),)

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    robot_id = Column(Integer, ForeignKey("robots.id", ondelete="CASCADE"), nullable=False, index=True)
    description = Column(Text)
    metadata_json = Column("metadata", JSON)
    created_at = Column(DateTime, server_default=func.now())

    robot = relationship("Robot", back_populates="task_types")
    items = relationship("Item", back_populates="task_type", cascade="all, delete-orphan")


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(String(255), index=True)
    robot_id = Column(Integer, ForeignKey("robots.id", ondelete="CASCADE"), nullable=False, index=True)
    task_type_id = Column(Integer, ForeignKey("task_types.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = Column(String(1024), nullable=False)
    total_frames = Column(Integer)
    fps = Column(Float)
    duration_s = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    image_count = Column(Integer)
    total_size_bytes = Column(BigInteger)
    camera_count = Column(Integer)
    index_status = Column(String(50), default="pending")
    index_error = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    indexed_at = Column(DateTime)
    deleted_at = Column(DateTime)

    robot = relationship("Robot", back_populates="items")
    task_type = relationship("TaskType", back_populates="items")
    cameras = relationship("CameraInfo", back_populates="item", cascade="all, delete-orphan")


class CameraInfo(Base):
    __tablename__ = "camera_infos"
    __table_args__ = (UniqueConstraint("item_id", "camera_key", name="uk_camera_item_key"),)

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_key = Column(String(100), nullable=False)
    display_name = Column(String(255))
    image_path = Column(String(1024), nullable=False)
    frame_count = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    file_pattern = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())

    item = relationship("Item", back_populates="cameras")


class IndexTask(Base):
    __tablename__ = "index_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_type = Column(String(50), nullable=False)
    target_path = Column(String(1024))
    status = Column(String(50), default="pending")
    progress = Column(Integer, default=0)
    total_episodes = Column(Integer)
    processed_episodes = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    logs = relationship("IndexLog", back_populates="task", cascade="all, delete-orphan")


class IndexLog(Base):
    __tablename__ = "index_logs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("index_tasks.id", ondelete="CASCADE"))
    level = Column(String(20))
    message = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    task = relationship("IndexTask", back_populates="logs")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


Index("idx_items_robot", Item.robot_id)
Index("idx_items_task_type", Item.task_type_id)
Index("idx_items_robot_type", Item.robot_id, Item.task_type_id)
Index("idx_items_deleted", Item.deleted_at, sqlite_where=Item.deleted_at.is_(None))
Index("idx_cameras_item", CameraInfo.item_id)
