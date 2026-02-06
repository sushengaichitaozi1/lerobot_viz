import os
from dataclasses import dataclass, field
from pathlib import Path


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_cors_origins() -> list[str]:
    return _split_csv(os.getenv("CORS_ORIGINS", "http://localhost:5090")) or ["http://localhost:5090"]


REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_video_encoding() -> dict:
    return {
        "codec": "libx264",
        "preset": "fast",
        "crf": 23,
        "pix_fmt": "yuv420p",
        "movflags": "frag_keyframe+empty_moov+default_base_moof",
        "tune": "zerolatency",
        "g": 30,
        "bframes": 0,
    }


def _default_stream_config() -> dict:
    return {
        "max_concurrent_streams": 5,
        "cache_enabled": True,
        "cache_size": 10,
        "cache_ttl": 300,
    }


def _default_resolutions() -> dict:
    return {
        "360p": {"width": 480, "height": 360},
        "480p": {"width": 640, "height": 480},
        "720p": {"width": 1280, "height": 720},
        "1080p": {"width": 1920, "height": 1080},
    }


def _default_rerun_config() -> dict:
    return {
        "downloads_dir": str(REPO_ROOT / "project" / "downloads" / "rrd"),
        "ws_port": 9087,
        "web_port": 9090,
        "auto_cleanup_days": 7,
        "batch_size": 32,
        "num_workers": 4,
    }


@dataclass
class Settings:
    api_title: str = "LeRobot Dataset API v2.0"
    data_root: Path = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "project" / "data"))).resolve()
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///lerobot_dataset_v2.db")
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-secret")
    jwt_expire_seconds: int = int(os.getenv("JWT_EXPIRE_SECONDS", "86400"))
    jwt_refresh_seconds: int = int(os.getenv("JWT_REFRESH_SECONDS", "604800"))
    admin_username: str = os.getenv("ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "admin")
    cors_origins: list[str] = field(default_factory=_default_cors_origins)
    primary_camera_key: str = os.getenv("PRIMARY_CAMERA_KEY", "top")
    ffmpeg_path: str = os.getenv("FFMPEG_PATH", "")
    category_map_path: Path = Path(
        os.getenv("CATEGORY_MAP_PATH", str(REPO_ROOT / "project" / "data" / "category_map.json"))
    ).resolve()
    default_page_size: int = int(os.getenv("DEFAULT_PAGE_SIZE", "50"))
    max_page_size: int = int(os.getenv("MAX_PAGE_SIZE", "200"))
    video_encoding: dict = field(default_factory=_default_video_encoding)
    stream_config: dict = field(default_factory=_default_stream_config)
    resolutions: dict = field(default_factory=_default_resolutions)
    rerun_config: dict = field(default_factory=_default_rerun_config)

    @property
    def rerun_downloads_dir(self) -> Path:
        return Path(self.rerun_config["downloads_dir"]).resolve()


settings = Settings()
