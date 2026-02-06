#!/usr/bin/env python
"""Re-index LeRobot dataset folders into the v2.0 database."""
import os
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parent
backend_dir = repo_root / "project" / "backend"
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

db_path = (repo_root / "lerobot_dataset_v2.db").as_posix()
data_root = (repo_root / "project" / "data").as_posix()

os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
os.environ["DATA_ROOT"] = data_root
os.environ["JWT_SECRET"] = "dev-secret"
os.environ["PRIMARY_CAMERA_KEY"] = "top"

from app import indexer, models  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402

root = Path(os.environ["DATA_ROOT"])
models.Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    task = models.IndexTask(
        task_type="scan",
        target_path=str(root),
        status="running",
        progress=0,
        started_at=datetime.utcnow(),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    total = indexer.index_dataset(db, root, task=task)
    print(f"Indexed {total} episodes from {root}")
