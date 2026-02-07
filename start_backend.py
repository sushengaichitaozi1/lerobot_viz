#!/usr/bin/env python
"""Start backend server with SQLite configuration."""
import os
import sys
from pathlib import Path

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    repo_root = Path(__file__).resolve().parent
    backend_dir = repo_root / "project" / "backend"
    sys.path.insert(0, str(backend_dir))
    os.chdir(backend_dir)

    db_path = (repo_root / "lerobot_dataset_v2.db").as_posix()
    data_root = (repo_root / "project" / "data").as_posix()

    os.environ.setdefault('DATABASE_URL', f'sqlite:///{db_path}')
    os.environ.setdefault('DATA_ROOT', data_root)
    os.environ.setdefault('JWT_SECRET', 'dev-secret')
    os.environ.setdefault('ADMIN_USERNAME', 'admin')
    os.environ.setdefault('ADMIN_PASSWORD', 'admin')
    os.environ.setdefault('CORS_ORIGINS', 'http://localhost:5090,http://127.0.0.1:5090')
    os.environ.setdefault('PRIMARY_CAMERA_KEY', 'top')

    print("=" * 50)
    print("Starting LeRobot Backend Server")
    print("=" * 50)
    print(f"DATABASE_URL: {os.environ['DATABASE_URL']}")
    print(f"DATA_ROOT: {os.environ['DATA_ROOT']}")
    print(f"JWT_SECRET: {os.environ['JWT_SECRET']}")
    print(f"ADMIN_USERNAME: {os.environ['ADMIN_USERNAME']}")
    print(f"ADMIN_PASSWORD: {os.environ['ADMIN_PASSWORD']}")
    print("=" * 50)

    import uvicorn
    uvicorn.run('app.main:app', host='127.0.0.1', port=8899, reload=False)
