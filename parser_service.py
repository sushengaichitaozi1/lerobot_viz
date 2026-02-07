from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parent / "project" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.parser_service import (
    infer_dataset_name,
    parse_dataset,
    parse_dataset_name,
    validate_dataset_name,
)

__all__ = ["parse_dataset", "parse_dataset_name", "validate_dataset_name", "infer_dataset_name"]
