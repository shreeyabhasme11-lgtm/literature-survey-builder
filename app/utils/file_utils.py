"""File system helpers."""
import shutil
from pathlib import Path
from typing import List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def list_files(directory: Path, extension: str = ".pdf") -> List[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(f"*{extension}"))


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def clean_filename(name: str) -> str:
    """Strip unsafe characters from a filename."""
    import re
    return re.sub(r"[^\w\-_\. ]", "_", name).strip()