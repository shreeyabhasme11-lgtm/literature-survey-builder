"""File validation before ingestion."""
from pathlib import Path
from app.config import MAX_UPLOAD_MB, ALLOWED_EXTENSIONS


def validate_file(path: Path) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    error_message is empty string when valid.
    """
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"'{path.name}' is not a PDF file."

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return False, (
            f"'{path.name}' is {size_mb:.1f} MB, "
            f"exceeding the {MAX_UPLOAD_MB} MB limit."
        )

    return True, ""