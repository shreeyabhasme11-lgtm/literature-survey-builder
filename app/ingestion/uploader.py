"""Handle Streamlit file uploads — save to disk, validate, return DocumentMeta."""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from app.config import UPLOADS
from app.ingestion.validator import validate_file
from app.metadata.schema import DocumentMeta
from app.utils.logger import get_logger

log = get_logger(__name__)


def save_uploaded_files(
    uploaded_files,
) -> Tuple[List[DocumentMeta], List[str]]:
    """
    Save Streamlit UploadedFile objects to data/uploads/.
    Returns (list of DocumentMeta, list of error strings).
    """
    UPLOADS.mkdir(parents=True, exist_ok=True)
    docs: List[DocumentMeta] = []
    errors: List[str] = []

    for uf in uploaded_files:
        dest = UPLOADS / uf.name

        # Write bytes to disk
        dest.write_bytes(uf.getbuffer())

        # Validate
        ok, err = validate_file(dest)
        if not ok:
            dest.unlink(missing_ok=True)
            errors.append(err)
            log.warning(f"Rejected '{uf.name}': {err}")
            continue

        doc_id = str(uuid.uuid4())[:8]
        size   = dest.stat().st_size
        ts     = datetime.now(timezone.utc).isoformat()

        doc = DocumentMeta(
            doc_id=doc_id,
            filename=uf.name,
            filepath=str(dest),
            size_bytes=size,
            upload_ts=ts,
        )
        docs.append(doc)
        log.info(f"Saved '{uf.name}' → {dest} (id={doc_id})")

    return docs, errors