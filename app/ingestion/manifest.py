"""Persist and load the document manifest as JSON."""
import json
from pathlib import Path
from typing import List

from app.metadata.schema import DocumentMeta
from app.utils.logger import get_logger

log = get_logger(__name__)

_MANIFEST_FILENAME = "manifest.json"


def save_manifest(docs: List[DocumentMeta], data_dir: Path) -> Path:
    path = data_dir / _MANIFEST_FILENAME
    payload = [
        {
            "doc_id":     d.doc_id,
            "filename":   d.filename,
            "filepath":   d.filepath,
            "size_bytes": d.size_bytes,
            "upload_ts":  d.upload_ts,
            "num_pages":  d.num_pages,
        }
        for d in docs
    ]
    path.write_text(json.dumps(payload, indent=2))
    log.info(f"Manifest saved → {path} ({len(docs)} docs)")
    return path


def load_manifest(data_dir: Path) -> List[DocumentMeta]:
    path = data_dir / _MANIFEST_FILENAME
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    docs = []
    for item in payload:
        d = DocumentMeta(
            doc_id    = item["doc_id"],
            filename  = item["filename"],
            filepath  = item["filepath"],
            size_bytes= item["size_bytes"],
            upload_ts = item["upload_ts"],
        )
        d.num_pages = item.get("num_pages", 0)
        docs.append(d)
    log.info(f"Manifest loaded ← {path} ({len(docs)} docs)")
    return docs