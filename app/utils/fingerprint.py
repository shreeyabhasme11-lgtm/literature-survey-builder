"""Pipeline fingerprinting for invalidating stale cache."""
import hashlib
from pathlib import Path
from typing import List
import streamlit as st

from app.ui import state


def get_pipeline_fingerprint(docs: List) -> str:
    """Compute a fingerprint based on the active documents and source code."""
    hasher = hashlib.sha256()
    
    # 1. Hash document metadata (names, sizes, mtimes)
    if docs:
        for doc in sorted(docs, key=lambda d: getattr(d, 'filename', str(d))):
            hasher.update(getattr(doc, 'filename', '').encode('utf-8'))
            try:
                filepath = getattr(doc, 'filepath', None)
                if filepath:
                    stat = Path(filepath).stat()
                    hasher.update(str(stat.st_size).encode('utf-8'))
                    hasher.update(str(stat.st_mtime).encode('utf-8'))
            except Exception:
                pass
                
    # 2. Hash source code contents
    app_dir = Path(__file__).parent.parent
    for py_file in sorted(app_dir.rglob('*.py')):
        if '__pycache__' in py_file.parts:
            continue
        try:
            content = py_file.read_bytes()
            hasher.update(py_file.name.encode('utf-8'))
            hasher.update(content)
        except Exception:
            pass
            
    return hasher.hexdigest()


def is_pipeline_fresh() -> bool:
    """Check if the current fingerprint matches the one stored in session state."""
    docs = state.get("documents") or []
    current_fp = get_pipeline_fingerprint(docs)
    stored_fp = state.get("pipeline_fingerprint")
    return current_fp == stored_fp


def require_fresh_pipeline() -> bool:
    """Returns True if fresh, False if stale (and shows a warning)."""
    if not is_pipeline_fresh():
        st.error(
            "🛑 **Pipeline State Stale**\n\n"
            "Code, configuration, or documents have changed since the last extraction. "
            "To prevent corrupted or stale output, you must re-run **Extract Text** (Tab 2) "
            "or click **Reset Pipeline** (Sidebar)."
        )
        return False
    return True
