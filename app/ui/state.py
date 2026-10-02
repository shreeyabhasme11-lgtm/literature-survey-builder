# app/ui/state.py
"""Centralized session_state helpers."""
import streamlit as st
from typing import Any


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_(key: str, value: Any) -> None:
    st.session_state[key] = value


def init(key: str, default: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = default


def init_all() -> None:
    defaults = {
        "documents": [],        # List[DocumentMeta]
        "pages": {},            # doc_id -> List[PageMeta]
        "sentences": [],        # List[SentenceMeta]
        "tfidf_matrix": None,
        "tfidf_vectorizer": None,
        "ranked_sentences": [], # List[SentenceMeta] sorted by score
        "topics": {},           # topic_id -> keywords
        "survey": {},           # section -> List[str] (composed paragraphs)
        "traces": [],           # List[dict] explainability
        "pipeline_stage": "upload",  # current stage
        "pipeline_fingerprint": None,
        "errors": [],
    }
    for k, v in defaults.items():
        init(k, v)