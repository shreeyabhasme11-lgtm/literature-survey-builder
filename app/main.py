"""Streamlit entrypoint — Explainable Literature Survey Builder."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Literature Survey Builder",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "ui" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

from app.ui import state, components
from app.ui.pages import (
    page_upload, page_extract, page_rank,
    page_compose, page_explain, page_export,
)
from app.config import ensure_dirs

ensure_dirs()
state.init_all()

with st.sidebar:
    st.markdown("## 📚 Literature Survey Builder")
    st.markdown("*Fully local · Deterministic · Explainable*")
    st.divider()

    components.progress_sidebar(state.get("pipeline_stage") or "upload")

    st.divider()

    st.markdown("### 🔄 Pipeline Control")
    st.caption(
        "Use reset after code changes, after switching PDFs, or when you want "
        "to fully clear extracted text, rankings, traces, and composed output."
    )

    if st.button(
        "🔄 Reset Pipeline",
        type="secondary",
        use_container_width=True,
        help="Clears all extracted sentences, rankings, topics, traces, and survey output.",
    ):
        keys_to_clear = [
            "documents",
            "pages",
            "sentences",
            "tfidf_matrix",
            "tfidf_vectorizer",
            "ranked_sentences",
            "topics",
            "survey",
            "traces",
            "pipeline_stage",
            "meta_mgr",
            "errors",
            "pipeline_fingerprint",
            "compose_result",
            "selected_sentences",
            "clusters",
            "normalized_sentences",
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()

    docs = state.get("documents") or []
    sentences = state.get("sentences") or []
    ranked = state.get("ranked_sentences") or []
    survey = state.get("survey") or {}

    if docs:
        st.markdown("### 📊 Session Stats")
        st.markdown(f"- **Documents:** {len(docs)}")
        st.markdown(f"- **Sentences extracted:** {len(sentences)}")
        st.markdown(f"- **Sentences ranked:** {len(ranked)}")
        st.markdown(f"- **Survey blocks:** {sum(len(v) for v in survey.values()) if survey else 0}")

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "Extractive evidence selection · Topic grouping · "
        "Cross-paper synthesis templates · Explainable traces · PDF export"
    )
    st.markdown("*No cloud. No API keys. No external LLM.*")

tabs = st.tabs([
    "📂 Upload",
    "🔍 Extract",
    "📊 Rank & Topics",
    "📝 Compose",
    "🔍 Explain",
    "⬇️ Export",
])

with tabs[0]: page_upload()
with tabs[1]: page_extract()
with tabs[2]: page_rank()
with tabs[3]: page_compose()
with tabs[4]: page_explain()
with tabs[5]: page_export()