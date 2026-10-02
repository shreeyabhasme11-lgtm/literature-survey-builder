"""Reusable Streamlit UI components."""
import streamlit as st
import pandas as pd
from typing import List
from app.metadata.schema import SentenceMeta, DocumentMeta


def render_doc_table(docs: List[DocumentMeta]) -> None:
    if not docs:
        st.info("No documents uploaded yet.")
        return
    rows = [{
        "ID": d.doc_id,
        "Filename": d.filename,
        "Size (KB)": round(d.size_bytes / 1024, 1),
        "Pages": d.num_pages,
        "Uploaded": d.upload_ts[:19].replace("T", " "),
    } for d in docs]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_sentence_table(sentences: List[SentenceMeta], max_rows: int = 200) -> None:
    if not sentences:
        st.info("No sentences available.")
        return
    rows = [{
        "ID": s.sent_id,
        "Source": s.filename,
        "Page": s.page_num,
        "Section": s.section,
        "TR Score": round(s.textrank_score, 4),
        "Topic": s.topic_id,
        "Cluster": s.subtopic_cluster,
        "Redundant": s.redundant,
        "Text": s.text[:120] + ("…" if len(s.text) > 120 else ""),
    } for s in sentences[:max_rows]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_topic_table(topics: dict) -> None:
    if not topics:
        st.info("No topics modeled yet.")
        return
    rows = [{"Topic ID": tid, "Keywords": kws} for tid, kws in topics.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def progress_sidebar(stage: str) -> None:
    stages = ["upload", "extract", "clean", "rank", "topics", "compose", "export"]
    idx = stages.index(stage) if stage in stages else 0
    st.sidebar.markdown("### Pipeline Progress")
    for i, s in enumerate(stages):
        icon = "✅" if i < idx else ("🔄" if i == idx else "⬜")
        st.sidebar.markdown(f"{icon} {s.capitalize()}")


def render_explainability_card(trace: dict) -> None:
    with st.expander(f"🔍 {trace.get('sent_id', '?')} — {trace.get('section', '?').upper()}"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Text:** {trace.get('text', '')}")
            st.markdown(f"**Source:** `{trace.get('filename', '')}` p.{trace.get('page_num', '?')}")
            st.markdown(f"**TextRank Score:** `{trace.get('textrank_score', 0):.4f}`")
        with col2:
            st.markdown(f"**Topic ID:** `{trace.get('topic_id', -1)}` — {trace.get('topic_keywords', '')}")
            st.markdown(f"**Section:** `{trace.get('section', '')}`")
            st.markdown(f"**Ordering Reason:** {trace.get('ordering_reason', 'N/A')}")
            st.markdown(f"**Coherence Score:** `{trace.get('coherence_score', 0):.4f}`")
            st.markdown(f"**Subtopic Cluster:** `{trace.get('subtopic_cluster', -1)}`")
            st.markdown(f"**Synthesis Role:** `{trace.get('synthesis_role', '')}`")
            st.markdown(f"**Evidence Group:** `{trace.get('evidence_group', '')}`")