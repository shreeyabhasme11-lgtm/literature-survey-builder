"""One render function per UI tab."""
import streamlit as st
from pathlib import Path
from app.ui import state, components
from app.config import ensure_dirs, OUTPUTS, TRACES
from app.utils.logger import get_logger

log = get_logger(__name__)


def page_upload() -> None:
    st.header("📂 Upload PDFs")
    st.markdown("Upload one or more academic PDF papers to begin the survey pipeline.")

    uploaded = st.file_uploader(
        "Select PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded and st.button("▶ Ingest Files", type="primary"):
        from app.ingestion.uploader import save_uploaded_files
        from app.metadata.metadata_manager import MetadataManager
        from app.config import DATA_DIR

        ensure_dirs()
        docs, errors = save_uploaded_files(uploaded)

        for e in errors:
            st.error(e)

        if docs:
            mgr = state.get("meta_mgr")
            if mgr is None:
                mgr = MetadataManager(DATA_DIR)
                state.set_("meta_mgr", mgr)

            existing = state.get("documents") or []
            existing_ids = {d.doc_id for d in existing}
            new_docs = [d for d in docs if d.doc_id not in existing_ids]

            for d in new_docs:
                mgr.add_document(d)

            state.set_("documents", existing + new_docs)

            # Invalidate downstream pipeline state after new ingestion
            for k in [
                "pages",
                "sentences",
                "tfidf_matrix",
                "tfidf_vectorizer",
                "ranked_sentences",
                "topics",
                "survey",
                "traces",
                "compose_result",
                "selected_sentences",
                "clusters",
            ]:
                state.set_(k, None)

            state.set_("pipeline_stage", "extract")
            st.success(f"✅ Ingested {len(new_docs)} document(s)")

    docs = state.get("documents") or []
    components.render_doc_table(docs)


def page_extract() -> None:
    st.header("🔍 Text Extraction")
    docs = state.get("documents") or []

    if not docs:
        st.warning("⬅ Upload documents first (Tab 1).")
        return

    if st.button("▶ Extract Text (with OCR fallback)", type="primary"):
        from app.extraction.pdf_extractor import extract_document
        from app.extraction.text_cleaner import clean_pages
        from app.extraction.sentence_splitter import split_sentences

        all_sentences = []
        pages_map = {}
        progress = st.progress(0)
        status = st.empty()

        # Invalidate downstream state before new extraction
        for k in ["ranked_sentences", "topics", "survey", "traces", "tfidf_matrix", "tfidf_vectorizer"]:
            state.set_(k, None)

        for i, doc in enumerate(docs):
            status.write(f"Processing **{doc.filename}**…")
            try:
                pages = extract_document(doc)
                if not pages:
                    st.warning(
                        f"⚠️ No pages extracted from '{doc.filename}'. "
                        f"Check the file path: `{doc.filepath}`"
                    )
                    progress.progress((i + 1) / len(docs))
                    continue

                pages = clean_pages(pages)
                log.info(f"clean_pages() returned {len(pages)} pages for '{doc.filename}'")
                pages_map[doc.doc_id] = pages

                sents = split_sentences(pages)
                log.info(f"split_sentences() returned {len(sents)} sentences for '{doc.filename}'")
                for s in sents:
                    s.filename = doc.filename
                all_sentences.extend(sents)

            except Exception as e:
                st.error(f"Error processing '{doc.filename}': {e}")
                log.error(f"Extraction error for '{doc.filename}': {e}", exc_info=True)

            progress.progress((i + 1) / len(docs))

        status.empty()
        state.set_("pages", pages_map)
        state.set_("sentences", all_sentences)
        state.set_("pipeline_stage", "clean")

        from app.utils.fingerprint import get_pipeline_fingerprint
        state.set_("pipeline_fingerprint", get_pipeline_fingerprint(docs))

        total_pages = sum(len(v) for v in pages_map.values())
        if all_sentences:
            st.success(
                f"✅ Extracted **{len(all_sentences)}** sentences "
                f"from **{total_pages}** pages across **{len(docs)}** document(s)"
            )
        else:
            st.error(
                "❌ 0 sentences extracted. Possible causes:\n"
                "- Files were moved after upload (check `data/uploads/`)\n"
                "- PyMuPDF could not read the PDFs\n"
                "- All sentences were filtered as noise (try lowering MIN_SENTENCE_WORDS in config.py)"
            )

    pages_map = state.get("pages") or {}
    sentences = state.get("sentences") or []

    if pages_map:
        st.subheader("Extraction Summary")
        cols = st.columns(3)
        cols[0].metric("Documents", len(docs))
        cols[1].metric("Total Pages", sum(len(v) for v in pages_map.values()))
        cols[2].metric("Sentences", len(sentences))

        st.subheader("Page Preview")
        doc_names = [d.filename for d in docs if d.doc_id in pages_map]

        if not doc_names:
            st.info("No pages extracted yet.")
            return

        sel = st.selectbox("Preview document", doc_names)
        doc = next((d for d in docs if d.filename == sel), None)

        if doc and doc.doc_id in pages_map:
            page_list = pages_map[doc.doc_id]

            if not page_list:
                st.warning(f"No pages found for '{sel}'.")
                return

            total_pages = len(page_list)
            p_num = st.slider(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                key=f"page_slider_{doc.doc_id}",
            )

            idx = max(0, min(p_num - 1, total_pages - 1))
            pg = page_list[idx]

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**Page:** {pg.page_num}")
            col2.markdown(f"**OCR used:** `{pg.ocr_used}`")
            col3.markdown(f"**Characters:** `{pg.char_count}`")

            preview_text = pg.text[:3000] if pg.text else "(empty page)"
            st.text_area("Extracted text", preview_text, height=300)


def page_rank() -> None:
    st.header("📊 Sentence Ranking")
    sentences = state.get("sentences") or []

    if not sentences:
        st.warning("⬅ Extract text first (Tab 2).")
        return

    from app.utils.fingerprint import require_fresh_pipeline
    if not require_fresh_pipeline():
        return

    with st.expander("⚙️ Ranking Settings"):
        ranker = st.radio("Ranker", ["TextRank", "LexRank"], horizontal=True)
        top_k = st.slider("Top-K per section", 3, 20, 8)
        redundancy_thresh = st.slider("Redundancy threshold", 0.5, 0.99, 0.80)

    if st.button("▶ Rank Sentences", type="primary"):
        from app.representation.tfidf_vectorizer import build_tfidf
        from app.ranking.textrank_ranker import textrank_score
        from app.ranking.lexrank_ranker import lexrank_score
        from app.ranking.redundancy_filter import filter_redundant
        from app.topics.lda_modeler import fit_lda
        from app.topics.topic_labeler import label_topics
        from app.sections.section_classifier import classify_sentences

        texts = [s.text for s in sentences]

        # Invalidate downstream composition outputs before reranking
        state.set_("survey", None)
        state.set_("traces", None)

        with st.spinner("Building TF-IDF vectors…"):
            matrix, vectorizer = build_tfidf(texts)
            state.set_("tfidf_matrix", matrix)
            state.set_("tfidf_vectorizer", vectorizer)

        with st.spinner(f"Running {ranker}…"):
            scores = (
                textrank_score(matrix)
                if ranker == "TextRank"
                else lexrank_score(matrix)
            )
            for s, sc in zip(sentences, scores):
                s.textrank_score = sc

        with st.spinner("Topic modeling (LDA)…"):
            topics, topic_kws = fit_lda(texts)
            for s, t in zip(sentences, topics):
                s.topic_id = int(t)
            kw_map = label_topics(topic_kws)
            for s in sentences:
                s.topic_keywords = kw_map.get(s.topic_id, "")
            state.set_("topics", kw_map)

        with st.spinner("Classifying sections…"):
            classify_sentences(sentences)

        with st.spinner("Filtering redundant sentences…"):
            filter_redundant(sentences, matrix, redundancy_thresh)

        ranked = sorted(sentences, key=lambda s: s.textrank_score, reverse=True)
        state.set_("sentences", sentences)
        state.set_("ranked_sentences", ranked)
        state.set_("pipeline_stage", "topics")
        st.success(f"✅ Ranked {len(ranked)} sentences")

    ranked = state.get("ranked_sentences") or []
    if ranked:
        st.subheader("Top Ranked Sentences")
        components.render_sentence_table(ranked[:100])
        st.subheader("Topic Overview")
        components.render_topic_table(state.get("topics") or {})


def page_compose() -> None:
    st.header("📝 Compose Literature Survey")
    ranked = state.get("ranked_sentences") or []

    if not ranked:
        st.warning("⬅ Run ranking first (Tab 3).")
        return

    from app.utils.fingerprint import require_fresh_pipeline
    if not require_fresh_pipeline():
        return

    with st.expander("⚙️ Composition Settings"):
        ordering = st.selectbox(
            "Coherence ordering strategy",
            ["similarity", "topic", "diversity", "chronological"],
        )
        top_k = st.slider("Evidence bundles per section", 2, 12, 6)

    if st.button("▶ Generate Survey", type="primary"):
        from app.coherence.subtopic_clusterer import cluster_subtopics
        from app.coherence.diversity_controller import enforce_diversity
        from app.composition.survey_composer import compose_survey
        from app.explainability.evidence_trace import build_traces

        sentences = state.get("sentences") or []
        matrix = state.get("tfidf_matrix")

        with st.spinner("Coherence optimization…"):
            cluster_subtopics(sentences, matrix)
            sentences = enforce_diversity(sentences)

        with st.spinner("Composing synthesized sections…"):
            log.info(
                f"compose_survey() called with {len(sentences)} sentences, top_k={top_k}, ordering={ordering}"
            )
            survey = compose_survey(sentences, top_k=top_k, ordering=ordering)
            log.info(f"compose_survey() returned {len(survey)} sections")
            state.set_("survey", survey)

        with st.spinner("Building explainability traces…"):
            traces = build_traces(sentences)
            state.set_("traces", traces)

        state.set_("pipeline_stage", "export")

        if survey:
            st.success("✅ Survey generated!")
        else:
            st.warning("Survey is empty — not enough ranked sentences passed the filters.")

    survey = state.get("survey") or {}
    if survey:
        _render_survey(survey)


def _render_survey(survey: dict) -> None:
    section_icons = {
        "introduction": "📖",
        "methods": "🔬",
        "findings": "📈",
        "limitations": "⚠️",
        "research_gaps": "🔭",
        "conclusion": "🎯",
    }
    for section, paragraphs in survey.items():
        icon = section_icons.get(section, "•")
        st.markdown(f"### {icon} {section.replace('_', ' ').title()}")
        for para in paragraphs:
            st.markdown(
                f'<div class="survey-section">{para}</div>',
                unsafe_allow_html=True,
            )


def page_explain() -> None:
    st.header("🔍 Explainability")
    traces = state.get("traces") or []

    if not traces:
        st.warning("⬅ Generate the survey first (Tab 4).")
        return

    sections = sorted({t.get("section", "unknown") for t in traces})
    sel_section = st.selectbox("Filter by section", ["All"] + sections)

    filtered = (
        traces
        if sel_section == "All"
        else [t for t in traces if t.get("section") == sel_section]
    )

    st.markdown(f"Showing **{len(filtered)}** sentences")
    for trace in filtered[:50]:
        components.render_explainability_card(trace)


def page_export() -> None:
    st.header("⬇️ Export")
    survey = state.get("survey") or {}
    traces = state.get("traces") or []

    if not survey:
        st.warning("⬅ Generate the survey first (Tab 4).")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 PDF Export")
        if st.button("Generate PDF"):
            from app.export.pdf_exporter import export_pdf

            path = OUTPUTS / "literature_survey.pdf"
            try:
                actual_path = export_pdf(survey, traces, path)
                if actual_path != path:
                    st.warning(
                        f"⚠️ 'literature_survey.pdf' is locked (likely open in another app like Acrobat or Chrome). "
                        f"Saved as '{actual_path.name}' instead."
                    )
                with open(actual_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        f,
                        file_name=actual_path.name,
                        mime="application/pdf",
                    )
            except Exception as e:
                st.error(f"PDF export failed: {e}")

    with col2:
        st.subheader("📋 JSON Trace Export")
        if st.button("Generate JSON"):
            from app.export.json_exporter import export_json

            path = TRACES / "evidence_trace.json"
            try:
                export_json(traces, path)
                with open(path, "r") as f:
                    st.download_button(
                        "⬇️ Download JSON",
                        f,
                        file_name="evidence_trace.json",
                        mime="application/json",
                    )
            except Exception as e:
                st.error(f"JSON export failed: {e}")