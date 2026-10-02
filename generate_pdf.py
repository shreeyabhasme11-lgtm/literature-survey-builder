"""Standalone script to regenerate the literature survey PDF."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from app.config import ensure_dirs, DATA_DIR, OUTPUTS
from app.ingestion.manifest import load_manifest
from app.extraction.pdf_extractor import extract_document
from app.extraction.text_cleaner import clean_pages
from app.extraction.sentence_splitter import split_sentences
from app.representation.tfidf_vectorizer import build_tfidf
from app.ranking.textrank_ranker import textrank_score
from app.ranking.redundancy_filter import filter_redundant
from app.topics.lda_modeler import fit_lda
from app.topics.topic_labeler import label_topics
from app.sections.section_classifier import classify_sentences
from app.coherence.subtopic_clusterer import cluster_subtopics
from app.coherence.diversity_controller import enforce_diversity
from app.composition.survey_composer import compose_survey
from app.export.pdf_exporter import export_pdf
from app.explainability.evidence_trace import build_traces

ensure_dirs()
docs = load_manifest(DATA_DIR)
ROOT = Path('.').resolve()
for doc in docs:
    if not Path(doc.filepath).exists():
        new_path = ROOT / 'data' / 'uploads' / doc.filename
        if new_path.exists():
            doc.filepath = str(new_path)

all_sentences = []
for doc in docs:
    pages = extract_document(doc)
    if not pages: continue
    pages = clean_pages(pages)
    sents = split_sentences(pages)
    for s in sents:
        s.filename = doc.filename
    all_sentences.extend(sents)

print(f"Extracted {len(all_sentences)} sentences from {len(docs)} docs")

texts = [s.text for s in all_sentences]
matrix, vectorizer = build_tfidf(texts)
scores = textrank_score(matrix)
for s, sc in zip(all_sentences, scores):
    s.textrank_score = sc
topics, topic_kws = fit_lda(texts)
for s, t in zip(all_sentences, topics):
    s.topic_id = int(t)
kw_map = label_topics(topic_kws)
for s in all_sentences:
    s.topic_keywords = kw_map.get(s.topic_id, '')
classify_sentences(all_sentences)
filter_redundant(all_sentences, matrix, 0.82)
cluster_subtopics(all_sentences, matrix)
all_sentences = enforce_diversity(all_sentences)
survey = compose_survey(all_sentences, top_k=8, ordering='similarity')
traces = build_traces(all_sentences)

out_path = OUTPUTS / 'literature_survey_latest.pdf'
export_pdf(survey, traces, out_path)
print(f'✅ PDF exported to: {out_path}')
