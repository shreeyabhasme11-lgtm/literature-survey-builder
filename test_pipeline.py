"""
End-to-end pipeline test — runs the full extraction → cleaning → splitting →
ranking → composition pipeline and dumps the survey output for inspection.
Run: python test_pipeline.py
"""
import sys
import os
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from app.config import ensure_dirs, DATA_DIR
from app.ingestion.manifest import load_manifest
from app.extraction.pdf_extractor import extract_document
from app.extraction.text_cleaner import clean_pages
from app.extraction.sentence_splitter import split_sentences

ensure_dirs()

# Load documents from manifest
docs = load_manifest(DATA_DIR)
print(f"Loaded {len(docs)} documents from manifest")

if not docs:
    print("No documents found in manifest. Upload PDFs first.")
    sys.exit(1)

# Fix filepaths if they point to old location
for doc in docs:
    old_path = Path(doc.filepath)
    if not old_path.exists():
        # Try the current project's uploads dir
        new_path = ROOT / "data" / "uploads" / doc.filename
        if new_path.exists():
            doc.filepath = str(new_path)
            print(f"  Fixed path for '{doc.filename}' -> {new_path}")
        else:
            print(f"  WARNING: '{doc.filename}' not found at either location!")

all_sentences = []
for doc in docs:
    print(f"\nProcessing '{doc.filename}'...")
    pages = extract_document(doc)
    if not pages:
        print(f"  No pages extracted!")
        continue
    print(f"  Extracted {len(pages)} pages")

    pages = clean_pages(pages)
    print(f"  Cleaned -> {len(pages)} pages")

    sents = split_sentences(pages)
    for s in sents:
        s.filename = doc.filename
    print(f"  Split -> {len(sents)} sentences")
    all_sentences.extend(sents)

print(f"\n{'='*80}")
print(f"TOTAL SENTENCES: {len(all_sentences)}")
print(f"{'='*80}")

# Show first 30 sentences for inspection
print("\nFIRST 30 SENTENCES:")
print("-" * 80)
for i, s in enumerate(all_sentences[:30]):
    print(f"  [{i+1}] {s.text[:120]}...")

# Now do ranking + composition if we have enough sentences
if len(all_sentences) < 5:
    print("Too few sentences for composition test.")
    sys.exit(0)

from app.representation.tfidf_vectorizer import build_tfidf
from app.ranking.textrank_ranker import textrank_score
from app.ranking.redundancy_filter import filter_redundant
from app.topics.lda_modeler import fit_lda
from app.topics.topic_labeler import label_topics
from app.sections.section_classifier import classify_sentences
from app.coherence.subtopic_clusterer import cluster_subtopics
from app.coherence.diversity_controller import enforce_diversity
from app.composition.survey_composer import compose_survey

texts = [s.text for s in all_sentences]
print("\nBuilding TF-IDF vectors...")
matrix, vectorizer = build_tfidf(texts)

print("Running TextRank...")
scores = textrank_score(matrix)
for s, sc in zip(all_sentences, scores):
    s.textrank_score = sc

print("Topic modeling (LDA)...")
topics, topic_kws = fit_lda(texts)
for s, t in zip(all_sentences, topics):
    s.topic_id = int(t)
kw_map = label_topics(topic_kws)
for s in all_sentences:
    s.topic_keywords = kw_map.get(s.topic_id, "")

print("Classifying sections...")
classify_sentences(all_sentences)

print("Filtering redundant sentences...")
filter_redundant(all_sentences, matrix, 0.82)

print("Clustering subtopics...")
cluster_subtopics(all_sentences, matrix)

print("Enforcing diversity...")
all_sentences = enforce_diversity(all_sentences)

print("Composing survey...")
survey = compose_survey(all_sentences, top_k=8, ordering="similarity")

print(f"\n{'='*80}")
print("SURVEY OUTPUT")
print(f"{'='*80}")

for section, paragraphs in survey.items():
    print(f"\n### {section.upper()}")
    print("-" * 60)
    for para in paragraphs:
        print(f"  {para[:200]}")

# Flag potential remaining issues
print(f"\n{'='*80}")
print("QUALITY CHECKS — Potential remaining issues:")
print(f"{'='*80}")

import re
issue_count = 0
for section, paragraphs in survey.items():
    for para in paragraphs:
        if para.startswith("*"):
            continue  # skip transition templates
        issues = []
        # Check truncated citation: year+bracket at start
        if re.match(r"^\d{4}[\)\]]+\s", para):
            issues.append("TRUNCATED_CITATION_START")
        # Check fused heading
        if re.match(r"^[A-Z]{3,}(\s+[A-Z]{2,}){2,}\s+[A-Z][a-z]", para):
            issues.append("FUSED_HEADING")
        # Check task prompt
        if re.match(r"^(LM input|LM output|Instruction|Task|Prompt):", para, re.IGNORECASE):
            issues.append("TASK_PROMPT")
        # Check parameter line
        if re.match(r"^[A-Za-z ]{1,20}:\s+\S.{0,30}\s+[\d\.]+\s+[\d\.]+", para):
            issues.append("PARAM_LINE")
        # Check trailing open citation
        if re.search(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*$", para):
            issues.append("TRAILING_OPEN_CITE")
        if issues:
            issue_count += 1
            print(f"  [{section}] {', '.join(issues)}: {para[:120]}")

if issue_count == 0:
    print("  ✅ No obvious issues detected!")
else:
    print(f"\n  ⚠️ {issue_count} potential issues found.")
