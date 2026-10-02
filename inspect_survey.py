"""
Full pipeline run + exhaustive survey inspection.
Prints every composed paragraph with quality flags.
"""
import sys, os, re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from app.config import ensure_dirs, DATA_DIR
from app.ingestion.manifest import load_manifest
from app.extraction.pdf_extractor import extract_document
from app.extraction.text_cleaner import clean_pages
from app.extraction.sentence_splitter import split_sentences

ensure_dirs()
docs = load_manifest(DATA_DIR)
for doc in docs:
    if not Path(doc.filepath).exists():
        new_path = ROOT / "data" / "uploads" / doc.filename
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
matrix, vectorizer = build_tfidf(texts)
scores = textrank_score(matrix)
for s, sc in zip(all_sentences, scores):
    s.textrank_score = sc
topics, topic_kws = fit_lda(texts)
for s, t in zip(all_sentences, topics):
    s.topic_id = int(t)
kw_map = label_topics(topic_kws)
for s in all_sentences:
    s.topic_keywords = kw_map.get(s.topic_id, "")
classify_sentences(all_sentences)
filter_redundant(all_sentences, matrix, 0.82)
cluster_subtopics(all_sentences, matrix)
all_sentences = enforce_diversity(all_sentences)
survey = compose_survey(all_sentences, top_k=8, ordering="similarity")

# ──────────────────────────────────────────────────────────────────────────
# EXHAUSTIVE INSPECTION
# ──────────────────────────────────────────────────────────────────────────

def flag_sentence(text):
    """Return a list of issue tags for a sentence."""
    flags = []
    # Skip transition template lines
    if text.startswith("*") and text.endswith("*"):
        return ["TRANSITION"]

    # --- Strip the trailing citation [DocName, p.N] for analysis ---
    core = re.sub(r'\s*\[[^\]]{1,50}, p\.\d+\]\s*$', '', text).strip()

    # 1. Truncated citation fragment start
    if re.match(r'^\d{4}[a-z]?[\)\]]+', core):
        flags.append("TRUNC_CITE_START")
    # 2. Fused heading + sentence (ALL-CAPS run → normal text)
    if re.match(r'^[A-Z]{3,}(\s+[A-Z]{2,}){1,}\s+[A-Z][a-z]', core):
        flags.append("FUSED_HEADING")
    # 3. Task/prompt/instruction line
    if re.match(r'^(LM input|LM output|Instruction|Task|Prompt|Given the following|'
                r'Your task is|You are given|Please provide|Write a |Generate a |'
                r'Based on the above|In this task|Answer the following|'
                r'Expected output|Human:|Assistant:|System:|User:)', core, re.I):
        flags.append("TASK_PROMPT")
    # 4. Parameter / data specification with bare numbers
    if re.match(r'^[A-Za-z ]{1,20}:\s+\S.{0,30}\s+[\d\.]+\s+[\d\.]+', core):
        flags.append("PARAM_LINE")
    # 5. Starts with lowercase (mid-word fragment)
    if re.match(r'^[a-z]', core):
        flags.append("LOWERCASE_START")
    # 6. Trailing open citation
    if re.search(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*$', core):
        flags.append("TRAILING_OPEN_CITE")
    # 7. Very short after removing citations
    clean_words = [w for w in core.split() if not re.match(r'^\[\d+\]$', w)]
    if len(clean_words) < 6:
        flags.append("TOO_SHORT")
    # 8. Null bytes
    if '\x00' in text:
        flags.append("NULL_BYTE")
    # 9. Sentence ends mid-word (truncated at page break)
    if core and not core[-1] in '.!?;:)]\'"' and len(core) > 20:
        last_word = core.split()[-1] if core.split() else ""
        if last_word and last_word[-1].isalpha() and last_word[-1].islower():
            # Might be truncated — check if it looks like an incomplete sentence
            if not any(core.endswith(w) for w in ['etc', 'al', 'vs', 'eg', 'ie']):
                flags.append("TRUNCATED_END")
    # 10. Contains figure/table reference as primary content
    if re.match(r'^(Figure|Fig\.|Table|Algorithm)\s+\d', core, re.I):
        flags.append("CAPTION_REF")
    # 11. Reference list entry
    if re.match(r'^\[\d+\]\s+[A-Z]', core):
        flags.append("REF_ENTRY")
    # 12. Mostly bracket citations [1] [2] [3]...
    if clean_words:
        bracket_count = sum(1 for w in core.split() if re.match(r'^\[\d+\]$', w))
        if bracket_count / len(core.split()) > 0.30:
            flags.append("CITATION_HEAVY")
    # 13. Contains code/SPICE snippets
    if re.search(r'\.(tran|ac|dc|op|param|model|lib)\b', core, re.I):
        flags.append("CODE_SNIPPET")
    # 14. Acknowledgement / thanks
    if re.search(r'\b(thank|grateful|gratitude|acknowledge|funded by|supported by|grant\s+no)\b', core, re.I):
        flags.append("ACKNOWLEDGEMENT")
    # 15. Bare venue / conference name
    if re.match(r'^(In |in )?(Proc\.|Proceedings of|IEEE Trans|ACM Trans|ICML|NeurIPS|ICLR|CVPR)', core, re.I):
        flags.append("VENUE_LINE")
    # 16. Ends with just a comma or open bracket
    if core.endswith(',') or core.endswith('(') or core.endswith('['):
        flags.append("DANGLING_END")
    # 17. Internal cross-reference dominating the sentence
    if re.match(r'^(As shown in|See|Refer to)\s+(Figure|Fig\.|Table|Section)', core, re.I):
        flags.append("CROSS_REF")
    # 18. Exam-style imperative
    if re.match(r'^(Explain|Discuss|Describe|Define|Compare|Analyse|Analyze|Illustrate|Evaluate)\b', core, re.I):
        flags.append("EXAM_QUESTION")
    # 19. Contains "†" or "‡" marker symbols (footnotes leaked through)
    if '†' in core or '‡' in core or '�' in core:
        flags.append("SYMBOL_ARTIFACT")
    # 20. Sentence is a paper/work title
    if re.match(r'^[A-Z][a-z]+[A-Z]', core) and ':' in core[:40] and len(clean_words) < 15:
        flags.append("PAPER_TITLE")
    # 21. Duplicated sentence within the same section (checked externally)
    
    return flags if flags else ["CLEAN"]

print("\n" + "=" * 100)
print("EXHAUSTIVE SURVEY INSPECTION")
print("=" * 100)

total_issues = 0
issue_details = []

for section, paragraphs in survey.items():
    print(f"\n{'─' * 80}")
    print(f"  SECTION: {section.upper()}")
    print(f"{'─' * 80}")
    
    seen_texts = set()
    for i, para in enumerate(paragraphs):
        flags = flag_sentence(para)
        
        # Check for duplicates within section
        core = re.sub(r'\s*\[[^\]]{1,50}, p\.\d+\]\s*$', '', para).strip()
        if core in seen_texts and "TRANSITION" not in flags:
            flags.append("DUPLICATE")
        seen_texts.add(core)
        
        is_clean = flags == ["CLEAN"]
        is_transition = flags == ["TRANSITION"]
        
        marker = "  " if is_clean else "⚠️" if not is_transition else "  "
        flag_str = ", ".join(flags)
        
        # Truncate for display but show full if flagged
        display = para if len(para) < 200 else para[:197] + "..."
        
        if not is_clean and not is_transition:
            total_issues += 1
            issue_details.append((section, flag_str, para))
        
        print(f"  {marker} [{flag_str:20s}] {display}")

print(f"\n{'=' * 100}")
print(f"SUMMARY: {total_issues} flagged sentences across {len(survey)} sections")
print(f"{'=' * 100}")

if issue_details:
    print("\n--- FLAGGED SENTENCES (full text) ---")
    for i, (sec, flags, text) in enumerate(issue_details, 1):
        print(f"\n  #{i} [{sec}] FLAGS: {flags}")
        print(f"     TEXT: {text}")
else:
    print("\n  ✅ All survey sentences are clean!")
