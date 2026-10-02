"""Focused quality check — look for remaining artifacts in survey output."""
import sys
import os
import re
from pathlib import Path

# Force UTF-8 output
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
    old_path = Path(doc.filepath)
    if not old_path.exists():
        new_path = ROOT / "data" / "uploads" / doc.filename
        if new_path.exists():
            doc.filepath = str(new_path)

all_sentences = []
for doc in docs:
    pages = extract_document(doc)
    if not pages:
        continue
    pages = clean_pages(pages)
    sents = split_sentences(pages)
    for s in sents:
        s.filename = doc.filename
    all_sentences.extend(sents)

print(f"Total sentences: {len(all_sentences)}")

# Check for specific artifact types
print("\n=== ARTIFACT TYPE 1: Truncated citation fragments ===")
count = 0
for s in all_sentences:
    if re.match(r"^\d{4}[\)\]]+\s", s.text):
        count += 1
        print(f"  {s.text[:120]}")
print(f"  Count: {count}")

print("\n=== ARTIFACT TYPE 2: Fused heading+sentence ===")
count = 0
for s in all_sentences:
    if re.match(r"^[A-Z]{3,}(\s+[A-Z]{2,}){2,}\s+[A-Z][a-z]", s.text):
        count += 1
        print(f"  {s.text[:120]}")
print(f"  Count: {count}")

print("\n=== ARTIFACT TYPE 3: Parameter/input/task prompt lines ===")
count = 0
for s in all_sentences:
    if re.match(r"^(LM input|LM output|Instruction|Task|Prompt|Given the following|Your task is|Expected output):", s.text, re.IGNORECASE):
        count += 1
        print(f"  {s.text[:120]}")
print(f"  Count: {count}")

print("\n=== ARTIFACT TYPE 4: Sentences starting mid-word ===")
count = 0
for s in all_sentences:
    if re.match(r"^[a-z]{2,}\s", s.text):
        count += 1
        if count <= 10:
            print(f"  {s.text[:120]}")
print(f"  Count: {count}")

print("\n=== ARTIFACT TYPE 5: Trailing open citation ===")
count = 0
for s in all_sentences:
    if re.search(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*$", s.text):
        count += 1
        if count <= 10:
            print(f"  {s.text[:120]}")
print(f"  Count: {count}")

print("\n=== ARTIFACT TYPE 6: Sentences with null bytes ===")
count = 0
for s in all_sentences:
    if '\x00' in s.text:
        count += 1
        if count <= 5:
            print(f"  {repr(s.text[:120])}")
print(f"  Count: {count}")

print("\n=== ARTIFACT TYPE 7: Very short sentences (< 4 words of actual text) ===")
count = 0
for s in all_sentences:
    real_words = [w for w in s.text.split() if not re.match(r"^\[\d+\]$", w)]
    if len(real_words) < 5:
        count += 1
        if count <= 10:
            print(f"  {s.text[:120]}")
print(f"  Count: {count}")

print("\nDone!")
