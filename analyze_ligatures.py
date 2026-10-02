"""Analyze raw PDF text to understand the null byte / ligature pattern."""
import sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from app.config import ensure_dirs, DATA_DIR
from app.ingestion.manifest import load_manifest
from app.extraction.pdf_extractor import extract_document

ensure_dirs()
docs = load_manifest(DATA_DIR)
for doc in docs:
    if not Path(doc.filepath).exists():
        new_path = ROOT / "data" / "uploads" / doc.filename
        if new_path.exists():
            doc.filepath = str(new_path)

# Focus on RCGenaiPaper4 page 1 which has the most issues
doc = next(d for d in docs if d.filename == "RCGenaiPaper4.pdf")
pages = extract_document(doc)

# Look at raw text of page 1 for null bytes
page = pages[0]
text = page.text

# Find all null bytes with surrounding context
print("=== NULL BYTE CONTEXTS IN PAGE 1 ===")
for m in re.finditer(r'.{0,15}\x00.{0,15}', text):
    snippet = repr(m.group())
    print(f"  Position {m.start()}: {snippet}")

# Also check page 2-5
for pn in range(1, min(5, len(pages))):
    page = pages[pn]
    text = page.text
    nulls = [(m.start(), repr(m.group())) for m in re.finditer(r'.{0,10}\x00.{0,10}', text)]
    if nulls:
        print(f"\n=== NULL BYTE CONTEXTS IN PAGE {pn+1} ===")
        for pos, snippet in nulls[:10]:
            print(f"  Position {pos}: {snippet}")

# Check what words have missing ligatures
# Find words that look broken: known suffix patterns after ligature removal
print("\n=== LIGATURE LOSS ANALYSIS (page 1) ===")
text = pages[0].text
# After removing \x00, what would remain?
cleaned = text.replace('\x00', '')
# Find words that look like they lost 'fi':
for m in re.finditer(r'\b(\w*)(nd|ndings|nite|nish|ned|ner|nally|rst|re|eld|gure|le|lter|x|ll|rm)\b', cleaned):
    context_start = max(0, m.start() - 20)
    context_end = min(len(cleaned), m.end() + 20)
    context = cleaned[context_start:context_end]
    print(f"  Possible broken word: '{m.group()}' in context: ...{context}...")
