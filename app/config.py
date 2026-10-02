"""Central configuration — all tuneable constants in one place."""
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR  = ROOT_DIR / "data"

# ── Data subdirectories ────────────────────────────────────────────────────
UPLOADS   = DATA_DIR / "uploads"
EXTRACTED = DATA_DIR / "extracted"
CLEANED   = DATA_DIR / "cleaned"
SENTENCES = DATA_DIR / "sentences"
VECTORS   = DATA_DIR / "vectors"
TOPICS    = DATA_DIR / "topics"
OUTPUTS   = DATA_DIR / "outputs"
TRACES    = DATA_DIR / "traces"


def ensure_dirs() -> None:
    """Create all data directories if they do not exist."""
    for d in (UPLOADS, EXTRACTED, CLEANED, SENTENCES,
              VECTORS, TOPICS, OUTPUTS, TRACES):
        d.mkdir(parents=True, exist_ok=True)


# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR

# ── Extraction ─────────────────────────────────────────────────────────────
OCR_CHAR_THRESHOLD   = 50      # pages with fewer chars trigger OCR
MAX_FILE_SIZE_MB     = 100     # used internally
MAX_UPLOAD_MB        = 100     # alias used by validator.py
ALLOWED_EXTENSIONS   = {".pdf"}

# ── Sentence filtering ─────────────────────────────────────────────────────
MIN_SENTENCE_WORDS   = 8       # discard very short fragments
MAX_SENTENCE_WORDS   = 120     # discard run-on extractions

# ── Ranking ────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.10   # minimum cosine sim edge in TextRank graph
TEXTRANK_DAMPING      = 0.85
TEXTRANK_ITERATIONS   = 100
REDUNDANCY_THRESHOLD  = 0.82   # cosine sim above which a sentence is redundant
TOP_K_PER_SECTION     = 8      # sentences selected per section in final survey

# ── Topic modeling ─────────────────────────────────────────────────────────
NUM_LDA_TOPICS        = 6
LDA_PASSES            = 10
USE_BERTOPIC          = False   # set True to use BERTopic instead of LDA

# ── Vector store ───────────────────────────────────────────────────────────
VECTOR_STORE          = "faiss"  # "faiss" | "chroma"
USE_EMBEDDINGS        = False    # set True to use sentence-transformers

# ── Coherence ──────────────────────────────────────────────────────────────
SOURCE_DIVERSITY_CAP  = 0.60   # max fraction of sentences from one source per section
N_SUBTOPIC_CLUSTERS   = 4

# ── Section labels (ordered for survey output) ────────────────────────────
SECTION_LABELS = [
    "introduction",
    "methods",
    "findings",
    "limitations",
    "research_gaps",
    "conclusion",
]