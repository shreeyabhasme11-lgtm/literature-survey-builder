"""
Sentence segmentation with document-agnostic, feature-based noise filtering.

Design philosophy
─────────────────
Rather than accumulating corpus-specific regex blacklists, this module detects
noise using STRUCTURAL and LINGUISTIC features that generalise across any
academic PDF domain:

    normalize_candidate()   →  fix OCR / broken-line artifacts
    extract_features()      →  compute per-sentence numeric features
    classify_noise()        →  label as code / caption / reference /
                               heading / metadata / fragment / clean
    should_keep_sentence()  →  final keep/drop decision

Adding support for a new corpus = tuning a feature threshold, not adding
another corpus-specific regex string.
"""


from __future__ import annotations

import re
import unicodedata
from typing import List, Callable, Optional

from app.metadata.schema import PageMeta, SentenceMeta
from app.config import MIN_SENTENCE_WORDS, MAX_SENTENCE_WORDS
from app.utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UNIVERSAL BROAD-CLASS REGEX
# Only patterns structurally universal across ALL academic documents.
# Do NOT add corpus-specific literal strings here.
# ══════════════════════════════════════════════════════════════════════════════

# --- Reference list entries --------------------------------------------------
_RE_REF_BRACKET  = re.compile(r"^\s*\[\d{1,3}\]\s+[A-Z]")
_RE_REF_STRIPPED = re.compile(r"^\s*\d{1,3}\]\s+[A-Z]")
_RE_REF_NUMBERED = re.compile(r"^\s*\d{1,3}\.\s+[A-Z][a-z]+,?\s+[A-Z]")

# --- Bibliography venue / metadata lines -------------------------------------
_RE_VENUE = re.compile(
    r"^(in\s+)?(Proc\.|Proceedings of|IEEE Trans|ACM Trans|"
    r"ICML|NeurIPS|ICLR|CVPR|ECCV|AAAI|IJCAI|EMNLP|ACL|NAACL)\b",
    re.IGNORECASE,
)
_RE_DOI            = re.compile(r"\b(doi|DOI)\s*:\s*10\.\d{4,}")
_RE_ARXIV          = re.compile(r"\barXiv:\d{4}\.\d{4,5}(v\d+)?\b", re.IGNORECASE)
_RE_ARXIV_PREPRINT = re.compile(r"\barXiv\s+preprint\b", re.IGNORECASE)
_RE_PREPRINT       = re.compile(r"^(Preprint|Under review|Submitted to|Published at)\b", re.IGNORECASE)
_RE_IMPACT_STMT    = re.compile(r"^(Impact Statement|Broader Impact)\b", re.IGNORECASE)

# --- Section headings --------------------------------------------------------
_RE_HDR_NUMBERED = re.compile(r"^\d+(\.\d+)*\s+[A-Z][A-Z\s\-]{4,}$")
_RE_HDR_ROMAN    = re.compile(r"^(I{1,3}|IV|V|VI{0,3}|IX|X{1,3})\.\s+[A-Z\s]{4,}$")
_RE_HDR_LETTERED = re.compile(r"^[A-Z]\.\s+[A-Z][A-Z\s]{4,}$")
_RE_HDR_ALL_CAPS = re.compile(r"^[A-Z][A-Z\s\-]{7,}$")

# --- Figure / table captions and cross-references ----------------------------
# CHANGE 1 — extended with 2 new alternates:
#   (a) "(b) Device pin level graph representation NM2 NM2..."
#   (b) "A circuit example from Figure 2 representing by..."
_RE_CAPTION = re.compile(
    r"^("
    r"Figure|Fig\.|Table|Algorithm|Listing|Appendix|Scheme|Exhibit"  # original
    r"|\([a-z]\)\s+\w"                                                # NEW: (b) Device...
    r"|A\s+\S.{0,30}?\s+example\s+from\s+Fig"                        # NEW: A circuit example from Fig...
    r")\s*[\d\w]",
    re.IGNORECASE,
)
_RE_TABLE_REF = re.compile(
    r"^(Refer to|See)\s+(Table|Figure|Fig\.|Algorithm)\s+\d+",
    re.IGNORECASE,
)
_RE_NAVIGATION = re.compile(
    r"^(As shown in|Shown in|Refer to|See|As discussed in|"
    r"As described in|As mentioned in|As illustrated in)\s+"
    r"(Figure|Fig\.|Table|Section|Sec\.|Equation|Eq\.|Appendix)\b",
    re.IGNORECASE,
)

# --- Exam / task question imperatives ----------------------------------------
_RE_EXAM_Q = re.compile(
    r"^(Explain|Discuss|Describe|Define|Compare|Analyse|Analyze|Illustrate|"
    r"Evaluate|Distinguish|Derive|Prove|Calculate|Compute|Solve|Find|List|"
    r"Enumerate|Outline|Summarise|Summarize|Justify|Critically)\b",
    re.IGNORECASE,
)

# --- Acknowledgements --------------------------------------------------------
_RE_ACK = re.compile(
    r"\b(thank|grateful|gratitude|acknowledge|funded by|supported by|"
    r"grant\s+no\.?|this work was supported)\b",
    re.IGNORECASE,
)

# --- Dangling connectors (sentence starts with a conjunction/preposition) ----
_RE_DANGLING_START = re.compile(
    r"^(also|and|but|or|so|yet|nor|including|where|which|that|"
    r"of|to|with|by|from|at|upon|whose)\b",
    re.IGNORECASE,
)

# --- Year-only citation fragment "2024)) addressing..." ----------------------
_RE_YEAR_FRAG = re.compile(r"^\d{4}[\)\]]+\s+\w")

# --- Trailing / leading OCR artifacts ----------------------------------------
_RE_TRAIL_ARXIV    = re.compile(r"\s+\d*\s*arXiv:\d{4}\.\d{4,5}v?\d*\s*$", re.IGNORECASE)
_RE_TRAIL_BRACKET  = re.compile(r"\s+\[\s*$")
_RE_LEAD_DIGIT_ART = re.compile(r"^\d+(This|These|The|A|An|We|It|In|For)\b")
_RE_CS_STAMP       = re.compile(r"\s*\[cs\.[A-Z]+\]\s*\d{1,2}\s+\w+\s+\d{4}")
_RE_TRAIL_ROM      = re.compile(r"\s+(I{1,3}|IV|V|VI{0,3}|IX|X{1,3})\.\s*$")
_RE_LEAD_BULLET    = re.compile(r"^[•·▪▸►▷\-\*]\s+")
_RE_LEAD_LIST_MARK = re.compile(r"^[a-z]\)\s+")
# CHANGE 2 — added ',' to character class: catches "3, including prompt..."
_RE_LEAD_DIGIT_LIST = re.compile(r"^\d+[\.\),]\s+[a-z]")

# --- OCR / Unicode garbage ---------------------------------------------------
_RE_BOX_SYMBOLS = re.compile(r"[\u25A0-\u25FF\u2600-\u26FF\u2700-\u27BF]{2,}")
_RE_BLOCK_LEAD  = re.compile(r"^[■□▪▫●○◆◇►▷✓✗✘✔]{1,}")

# --- Boilerplate -------------------------------------------------------------
_RE_BOILERPLATE = re.compile(
    r"^(Code(s)?\s+and\s+(the\s+)?benchmark|"
    r"There are some tips you should remember|"
    r"All rights reserved|"
    r"Permission is granted|"
    r"Copyright \d{4})\b",
    re.IGNORECASE,
)

# --- Task / prompt instruction lines that survive page-level cleaning ---------
_RE_TASK_PROMPT = re.compile(
    r"^(Given the following|Your task is|You are given|Please provide|"
    r"Write a\b|Generate a\b|Based on the above|In this task|"
    r"Answer the following|Instruction:|Task:|Prompt:|User:|System:|"
    r"Human:|Assistant:|LM input:|LM output:|Expected output:)",
    re.IGNORECASE,
)

# --- Parameter / input data lines with mixed text and bare numbers -----------
_RE_PARAM_DATA_LINE = re.compile(
    r"^[A-Za-z][A-Za-z ]{0,25}:\s+.*\d+\.\d+.*\d+\.\d+",
)

# --- Trailing open citation fragment: "(Author et al.," without close --------
_RE_TRAIL_OPEN_CITE = re.compile(
    r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*$"
)

# --- Leading year+bracket fragment: "2024) proposes..." ----------------------
_RE_LEAD_YEAR_BRACKET = re.compile(
    r"^\d{4}[a-z]?[\)\]]+[,;]?\s+"
)

# --- Truncation terminal words -----------------------------------------------
_TRUNCATION_TERMINALS: frozenset[str] = frozenset({
    "the", "a", "an", "its", "of", "to", "in", "for", "with", "and", "or",
    "this", "these", "that", "which", "due", "by", "from", "at", "as", "on",
    "upon", "both", "each", "other", "such", "any", "all", "more", "most",
    "their", "our", "your", "his", "her", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "not", "no", "is", "are",
    "was", "were",
})

# ── NEW FIX 1 — Paper title / bibliography reference lines ───────────────────
_RE_TITLE_REF = re.compile(
    r"^[A-Za-z][\w\-]*:\s+[A-Z(A][a-z].*"
    r"\b(language model|code generation|circuit|verilog|neural|survey|"
    r"evaluation|benchmark|dataset|framework|method|system|design|"
    r"learning|generation|network|transformer|attention|gnn|llm|"
    r"autoencoder|training|inference|representation|architecture)\b",
    re.IGNORECASE,
)

# ── NEW FIX 1 (supplement) — Table/figure caption "Summary of X" lines ───────
_RE_SUMMARY_CAPTION = re.compile(r"^Summary\s+of\s+\w", re.IGNORECASE)

# ── NEW FIX 2 — Variable / symbol definition lines ───────────────────────────
_RE_VAR_DEF = re.compile(
    r"^[a-zA-Z_]\s+(?:is|denotes?|represents?|refers?\s+to|equals?)\s+the\s+\w+",
    re.IGNORECASE,
)

# ── NEW FIX 3 — Fused numbered subsection heading + sentence ─────────────────
# Matches: "2) FINE-TUNING Decoder-only transformers can be fine-tuned..."
_RE_FUSED_NUMBERED_HDR = re.compile(
    r"^\d+\)\s+[A-Z][A-Z\-\s]{3,}[A-Z]\s+[A-Z][a-z]"
)

_RE_FUSED_DECIMAL_HDR = re.compile(
    r"^\d+\.\d+\s+[A-Z][A-Z\s\-]{3,}[A-Z]\s+[A-Z][a-z]"
)

# ── NEW FIX C — Fused lettered/numbered subsection heading + sentence ─────────
# Catches: "A. CANDID AUTOMATION METHODS One key challenge..."
# Catches: "2) INHERENT TRADE-OFFS BETWEEN PERFORMANCE METRICS A core challenge..."
# Catches: "A. CHALLENGES IN ANALOG IC DESIGN 1) TIME-CONSUMING SIMULATIONS..."
_RE_FUSED_LETTERED_HDR = re.compile(
    r"^([A-Z]\.\s+[A-Z][A-Z\s\-]{3,}[A-Z]|"
    r"\d+\)\s+[A-Z][A-Z\s\-]{3,}[A-Z])"
    r"\s+[A-Z0-9]",
)

# ── NEW FIX 4 — Internal proof / section cross-references ────────────────────
_RE_INTERNAL_REF = re.compile(
    r"\b(shown|proved|discussed|presented|described|demonstrated|"
    r"derived|established|defined|noted|mentioned)\s+"
    r"(its|the|our|this|in)\s+"
    r"(proof|result|analysis|derivation|section|appendix|lemma|theorem|"
    r"proposition|corollary|remark|definition|equation|claim)\s*"
    r"(?:in\s+)?(Section|Appendix|Table|Figure|Sec\.|Eq\.|Lemma|Theorem)"
    r"\s*[\d\.]+",
    re.IGNORECASE,
)

# ── NEW FIX 5 — Objective / task description lines ───────────────────────────
_RE_OBJECTIVE_LINE = re.compile(
    r"^(Provide|Present|Offer|Give)\s+a\s+"
    r"(methodological|comprehensive|detailed|systematic|brief|short|complete|"
    r"thorough|unified|broad|general|structured)\s+"
    r"(review|overview|summary|survey|analysis|description|account|examination)\b",
    re.IGNORECASE,
)

# ── CHANGE 3 — Trailing "in an FIGURE/TABLE N" terminal ──────────────────────
# Catches: "...relationships and trade-offs in an FIGURE 7."
_RE_TRAIL_FIGURE_REF = re.compile(
    r"\bin\s+an?\s+(FIGURE|TABLE|Fig\.?)\s+\d+\.?\s*$",
    re.IGNORECASE,
)

# ── NEW FIX A — Task/item enumeration caption lines ──────────────────────────
# Catches: "All analog circuit design tasks are listed with their corresponding types."
_RE_ALL_LISTED = re.compile(
    r"^All\s+\w.{0,60}\s+are\s+listed\s+with\s+(their\s+)?corresponding\b",
    re.IGNORECASE,
)

# ── NEW FIX B — Parameter / data input lines ─────────────────────────────────
# Catches: "LM input: Duty cycle options 0.1 0.3 0.5 0.7 0.9, voltage..."
# Pattern: short label + colon + mix of words and bare numbers
_RE_PARAM_LINE = re.compile(
    r"^[A-Za-z ]{1,20}:\s+\S.{0,30}\s+[\d\.]+\s+[\d\.]+",
)

# ── NEW FIX D — Sentences containing HTML/XML markup tags ────────────────
# Catches: "Only one token is marked as <select>, indicating the chosen duty cycle."
_RE_HTML_TAG = re.compile(r"<[a-z][a-z0-9]*(?:\s[^>]*)?>|</[a-z][a-z0-9]*>", re.IGNORECASE)

# ── NEW FIX E — Metric / performance table caption lines ─────────────────
# Catches: "Success rates of models trained with..." "MSE of voltage conversion ratio..."
# These are table/figure captions that describe tabular data, not prose.
_RE_METRIC_CAPTION = re.compile(
    r"^(Success rates?|MSE of|Accuracy of|RMSE of|Precision of|Recall of|F1 of|"
    r"Performance of|Comparison of|Results of|Ablation of|AUC of|"
    r"Error rates?|Hit rates?|Pass rates?|Win rates?)\s+\w",
    re.IGNORECASE,
)

# ── NEW FIX F — "is shown in Figure N" figure cross-reference sentences ──────
# Catches: "An example of the power converter...is shown in Figure 2."
# These are image/figure caption-style references that act as captions.
_RE_SHOWN_IN_FIGURE = re.compile(
    r"\bis\s+shown\s+in\s+(Figure|Fig\.|Table)\s+\d+",
    re.IGNORECASE,
)

# ── Code detection ───────────────────────────────────────────────────────────
_CODE_KEYWORD_TOKENS: tuple[str, ...] = (
    "def ", "class ", "import ", "return ", "elif ", "lambda ",
    "print(", "if __name__",
    ".tran", ".ac ", ".dc ", ".op\n", ".param", ".model", ".lib",
    ".end\n", ".subckt", ".include",
    "endmodule", "always @", "posedge ", "negedge ",
    "circuit.r(", "circuit.c(", "circuit.l(", "circuit.v(",
    "circuit.mosfet(", "circuit.bjt(", "circuit.gnd",
    "circuit.subcircuit",
    "function [",
)

_RE_CODE_SIGNALS: list[re.Pattern] = [
    re.compile(r"^(M|R|C|L|V|I|Q|D|X|E|F|G|H)\d+\s+\w+\s+\w+\s+\w+", re.IGNORECASE),
    re.compile(r"^(VDD|VIN|VOUT|GND|Vdd|Vin|Vout)\s+\w+\s+\w+\s+[\d\.]+"),
    re.compile(r"^\s*\.(tran|ac|dc|op|param|model|lib|end|subckt|include)\b", re.IGNORECASE),
    re.compile(r"\bW\s*=\s*\d+[unp]?\s+L\s*=\s*\d+[unp]?\b", re.IGNORECASE),
    re.compile(r"\bw\s*=\s*\d+e-\d+\s*,\s*l\s*=\s*\d+e-\d+\b", re.IGNORECASE),
    re.compile(r"^\s*[a-z_]\w*\s*=\s*[A-Za-z_]\w*\s*\([^)]{5,80}\)\s*$"),
    re.compile(r"^\s*[A-Za-z_]\w*\s*\(\s*['\"][^'\"]{1,30}['\"]\s*,\s*['\"]"),
    re.compile(r"^\s*(def |class |import |from \w+ import |return |yield )\b"),
    re.compile(r"\b(module\s+\w+|endmodule|always\s*@|wire\s+\w+|reg\s+\w+)\b"),
    re.compile(r"^function\s+[\[\w]"),
    re.compile(r";.*;\s*$"),
    re.compile(r"[{}\[\]]{3,}"),
    re.compile(r"(\w+\s*->\s*\w+|\w+\s*::\s*\w+)"),
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class SentenceFeatures:
    """Lightweight feature container — plain class, no dataclass."""
    __slots__ = (
        "text", "words", "word_count",
        "alpha_ratio", "digit_ratio", "upper_word_ratio",
        "symbol_density", "citation_count", "citation_density",
        "code_keyword_hits", "paren_depth",
        "ends_truncated", "has_verb", "has_ocr_garbage",
        "has_named_entity",
    )
    def __init__(self):
        self.text = ""
        self.words: list[str] = []
        self.word_count = 0
        self.alpha_ratio = 0.0
        self.digit_ratio = 0.0
        self.upper_word_ratio = 0.0
        self.symbol_density = 0.0
        self.citation_count = 0
        self.citation_density = 0.0
        self.code_keyword_hits = 0
        self.paren_depth = 0
        self.ends_truncated = False
        self.has_verb = False
        self.has_ocr_garbage = False
        self.has_named_entity = False


def extract_features(text: str) -> SentenceFeatures:
    f = SentenceFeatures()
    f.text = text

    total_chars = len(text) or 1
    f.alpha_ratio    = sum(1 for c in text if c.isalpha()) / total_chars
    f.digit_ratio    = sum(1 for c in text if c.isdigit()) / total_chars
    f.symbol_density = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    ) / total_chars

    words = text.split()
    f.words = words
    f.word_count = len(words)

    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    f.upper_word_ratio = upper_words / max(f.word_count, 1)

    bracket_cites = re.findall(r"\[\d+(?:[,\s]\d+)*\]", text)
    paren_cites   = re.findall(
        r"\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s+\d{4}[a-z]?\)", text
    )
    f.citation_count   = len(bracket_cites) + len(paren_cites)
    f.citation_density = f.citation_count / max(f.word_count, 1)

    text_lower = text.lower()
    f.code_keyword_hits = sum(
        1 for tok in _CODE_KEYWORD_TOKENS if tok in text_lower
    )

    depth = max_depth = 0
    for ch in text:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth = max(0, depth - 1)
    f.paren_depth = max_depth

    if words:
        clean_last = re.sub(r"[\[\]\.\,\;\:\!\?\"\']+", "", words[-1]).lower()
        f.ends_truncated = clean_last in _TRUNCATION_TERMINALS

    f.has_verb = bool(re.search(
        r"\b(is|are|was|were|has|have|had|can|could|will|would|should|may|"
        r"might|must|show|shows|shown|indicate|demonstrates?|proposes?|"
        r"present|presents|achiev|obtain|use[sd]?|train|evaluate|compar|"
        r"found|remain|suggest|reveal|improv|reduc|increas|decreas|allow|"
        r"enable|provid|requir|includ|contain|perform|apply|analyz|measur)\b",
        text, re.IGNORECASE,
    ))

    f.has_ocr_garbage = bool(
        _RE_BOX_SYMBOLS.search(text) or _RE_BLOCK_LEAD.match(text)
    )

    f.has_named_entity = False
    if len(words) > 1:
        for w in words[1:]:
            clean_w = re.sub(r"[\[\]\.\,\;\:\!\?\"\']+", "", w)
            if re.match(r"^[A-Z][a-z0-9]+$", clean_w):
                f.has_named_entity = True
                break

    return f


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STRUCTURAL CLASSIFIERS
# Each returns True if the text belongs to that noise class.
# ══════════════════════════════════════════════════════════════════════════════

def is_reference_like(text: str, f: SentenceFeatures) -> bool:
    """Reference list entries, bibliography lines, paper title citations."""
    if _RE_REF_BRACKET.match(text):       return True
    if _RE_REF_STRIPPED.match(text):      return True
    if _RE_REF_NUMBERED.match(text):      return True
    if _RE_DOI.search(text):              return True
    if _RE_ARXIV.search(text):            return True
    if _RE_ARXIV_PREPRINT.search(text):   return True
    if _RE_VENUE.match(text):             return True
    if _RE_TITLE_REF.search(text):        return True
    if _RE_SUMMARY_CAPTION.match(text):   return True
    if f.citation_density > 0.30 and f.word_count < 25 and not f.has_verb:
        return True
    # NEW FIX G: Short verbless paper title / bibliography entry
    # e.g. "Computer-aided design of analog and mixed-signal integrated circuits."
    # These end with "." and contain academic keywords but have no verb.
    if (not f.has_verb and f.word_count <= 18 and text.endswith('.')
            and re.search(
                r'\b(design|circuit|model|learning|network|system|method|'
                r'framework|survey|benchmark|dataset|generation|evaluation|'
                r'architecture|training|inference|representation|analysis|'
                r'detection|recognition|classification|synthesis|optimization)\b',
                text, re.IGNORECASE,
            )):
        return True
    return False


def is_section_heading_like(text: str, f: SentenceFeatures) -> bool:
    """Standalone section headings — numbered, Roman, lettered, all-caps, or fused."""
    print(f"[DEBUG splitter] is_section_heading_like: {text[:50]}")
    stripped = text.strip()
    if _RE_HDR_NUMBERED.match(stripped): return True
    if _RE_HDR_ROMAN.match(stripped):    return True
    if _RE_HDR_LETTERED.match(stripped): return True
    if _RE_HDR_ALL_CAPS.match(stripped): return True
    if (f.word_count <= 7 and not f.has_verb
            and f.upper_word_ratio > 0.45 and f.alpha_ratio > 0.70):
        return True
    cap_run = 0
    for w in f.words[:5]:
        if re.match(r"^[A-Z]{2,}[\-]?[A-Z]*$", w):
            cap_run += 1
        else:
            break
    if cap_run >= 2 and f.word_count >= 6:
        return True
    if _RE_FUSED_NUMBERED_HDR.match(stripped): return True
    if _RE_FUSED_LETTERED_HDR.match(stripped): return True  # ← NEW FIX C
    if _RE_FUSED_DECIMAL_HDR.match(stripped): return True

    # Fused all-caps label + sentence (Pattern 4)
    if re.match(r'^[A-Z]{4,}\s+[A-Z][a-z]', stripped): return True

    # Fused Title Case heading + sentence (Pattern 1)
    words = f.words
    if len(words) >= 6:
        for n in (2, 3, 4):
            prefix_words = words[:n]
            if all(re.match(r"^[A-Z][a-zA-Z]*$", w) for w in prefix_words):
                verbs = {"is", "are", "was", "were", "has", "have", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must"}
                if not any(w.lower() in verbs for w in prefix_words):
                    if re.match(r"^[A-Z]", words[n]):
                        remaining_text = " ".join(words[n:])
                        has_verb_in_rest = bool(re.search(
                            r"\b(is|are|was|were|has|have|had|can|could|will|would|should|may|"
                            r"might|must|show|shows|shown|indicate|demonstrates?|proposes?|"
                            r"present|presents|achiev|obtain|use[sd]?|train|evaluate|compar|"
                            r"found|remain|suggest|reveal|improv|reduc|increas|decreas|allow|"
                            r"enable|provid|requir|includ|contain|perform|apply|analyz|measur)\b",
                            remaining_text, re.IGNORECASE,
                        ))
                        if has_verb_in_rest:
                            return True
    return False


def is_caption_like(text: str, f: SentenceFeatures) -> bool:
    """Figure/table captions, bare cross-references, variable/param definitions."""
    if _RE_CAPTION.match(text):           return True
    if _RE_TABLE_REF.match(text):         return True
    if _RE_NAVIGATION.match(text):        return True
    if _RE_VAR_DEF.match(text):           return True
    if _RE_INTERNAL_REF.search(text):     return True
    if _RE_TRAIL_FIGURE_REF.search(text): return True   # CHANGE 4
    if _RE_ALL_LISTED.match(text):        return True   # NEW FIX A
    if _RE_PARAM_LINE.match(text):        return True   # NEW FIX B
    if _RE_HTML_TAG.search(text):         return True   # NEW FIX D: HTML/XML markup
    if _RE_METRIC_CAPTION.match(text):    return True   # NEW FIX E: metric table captions
    if _RE_SHOWN_IN_FIGURE.search(text):  return True   # NEW FIX F: "is shown in Figure N"
    # Short verb-free cross-reference to figure/table
    if (f.word_count < 14
            and not f.has_verb
            and re.search(
                r"\b(Figure|Fig\.|Table|Equation|Eq\.)\s+\d+\b",
                text, re.IGNORECASE,
            )):
        return True
    return False


def is_code_like(text: str, f: SentenceFeatures) -> bool:
    """Source code, SPICE netlists, pseudocode, API calls."""
    for pat in _RE_CODE_SIGNALS:
        if pat.search(text):
            return True
    if f.code_keyword_hits >= 2:
        return True
    if f.paren_depth >= 4 and f.alpha_ratio < 0.50:
        return True
    if f.symbol_density > 0.32 and f.alpha_ratio < 0.55:
        return True
    if text.count(';') >= 2:
        return True
    if text.count('{') + text.count('}') >= 2:
        return True
    return False


def is_metadata_like(text: str, f: SentenceFeatures) -> bool:
    """Paper metadata: preprint notices, boilerplate, acknowledgements, prompts."""
    if _RE_PREPRINT.match(text):      return True
    if _RE_IMPACT_STMT.match(text):   return True
    if _RE_BOILERPLATE.match(text):   return True
    if _RE_ACK.search(text):          return True
    if _RE_TASK_PROMPT.match(text):   return True   # NEW: task/prompt lines
    if _RE_PARAM_DATA_LINE.match(text): return True  # NEW: param data lines
    
    # Paper-internal parameter description lines (Requested Fix 4)
    if re.search(r'\b(duty cycle|voltage conversion ratio|efficiency)\b', text) and f.word_count < 20: return True

    # "This paper proposes" / "This work proposes" self-introduction sentences (Requested Fix 2)
    if re.match(r'^(This paper|This work|In this paper|In this work)\s+(proposes|presents|introduces|describes)\b', text, re.IGNORECASE): return True

    # "We examine / we investigate / we study" scope sentences (Requested Fix 3)
    if re.search(r'\bwe (examine|investigate|study|present|introduce|propose|describe|evaluate)\b', text): return True

    # Technical definition sentences (Requested Fix 4)
    if re.match(r'^The \w[\w\s]+ is defined as\b', text): return True

    # "We focus on / we aim to" paper-scope declarations (Requested Fix 6)
    if re.search(r'\bin this work\b', text, re.IGNORECASE): return True
    if re.match(r'^We (focus|aim|seek|intend|target)\b', text): return True

    # Paper table-of-contents / section-pointer sentences (Requested Fix 5)
    if re.match(r'^(Finally|First|Second|Third|Lastly),?\s+section\s+', text, re.IGNORECASE): return True
    if re.search(r'\bsection\s+(VI|VII|VIII|IV|V|II|III|I)\b', text, re.IGNORECASE): return True

    return False


def is_exam_question(text: str) -> bool:
    """Exam question imperatives and objective/task description lines."""
    if _RE_OBJECTIVE_LINE.match(text): return True
    return bool(_RE_EXAM_Q.match(text))


def is_fragment(text: str, f: SentenceFeatures) -> bool:
    """Linguistic fragments: dangling connectors, truncated sentences, open citations."""
    print(f"[DEBUG splitter] is_fragment: {text[:50]}")
    if _RE_DANGLING_START.match(text): return True
    if _RE_YEAR_FRAG.match(text):      return True
    if _RE_LEAD_YEAR_BRACKET.match(text): return True  # NEW: "2024) proposes..."
    if re.match(r"^[A-Z],\s+\w", text): return True  # catches "I, due to their..."
    if _RE_TRAIL_OPEN_CITE.search(text): return True  # NEW: trailing "(Author et al.,"
    
    # Ellipsis / Truncated mid-formula sentence checking (Pattern 3)
    if re.search(r'\.\s*\.\s*\.?\s*$', text): return True

    # Math/formula definition lines (Requested Fix 1)
    if re.search(r'[=×→⊂∈∀∃]', text) and f.word_count < 25: return True
    if re.match(r'^Here,?\s+\w+\s*:', text): return True

    # Sentences referencing Figure N or Table N (Requested Fix 3)
    if re.search(r'\b(Figure|Table)\s+\d+\b', text): return True

    # Internal section cross-reference sentences (Requested Fix 5)
    if re.search(r'\bSection\s+\d+[\.\d]*\b', text): return True

    # Numbered list item fragments starting with digit + paren (Requested Fix 1)
    if re.match(r'^\d+\)\s+[a-z]', text): return True

    # Bibliography/reference lines containing journal metadata (Requested Fix 2)
    if re.search(r',"\s+[A-Z][a-z]', text): return True
    if re.search(r'\b(vol\.|pp\.|doi:|arXiv|proceedings|conference|journal|NeurIPS|ICML|ICLR|IEEE)\b', text, re.IGNORECASE): return True

    # Vague low-information sentences (Requested Fix 4)
    if f.word_count < 14 and not f.has_named_entity and f.upper_word_ratio < 0.1: return True

    # Figure caption / leaderboard lines (Requested Fix 1 from prev step)
    if re.search(r'Figure\s+\d+\s*:', text): return True
    if re.search(r'(Table\s+\d+\s*:|Leaderboard)', text): return True

    # Task prompt / imperative instruction lines (Requested Fix 2 from prev step)
    if re.match(r'^(Design|Please|Make sure|Note that|Ensure|Use)\b', text): return True

    # Truncated sentences ending with a bare number or footnote ref (Requested Fix 3 from prev step)
    if re.search(r'\s+\d{1,3}\s*$', text) and f.word_count < 30: return True

    # Enumeration / definition checking (Pattern 5)
    if re.match(r'^.{0,60}\([1-9]\)\s+the\b', text): return True
    if re.search(r'\([1-9]\).{0,30}\([2-9]\)', text): return True

    # Sentence starting with lowercase = broken ligature / mid-word fragment
    if re.match(r"^[a-z]", text): return True
    if f.ends_truncated and f.word_count < 25 and not f.has_verb:
        return True

    # ── Truncated mid-word end detection ─────────────────────────────────
    # Catches sentences cut at page/column boundary mid-word.
    if f.word_count > 0:
        stripped = text.rstrip()
        last_char = stripped[-1] if stripped else ""
        # Sentence ends without any terminal punctuation and last word
        # looks like a truncated word fragment
        if last_char.isalpha():
            last_word = f.words[-1].rstrip(".,;:!?")
            last_word_lower = last_word.lower()
            # NEW FIX H: Sentence ends on a function word (preposition/article/determiner)
            # with no terminal punctuation → clearly truncated mid-clause.
            # e.g. "...remain intractable to such abstraction due to their"
            if last_word_lower in _TRUNCATION_TERMINALS:
                return True
            # Short last-word that isn't a real stopper
            if (len(last_word) >= 2
                    and last_word_lower not in {
                        "short", "long", "well", "more", "less", "much",
                        "them", "here", "there", "thus", "hence", "since",
                    }):
                # Check: last 3 chars have no vowel → likely truncated
                tail = last_word[-3:].lower() if len(last_word) >= 3 else last_word.lower()
                vowels = set("aeiou")
                if not vowels.intersection(tail):
                    return True  # e.g. "insufcien", "complexi[ty]" → "complxty" wouldn't have vowels

        # Sentence ends with semicolon = incomplete enumeration
        if last_char == ";":
            return True

    if f.word_count > 0:
        numeric = sum(
            1 for w in f.words if re.match(r"^[\d\.\,\%\-\+\/\:]+$", w)
        )
        if numeric / f.word_count > 0.60:
            return True
    if f.word_count > 0:
        brackets = sum(
            1 for w in f.words if re.match(r"^\[\d+\]$", w)
        )
        if brackets / f.word_count > 0.30:
            return True
    return False


def is_ocr_garbage(text: str, f: SentenceFeatures) -> bool:
    """OCR artefacts: symbol runs, garbled low-alpha content."""
    if f.has_ocr_garbage:                              return True
    if f.alpha_ratio < 0.35 and f.word_count < 15:    return True
    if f.digit_ratio > 0.45 and f.alpha_ratio < 0.30: return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_candidate(text: str) -> str:
    """Remove OCR / PDF extraction artifacts from a raw sentence candidate."""
    text = unicodedata.normalize("NFKC", text)
    text = _RE_TRAIL_ARXIV.sub("", text)
    text = _RE_TRAIL_BRACKET.sub("", text)
    text = _RE_LEAD_DIGIT_ART.sub(r"\1", text)
    text = _RE_CS_STAMP.sub("", text)
    text = re.sub(r"\s*\[cs\.?\s*$", "", text)
    text = _RE_TRAIL_ROM.sub("", text)
    text = _RE_LEAD_BULLET.sub("", text)
    text = _RE_LEAD_LIST_MARK.sub("", text)
    text = _RE_LEAD_DIGIT_LIST.sub("", text)   # CHANGE 2 — now catches "3, ..."
    # Strip trailing open parenthetical/bracket citation fragments
    text = re.sub(r"\s*[\(\[]\s*$", "", text)
    # Strip trailing comma+space at end (broken citation tail)
    text = re.sub(r",\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — NOISE LABELS AND MAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class NoiseLabel:
    CLEAN     = "clean"
    CODE      = "code"
    CAPTION   = "caption"
    REFERENCE = "reference"
    HEADING   = "heading"
    METADATA  = "metadata"
    FRAGMENT  = "fragment"
    EXAM      = "exam_question"
    OCR       = "ocr_garbage"
    TOO_SHORT = "too_short"
    TOO_LONG  = "too_long"


def classify_noise(text: str) -> str:
    """
    Return a NoiseLabel for the sentence.
    Evaluated in priority order: cheapest / most certain checks first.
    """
    f = extract_features(text)

    if f.word_count < MIN_SENTENCE_WORDS: return NoiseLabel.TOO_SHORT
    if f.word_count > MAX_SENTENCE_WORDS: return NoiseLabel.TOO_LONG

    if is_ocr_garbage(text, f):           return NoiseLabel.OCR
    if is_code_like(text, f):             return NoiseLabel.CODE
    if is_reference_like(text, f):        return NoiseLabel.REFERENCE
    if is_section_heading_like(text, f):  return NoiseLabel.HEADING
    if is_caption_like(text, f):          return NoiseLabel.CAPTION
    if is_metadata_like(text, f):         return NoiseLabel.METADATA
    if is_exam_question(text):            return NoiseLabel.EXAM
    if is_fragment(text, f):              return NoiseLabel.FRAGMENT

    return NoiseLabel.CLEAN


def should_keep_sentence(text: str) -> tuple[bool, str]:
    """Returns (keep: bool, noise_label: str)."""
    label = classify_noise(text)
    return label == NoiseLabel.CLEAN, label


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HEADER / FOOTER DEDUPLICATION (document-level pass)
# Lines appearing verbatim on >= N pages are structural, not content.
# ══════════════════════════════════════════════════════════════════════════════

def detect_repeated_lines(pages: List[PageMeta], min_repeat: int = 3) -> set[str]:
    """Return lines present on >= min_repeat distinct pages."""
    from collections import Counter
    line_page_count: Counter = Counter()
    for page in pages:
        if not page.text:
            continue
        seen_this_page: set[str] = set()
        for line in page.text.splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen_this_page:
                line_page_count[stripped] += 1
                seen_this_page.add(stripped)
    return {line for line, cnt in line_page_count.items() if cnt >= min_repeat}


def remove_repeated_lines(text: str, repeated: set[str]) -> str:
    if not repeated:
        return text
    return "\n".join(
        line for line in text.splitlines() if line.strip() not in repeated
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SENTENCE SPLITTER BACKEND
# ══════════════════════════════════════════════════════════════════════════════

_splitter_fn: Callable[[str], List[str]] | None = None


def _get_splitter() -> Callable[[str], List[str]]:
    """Return best available sentence splitter: spaCy → NLTK → regex."""
    try:
        # pyrefly: ignore [missing-import]
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        nlp.add_pipe("sentencizer")
        def spacy_split(text: str) -> List[str]:
            return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        log.info("Sentence splitter: spaCy sentencizer")
        return spacy_split
    except Exception:
        pass

    try:
        # pyrefly: ignore [missing-import]
        import nltk
        for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(resource)
                break
            except LookupError:
                pass
        else:
            nltk.download("punkt",     quiet=True)
            nltk.download("punkt_tab", quiet=True)
        # pyrefly: ignore [missing-import]
        from nltk.tokenize import sent_tokenize
        log.info("Sentence splitter: NLTK punkt")
        return sent_tokenize
    except Exception:
        pass

    _SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    def regex_split(text: str) -> List[str]:
        return [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    log.info("Sentence splitter: regex fallback")
    return regex_split


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def split_sentences(pages: List[PageMeta]) -> List[SentenceMeta]:
    print(f"[DEBUG splitter] split_sentences: splitting {len(pages)} pages")
    """
    Main entry point. Segments all pages into filtered SentenceMeta objects.

    Per-document pipeline:
        1. Detect repeated header/footer lines (document-level pass)
        2. Remove those lines from each page's text
        3. Split remaining text into raw candidates
        4. Normalize each candidate (strip OCR artifacts)
        5. Feature-extract → classify noise → keep only CLEAN sentences
    """
    global _splitter_fn
    if _splitter_fn is None:
        _splitter_fn = _get_splitter()

    repeated_lines = detect_repeated_lines(pages, min_repeat=3)
    if repeated_lines:
        log.debug(f"Suppressing {len(repeated_lines)} repeated header/footer lines")

    sentences: List[SentenceMeta] = []
    sent_counter = 0
    noise_stats: dict[str, int] = {}

    for page in pages:
        if not page.text or not page.text.strip():
            continue

        page_text = remove_repeated_lines(page.text, repeated_lines)
        if not page_text.strip():
            continue

        raw_candidates = _splitter_fn(page_text)

        for raw in raw_candidates:
            candidate = normalize_candidate(raw)
            if not candidate:
                continue

            keep, label = should_keep_sentence(candidate)

            if not keep:
                noise_stats[label] = noise_stats.get(label, 0) + 1
                continue

            sent_counter += 1
            sent_id = f"{page.doc_id}_p{page.page_num}_s{sent_counter}"

            sentences.append(SentenceMeta(
                sent_id=sent_id,
                doc_id=page.doc_id,
                filename="",
                page_num=page.page_num,
                text=candidate,
                word_count=len(candidate.split()),
            ))

    log.info(
        f"split_sentences: {len(sentences)} kept from {len(pages)} pages | "
        f"noise: {noise_stats}"
    )
    return sentences