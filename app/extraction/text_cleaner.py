"""Page-level text cleaning before sentence splitting."""
import re
from typing import List
from app.metadata.schema import PageMeta
from app.utils.logger import get_logger


log = get_logger(__name__)


_HEADER_FOOTER_MIN_PAGES = 3


# ── Patterns ──────────────────────────────────────────────────────────────────
_FUSED_NUMERIC_HEADING = re.compile(
    r"^(\d+(\.\d+)*)\s+[A-Z][A-Z\s]{5,}(?=[A-Z][a-z])",
    re.MULTILINE,
)
_FUSED_ROMAN_HEADING = re.compile(
    r"^(I{1,3}|IV|V|VI{0,3}|IX|X{1,3})\.\s+[A-Z][A-Z\s]{3,}[:\-]\s*",
    re.MULTILINE,
)
_STANDALONE_NUMBERED_HEADING = re.compile(
    r"^\d+\s+[A-Z][A-Z\s]{5,}$",
    re.MULTILINE,
)
_APPENDIX_HEADING = re.compile(
    r"^[A-Z]\.\d+\s+[A-Z][A-Z\s]{4,}$",
    re.MULTILINE,
)
_ARXIV_FOOTER = re.compile(
    r"\s*\d*\s*arXiv:\d{4}\.\d{4,5}v?\d*\b",
    re.IGNORECASE,
)
_ALLCAPS_HEADING_PREFIX = re.compile(
    r"^([A-Z]{4,}(?:\s+[A-Z]{2,})*)\s+(?=[A-Z][a-z])",
    re.MULTILINE,
)
_PROMPT_BOILERPLATE = re.compile(
    r"^There are some tips you should remember.*$",
    re.MULTILINE | re.IGNORECASE,
)
_ASSISTANT_BOILERPLATE = re.compile(
    r"\(same with Assistant[^\)]*\)",
    re.IGNORECASE,
)
_FOOTNOTE_SUPERSCRIPT = re.compile(r"\s+\d+(This|These|The|A|An|We|It|In|For)\b")
_SECTION_LABEL_LINE = re.compile(
    r"^\s*(Abstract|Keywords?|References?|Bibliography|Acknowledgements?|Appendix)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_EMAIL_URL_LINE = re.compile(
    r"^\s*(https?://\S+|\S+@\S+\.\S+)\s*$",
    re.MULTILINE,
)
# ── Additional boilerplate patterns ──────────────────────────────────────────
_TASK_PROMPT_LINE = re.compile(
    r"^(Given the following|Your task is to|You are given|Please provide|"
    r"Write a|Generate a|Based on the above|In this task|Answer the following|"
    r"Instruction:|Task:|Prompt:|User:|System:|Human:|Assistant:)\b",
    re.MULTILINE | re.IGNORECASE,
)
_IO_LABEL_LINE = re.compile(
    r"^\s*(Input|Output|Parameters?|Args?|Returns?|Yields?|Raises?|Note[s]?|"
    r"Example[s]?|Default[s]?|Type|Format|Description|Constraints?)\s*:\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_ISOLATED_CITATION_FRAG = re.compile(
    r"^\s*\(?\d{4}[a-z]?[\)\]]+[,\.;]?\s*$",
    re.MULTILINE,
)
_BARE_AUTHOR_CITE = re.compile(
    r"^\s*[A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\(?\d{4}[a-z]?\)?[,\.;]?\s*$",
    re.MULTILINE,
)
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_WHITESPACE = re.compile(r"[ \t]{2,}")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


# ── Private helpers ───────────────────────────────────────────────────────────

def _strip_fused_headings(text: str) -> str:
    text = _FUSED_NUMERIC_HEADING.sub("", text)
    text = _FUSED_ROMAN_HEADING.sub("", text)
    text = _STANDALONE_NUMBERED_HEADING.sub("", text)
    text = _APPENDIX_HEADING.sub("", text)
    text = _ALLCAPS_HEADING_PREFIX.sub("", text)
    return text


def _strip_boilerplate(text: str) -> str:
    text = _PROMPT_BOILERPLATE.sub("", text)
    text = _ASSISTANT_BOILERPLATE.sub("", text)
    text = _SECTION_LABEL_LINE.sub("", text)
    text = _EMAIL_URL_LINE.sub("", text)
    text = _TASK_PROMPT_LINE.sub("", text)
    text = _IO_LABEL_LINE.sub("", text)
    text = _ISOLATED_CITATION_FRAG.sub("", text)
    text = _BARE_AUTHOR_CITE.sub("", text)
    return text


def _repair_text(text: str) -> str:
    # Strip null bytes from broken PDF encoding (common ligature loss)
    text = text.replace('\x00', '')
    # Fix common Unicode ligature decomposition issues
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = text.replace('\ufb00', 'ff').replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')

    # ── Restore dropped PDF ligatures ────────────────────────────────────
    # Many PDFs silently drop fi/fl/ff/ffi/ffl glyphs, turning e.g.
    # "significant" → "signicant", "reflect" → "reect".
    # We fix the most common academic-English words via regex sub.
    text = _restore_ligatures(text)

    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _ARXIV_FOOTER.sub("", text)
    text = _FOOTNOTE_SUPERSCRIPT.sub(r" \1", text)
    text = _WHITESPACE.sub(" ", text)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    return text


# ── Ligature restoration table ────────────────────────────────────────────
# Maps (broken_form_regex, replacement) for words where fi/fl/ff/ffi/ffl
# was silently dropped by the PDF renderer.  Boundary-anchored (\b) to
# avoid false positives.
import re as _re

_LIGATURE_FIXES: list[tuple[_re.Pattern, str]] = [
    # ── fi ──
    # IMPORTANT: Patterns below only fire when the 'fi' ligature is ACTUALLY MISSING.
    # Guard against double-application by requiring the captured char to NOT already be 'f'.
    # e.g. 'signi[^f]' catches broken 'signcant' but NOT already-correct 'significant'.
    (_re.compile(r'\bsigni([^fi])', _re.I), r'signifi\1'),   # signcant→significant, NOT significant→signifificant
    (_re.compile(r'\bspeci([^fi])', _re.I), r'specifi\1'),   # specced→specified, NOT specified→specifified
    (_re.compile(r'\bclassi([^fi])', _re.I), r'classifi\1'), # classcation→classification, NOT classified→classifified
    (_re.compile(r'\bidenti([^fi])', _re.I), r'identifi\1'), # identcation→identification
    # 'veri' is tricky: 'Verilog' starts with 'Veril', 'verified' starts with 'verif'.
    # Only restore when the word is truly broken: 'vercation' (missing 'fi').
    # Safe form: match 'veri' followed by a char that is neither 'f', 'l', nor 'o' (Verilog)
    (_re.compile(r'\bveri([^flo])', _re.I), r'verifi\1'),    # vercation→verification, NOT Verilog, NOT verified
    (_re.compile(r'\bjusti([^fi])', _re.I),  r'justifi\1'),  # justcation→justification, NOT justified
    (_re.compile(r'\bquanti([^fi])', _re.I), r'quantifi\1'), # quantcation→quantification
    (_re.compile(r'\bmodify|modi([^fi])', _re.I), r'modifi\1'),
    (_re.compile(r'\bne-tun', _re.I),     r'fine-tun'),
    (_re.compile(r'\bne\s*-\s*tuned', _re.I), r'fine-tuned'),
    (_re.compile(r'\bne\s*-\s*tuning', _re.I), r'fine-tuning'),
    (_re.compile(r'\bndings\b', _re.I),   r'findings'),
    (_re.compile(r'\brst\b', _re.I),      r'first'),
    (_re.compile(r'\bnite\b', _re.I),     r'finite'),
    (_re.compile(r'\bnished\b', _re.I),   r'finished'),
    (_re.compile(r'\bnely\b', _re.I),     r'finely'),
    (_re.compile(r'\bnal\b', _re.I),      r'final'),
    (_re.compile(r'\bnally\b', _re.I),    r'finally'),
    (_re.compile(r'\beld\b', _re.I),      r'field'),
    (_re.compile(r'\belds\b', _re.I),     r'fields'),
    (_re.compile(r'\bgure\b', _re.I),     r'figure'),
    (_re.compile(r'\bgures\b', _re.I),    r'figures'),
    (_re.compile(r'\ble\b(?!\s*[=<>])', _re.I), r'file'),     # avoid "le" in code
    (_re.compile(r'\bles\b', _re.I),      r'files'),
    (_re.compile(r'\blter\b', _re.I),     r'filter'),
    (_re.compile(r'\blters\b', _re.I),    r'filters'),
    (_re.compile(r'\bltering\b', _re.I),  r'filtering'),
    (_re.compile(r'\blled\b', _re.I),     r'filled'),
    (_re.compile(r'\bxed\b', _re.I),      r'fixed'),
    (_re.compile(r'\bx\b(?=\s+(?:the|a|an|this|that|errors?|bugs?|issues?))', _re.I), r'fix'),
    (_re.compile(r'\bxes\b', _re.I),      r'fixes'),
    (_re.compile(r'\bbenets\b', _re.I),   r'benefits'),
    (_re.compile(r'\bbenet\b', _re.I),    r'benefit'),
    (_re.compile(r'\bnetuned\b', _re.I),  r'finetuned'),
    (_re.compile(r'\bnetuning\b', _re.I), r'finetuning'),
    # Words where "fi" appears mid-word
    (_re.compile(r'\bef([^f])cien', _re.I), r'effi\1cien'),   # ecient→efficient, NOT efficient→effficient
    (_re.compile(r'\binsuf([^f])cien', _re.I), r'insuffi\1cien'),
    (_re.compile(r'\bpro([^f])cien', _re.I), r'profi\1cien'),
    (_re.compile(r'\bcon([^f])gur', _re.I), r'confi\1gur'),   # conguration→configuration
    (_re.compile(r'\bcon([^f])rm', _re.I),  r'confirm'),       # conrm→confirm
    (_re.compile(r'\bcon([^f])den', _re.I), r'confiden'),      # condence→confidence
    (_re.compile(r'\bclari([^fi])', _re.I),  r'clarifi\1'),   # clarcation→clarification
    (_re.compile(r'\bampli([^fi])', _re.I),  r'amplifi\1'),   # amplication→amplification
    (_re.compile(r'\bsimpli([^fi])', _re.I), r'simplifi\1'),  # simplication→simplification
    (_re.compile(r'\brecti([^fi])', _re.I),  r'rectifi\1'),   # rectcation→rectification
    (_re.compile(r'\bexempli([^fi])', _re.I), r'exemplifi\1'), # exemplication→exemplification
    # ── fl ──
    (_re.compile(r'\breect', _re.I),      r'reflect'),
    (_re.compile(r'\bconice?t', _re.I),   r'conflict'),
    (_re.compile(r'\bin([^f])uenc', _re.I), r'influ\1enc'),    # inuence→influence (NOT influence→inflfuence)
    (_re.compile(r'\bow\b', _re.I),       r'flow'),
    (_re.compile(r'\bows\b', _re.I),      r'flows'),
    # ── ff ──
    # When 'ff' is dropped by a PDF renderer the resulting broken forms are shorter words.
    # Match them directly — do NOT use a capturing group placeholder approach.
    # 'effect' (e-ff-e-c-t) → ff dropped → 'eect' — but 'eect' is too similar to 'eject',
    # so skip this one (low frequency / too risky). 'electrical'/'electric' already handled by
    # NOT having these wrong patterns.
    (_re.compile(r'\beort\b', _re.I),     r'effort'),       # eort → effort
    (_re.compile(r'\boer\b', _re.I),      r'offer'),         # oer → offer
    (_re.compile(r'\bdierent\b', _re.I),  r'different'),     # dierent → different
    (_re.compile(r'\bdicult\b', _re.I),   r'difficult'),     # dicult → difficult
    (_re.compile(r'\becient\b', _re.I),   r'efficient'),     # ecient → efficient
    (_re.compile(r'\becacy\b', _re.I),    r'efficacy'),      # ecacy → efficacy
]


def _restore_ligatures(text: str) -> str:
    """Restore common English words broken by dropped PDF ligatures."""
    for pattern, replacement in _LIGATURE_FIXES:
        text = pattern.sub(replacement, text)
    return text


def _detect_headers_footers(pages: List[PageMeta]) -> set:
    from collections import Counter
    line_counts: Counter = Counter()
    for page in pages:
        lines = set(
            line.strip() for line in page.text.splitlines()
            if len(line.strip()) > 5
        )
        line_counts.update(lines)
    return {
        line for line, count in line_counts.items()
        if count >= _HEADER_FOOTER_MIN_PAGES
    }


def _strip_headers_footers(text: str, repeated_lines: set) -> str:
    if not repeated_lines:
        return text
    lines = text.splitlines()
    cleaned = [ln for ln in lines if ln.strip() not in repeated_lines]
    return "\n".join(cleaned)


# ── Public helpers ────────────────────────────────────────────────────────────

def merge_broken_parenthetical_citations(text: str) -> str:
    """
    Rejoin lines where an open parenthesis ( or bracket [ was not closed
    before the newline. Prevents the sentence splitter from cutting inside
    multi-author citation blocks like:

        "The advent of LLMs (Radford et al.,\n2019; Raffel et al., 2020)"

    which would otherwise produce the broken fragment:

        "The advent of LLMs (Radford et al.,"

    Algorithm: track bracket depth character-by-character. When depth > 0
    at end of a line, buffer the next line onto the current one. Emit only
    when depth returns to 0. Unclosed citations at end of page are flushed.
    """
    lines = text.split('\n')
    merged: list[str] = []
    buffer = ''
    depth = 0

    for line in lines:
        for ch in line:
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth = max(0, depth - 1)

        if buffer:
            buffer = buffer.rstrip() + ' ' + line.lstrip()
        else:
            buffer = line

        if depth == 0:
            merged.append(buffer)
            buffer = ''
            # depth is already 0 here — no need to reset explicitly

    # Flush any remaining buffer (unclosed citation at page boundary)
    if buffer:
        merged.append(buffer)

    return '\n'.join(merged)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def clean_pages(pages: List[PageMeta]) -> List[PageMeta]:
    print(f"[DEBUG cleaner] clean_pages: cleaning {len(pages)} pages")
    """Full cleaning pipeline applied to all pages before sentence splitting.

    Pipeline order (each step feeds into the next):
        1. _strip_headers_footers  — remove repeated page headers/footers
        2. _repair_text            — fix hyphen linebreaks, OCR artifacts,
                                     normalize whitespace
        3. _strip_boilerplate      — remove boilerplate/section-label lines
        4. _strip_fused_headings   — split fused ALL-CAPS heading + sentence
        5. merge_broken_parenthetical_citations
                                   — rejoin lines with unclosed ( or [
                                     so the splitter never sees a truncated
                                     citation fragment as a sentence boundary
        6. text.strip()            — final whitespace trim
    """
    repeated_lines = _detect_headers_footers(pages)
    if repeated_lines:
        log.info(f"Detected {len(repeated_lines)} repeated header/footer lines")

    cleaned_pages = []
    for page in pages:
        text = page.text
        # PDF font-encoding corruption of the word "naïve"
        import unicodedata
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r'\bna\s*[^\w\s]{0,3}ve\b', 'naive', text)
        text = re.sub(r'\bna\s{1,3}ve\b', 'naive', text)

        text = _strip_headers_footers(text, repeated_lines)   # step 1
        text = _repair_text(text)                             # step 2
        text = _strip_boilerplate(text)                       # step 3
        text = _strip_fused_headings(text)                    # step 4
        text = merge_broken_parenthetical_citations(text)     # step 5  ← NEW
        text = text.strip()                                   # step 6

        cleaned_pages.append(PageMeta(
            doc_id=page.doc_id,
            page_num=page.page_num,
            char_count=len(text),
            ocr_used=page.ocr_used,
            text=text,
        ))

    log.info(f"clean_pages: cleaned {len(cleaned_pages)} pages")
    return cleaned_pages