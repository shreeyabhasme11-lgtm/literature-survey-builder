"""
Deterministic literature survey composer.

Instead of emitting a flat sequence of extracted sentences, this composer:
1. groups evidence by section and topic/subtopic,
2. prefers bundles supported by multiple papers,
3. generates short deterministic synthesis paragraphs (including compare/contrast
   when multiple distinct exemplars exist),
4. appends compact traceable citations.

No LLM. No generative AI. Every paragraph remains source-traceable.
"""
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from app.metadata.schema import SentenceMeta
from app.config import SECTION_LABELS, TOP_K_PER_SECTION
from app.sections.section_builder import group_by_section
from app.coherence.sentence_orderer import order_sentences
from app.composition.citation_formatter import format_citation
from app.utils.logger import get_logger

log = get_logger(__name__)


_POST_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d{4}[a-z]?[\)\]]+\s+"),
    re.compile(r"^[A-Z]{3,}(?:\s+[A-Z]{2,}){2,}\s+[A-Z][a-z]"),
    re.compile(
        r"^(Given the following|Your task is|You are given|Please provide|"
        r"Write a\b|Generate a\b|Based on the above|In this task|"
        r"LM input:|LM output:|Expected output:|Instruction:|Task:|Prompt:|"
        r"Now I would like you to|Help me design|Please help me design)",
        re.IGNORECASE,
    ),
    re.compile(r"^[A-Za-z ]{1,20}:\s+\S.{0,30}\s+[\d\.]+\s+[\d\.]+"),
    re.compile(r"^(\[\d+\]\s*){3,}$"),
    re.compile(r"^[A-Z][,\.;]\s"),
    re.compile(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*$"),
    re.compile(r"^[a-z]"),
    re.compile(r"^\s*[A-Z]\.\d+"),
    re.compile(r"\bappendix\b", re.IGNORECASE),
    re.compile(r"\bprompt template\b", re.IGNORECASE),
    re.compile(r"\bcircuit tool library overview\b", re.IGNORECASE),
    re.compile(r"\bsubsequently, the keys and values\b", re.IGNORECASE),
    re.compile(r"\bnow i would like you to help me\b", re.IGNORECASE),
    re.compile(r"\bhelp me design a complex analog circuit\b", re.IGNORECASE),
    re.compile(r"\bterminal ports?\b", re.IGNORECASE),
    re.compile(r"\bvoltage input port\b", re.IGNORECASE),
    re.compile(r"\bvoltage output port\b", re.IGNORECASE),
    re.compile(r"\bground gnd\b", re.IGNORECASE),
    re.compile(r"\btransistor setup\b", re.IGNORECASE),
    re.compile(r"\bgate of [A-Za-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bdrain of [A-Za-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bsource of [A-Za-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bconnected to the drain\b", re.IGNORECASE),
    re.compile(r"\bconnected to the gate\b", re.IGNORECASE),
    re.compile(r"\bconnected to the source\b", re.IGNORECASE),
    re.compile(r"\bvin\b.*\bvout\b.*\bgnd\b", re.IGNORECASE),
    re.compile(r"\bM\d+\s*\((nmos|pmos)\)", re.IGNORECASE),
    re.compile(r"\bfour lines in the spice code\b", re.IGNORECASE),
    re.compile(r"\bsubstitute the original input voltage\b", re.IGNORECASE),
    re.compile(r"\bauthor(s)?\b", re.IGNORECASE),
]


def _looks_like_author_list(text: str) -> bool:
    """
    Detect lines that are mostly author-name sequences, e.g.
    'Radford, A., Wu, J., Child, R., ...'
    """
    comma_count = text.count(",")
    if comma_count < 4:
        return False

    tokens = [t.strip() for t in text.split(",")]
    short_name_like = 0
    for tok in tokens[:10]:
        if re.match(r"^[A-Z][a-z\-']+\s*[A-Z]\.?$", tok):
            short_name_like += 1
        elif re.match(r"^[A-Z]\.?$", tok):
            short_name_like += 1
        elif re.match(r"^[A-Z][a-z\-']+$", tok):
            short_name_like += 1

    return short_name_like >= 4


def _is_too_technical_fragment(text: str) -> bool:
    """
    Filter very low-level implementation fragments that are poor literature-review prose.
    """
    lowered = text.lower()

    technical_markers = [
        "nmos", "pmos", "vin", "vout", "gnd", "spice code",
        "gate of", "drain of", "source of", "terminal ports",
        "input voltage", "output closest to", "transistor setup",
    ]
    marker_hits = sum(1 for m in technical_markers if m in lowered)

    if marker_hits >= 2:
        return True

    if re.search(r"\bm\d+\b", lowered) and marker_hits >= 1:
        return True

    return False


def _is_post_noisy(text: str) -> bool:
    """
    Return True if a sentence still looks noisy even after earlier filters.
    """
    stripped = text.strip()

    for pat in _POST_NOISE_PATTERNS:
        if pat.search(stripped):
            return True

    if _looks_like_author_list(stripped):
        return True

    if _is_too_technical_fragment(stripped):
        return True

    words = stripped.split()
    if words:
        bracket_count = sum(1 for w in words if re.match(r"^\[\d+\]$", w))
        if bracket_count / len(words) > 0.35:
            return True

    clean_words = re.sub(r"\[[^\]]+\]", "", stripped).split()
    if len(clean_words) < 5:
        return True

    # Reject lines with too many symbol-heavy tokens
    symbol_heavy = sum(1 for w in clean_words if re.search(r"[\(\)_/=×\+\-\*]", w))
    if clean_words and symbol_heavy / len(clean_words) > 0.25:
        return True

    return False


def _normalize_phrase(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trim_sentence(text: str, max_words: int = 24) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.rstrip(".")
    return " ".join(words[:max_words]).rstrip(".,;:") + "…"


def _section_lead(section: str, paper_count: int, bundle_count: int) -> str:
    if section == "introduction":
        return (
            f"Across {paper_count} paper(s), the literature frames this area around "
            f"{bundle_count} recurring theme(s)."
        )
    if section == "methods":
        return (
            f"Across the reviewed studies, the methodological discussion clusters into "
            f"{bundle_count} recurring approach(es)."
        )
    if section == "findings":
        return (
            f"The findings section highlights {bundle_count} main pattern(s) reported "
            f"across {paper_count} paper(s)."
        )
    if section == "limitations":
        return (
            f"Across the corpus, the reported limitations converge around "
            f"{bundle_count} recurring concern(s)."
        )
    if section == "research_gaps":
        return (
            f"The reviewed papers collectively point to {bundle_count} main research gap(s)."
        )
    if section == "conclusion":
        return (
            f"Taken together, the literature supports {bundle_count} main takeaway(s) "
            f"across {paper_count} paper(s)."
        )
    return f"This section synthesizes {bundle_count} recurring pattern(s) across the reviewed papers."


def _bundle_key(s: SentenceMeta) -> Tuple[int, int]:
    return (s.topic_id, s.subtopic_cluster)


def _build_bundles(sentences: List[SentenceMeta]) -> List[List[SentenceMeta]]:
    grouped: dict[Tuple[int, int], List[SentenceMeta]] = defaultdict(list)
    for s in sentences:
        if s.redundant or _is_post_noisy(s.text):
            continue
        grouped[_bundle_key(s)].append(s)

    bundles: List[List[SentenceMeta]] = []
    for _, items in grouped.items():
        items = sorted(items, key=lambda x: (-x.textrank_score, x.doc_id, x.page_num))
        bundles.append(items)

    def bundle_rank(bundle: List[SentenceMeta]) -> Tuple[int, float]:
        doc_diversity = len({s.doc_id for s in bundle})
        score = sum(s.textrank_score for s in bundle[:3])
        return (doc_diversity, score)

    bundles.sort(key=bundle_rank, reverse=True)
    return bundles


def _choose_opening(section: str, doc_count: int, keywords: str | None) -> str:
    kw_part = f" around {keywords}" if keywords else ""

    if section == "methods":
        if doc_count >= 3:
            return f"Most studies adopt related methodological choices{kw_part}"
        if doc_count == 2:
            return f"Two studies describe similar methodological decisions{kw_part}"
        return f"One methodological contribution focuses{kw_part}"

    if section == "findings":
        if doc_count >= 3:
            return f"Across several papers, a consistent finding emerges{kw_part}"
        if doc_count == 2:
            return f"Two papers report a similar result{kw_part}"
        return f"One study reports that{kw_part}"

    if section == "limitations":
        if doc_count >= 3:
            return f"A recurring limitation reported across multiple papers concerns{kw_part}"
        if doc_count == 2:
            return f"Two papers flag a similar limitation{kw_part}"
        return f"At least one study notes a limitation related to{kw_part}"

    if section == "research_gaps":
        if doc_count >= 3:
            return f"Taken together, the literature reveals an open gap{kw_part}"
        if doc_count == 2:
            return f"Two studies highlight an open question{kw_part}"
        return f"One paper points to an open issue{kw_part}"

    if section == "introduction":
        if doc_count >= 3:
            return f"Across several foundational papers, the problem is framed{kw_part}"
        if doc_count == 2:
            return f"Two core papers frame the area{kw_part}"
        return f"A single paper frames the area{kw_part}"

    if section == "conclusion":
        if doc_count >= 3:
            return f"Collectively, the reviewed work suggests{kw_part}"
        if doc_count == 2:
            return f"Across two key studies, the main takeaway is{kw_part}"
        return f"Based on one study, a preliminary takeaway is{kw_part}"

    if doc_count >= 3:
        return f"Across multiple papers, there is consistent emphasis{kw_part}"
    if doc_count == 2:
        return f"Across two papers, a shared pattern emerges{kw_part}"
    return f"One paper highlights{kw_part}"


def _choose_exemplars(bundle: List[SentenceMeta], max_items: int = 3) -> List[str]:
    """
    Pick cleaner, more abstract exemplars for synthesis.
    Prefer longer conceptual sentences and reject noisy or low-level fragments.
    """
    candidates = sorted(
        bundle,
        key=lambda s: (
            _is_post_noisy(s.text),
            -s.word_count,
            -s.textrank_score,
            s.doc_id,
            s.page_num,
        ),
    )

    exemplar_texts: List[str] = []
    seen = set()

    for s in candidates:
        norm = _normalize_phrase(s.text)
        if not norm:
            continue
        if norm.lower() in seen:
            continue
        if _is_post_noisy(norm):
            continue
        if len(norm.split()) < 8:
            continue

        seen.add(norm.lower())
        exemplar_texts.append(_trim_sentence(norm, max_words=22))
        if len(exemplar_texts) >= max_items:
            break

    return exemplar_texts


def _bundle_summary(section: str, bundle: List[SentenceMeta]) -> str:
    docs = list(dict.fromkeys(s.filename for s in bundle))
    doc_count = len({s.doc_id for s in bundle})
    lead = bundle[0]
    keywords = lead.topic_keywords.strip()

    exemplar_texts = _choose_exemplars(bundle, max_items=3)

    evidence_hint = ""
    if keywords:
        evidence_hint = f" around {keywords}"
    elif exemplar_texts:
        evidence_hint = f" around {exemplar_texts[0].lower()}"

    opener = _choose_opening(section, doc_count, keywords if keywords else None)

    if section == "findings":
        verb = "the evidence suggests that"
    elif section == "methods":
        verb = "the methods indicate that"
    elif section == "limitations":
        verb = "the literature repeatedly reports that"
    elif section == "research_gaps":
        verb = "the literature points to the need for"
    else:
        verb = "the literature indicates that"

    if len(exemplar_texts) >= 3:
        a, b, c = exemplar_texts[:3]
        detail = (
            f": while some studies emphasize {a.lower()}, "
            f"others focus on {b.lower()}, and additional work highlights {c.lower()}."
        )
    elif len(exemplar_texts) == 2:
        a, b = exemplar_texts[:2]
        detail = (
            f": while some studies emphasize {a.lower()}, "
            f"others highlight {b.lower()}."
        )
    elif len(exemplar_texts) == 1:
        detail = f", and {verb} {exemplar_texts[0].lower()}."
    else:
        detail = "."

    if doc_count >= 2:
        source_part = f" This pattern is supported by {doc_count} distinct source paper(s)"
    else:
        source_part = f" This point is currently supported by 1 source paper"

    if docs:
        shown = ", ".join(docs[:3])
        if len(docs) > 3:
            shown += ", among others"
        source_part += f" ({shown})."

    if evidence_hint and evidence_hint not in opener:
        return opener + evidence_hint + detail + source_part
    return opener + detail + source_part


def _compact_support_line(bundle: List[SentenceMeta], max_items: int = 3) -> str:
    chosen = bundle[:max_items]
    cites = [format_citation(s) for s in chosen]
    cite_text = " ".join(cites).strip()
    return f"**Supporting evidence:** {cite_text}" if cite_text else ""


def compose_survey(
    sentences: List[SentenceMeta],
    top_k: int = TOP_K_PER_SECTION,
    ordering: str = "similarity",
) -> Dict[str, List[str]]:
    log.info(
        "compose_survey(): composing survey with %s sentences, top_k=%s, ordering=%s",
        len(sentences),
        top_k,
        ordering,
    )

    groups = group_by_section(sentences)
    survey: Dict[str, List[str]] = {}

    for section in SECTION_LABELS:
        raw_section_sents = [s for s in groups.get(section, []) if not s.redundant]
        if not raw_section_sents:
            continue

        ordered = order_sentences(raw_section_sents, None, strategy=ordering)
        bundles = _build_bundles(ordered)
        if not bundles:
            continue

        selected_bundles = bundles[:top_k]
        paper_count = len({s.doc_id for s in raw_section_sents})

        paragraphs: List[str] = []
        paragraphs.append(_section_lead(section, paper_count, len(selected_bundles)))

        for bundle in selected_bundles:
            paragraph = _bundle_summary(section, bundle)
            support = _compact_support_line(bundle)

            for s in bundle[:3]:
                s.selected = True
                s.synthesis_role = "supporting_evidence"
                s.evidence_group = f"{section}|topic={s.topic_id}|cluster={s.subtopic_cluster}"

            paragraphs.append(paragraph)
            if support:
                paragraphs.append(support)

        survey[section] = paragraphs

    log.info(
        "compose_survey(): %s sections, %s total paragraphs",
        len(survey),
        sum(len(v) for v in survey.values()),
    )
    return survey