"""
Keyword and regex rules for literature survey section classification.

Each section has:
  - POSITIVE patterns  → increase section score
  - NEGATIVE/EXCLUSION patterns → sentences matching these are suppressed
    regardless of positive score (set score to 0 for that section).
"""
import re
from typing import Dict


# ── Positive keyword rules per section ──────────────────────────────────────
_SECTION_RULES: Dict[str, list] = {
    "introduction": [
        r"\b(introduc|background|motivation|overview|context|problem statement|aim|objective|scope)\b",
        r"\b(related work|literature review|prior work|survey|previous stud|state of the art)\b",
        r"\b(this paper|this work|this report|this project|this thesis)\s+(present|address|focus|describe|propose|aim)\b",
        r"\b(is defined as|is known as|refers to|is a type of|is an approach)\b",
    ],
    "methods": [
        r"\b(method|approach|algorithm|technique|framework|architecture|model|design|implement|procedure)\b",
        r"\b(dataset|experiment setup|proposed|system design|pipeline|workflow|module)\b",
        r"\b(we use|we propose|we design|we develop|our model|our method|our system|our approach)\b",
        r"\b(is computed|is calculated|is trained|is optimized|is applied|is used to)\b",
        r"\b(consists of|comprises|is composed of|is built on|is based on)\b",
        r"\b(step \d|phase \d|stage \d|layer \d|equation|formula|parameter)\b",
    ],
    "findings": [
        r"\b(result|finding|performance|accuracy|precision|recall|f1|improvement|outperform|achieve)\b",
        r"\b(show|demonstrate|indicate|reveal|confirm|validate|prove|establish|observe)\b",
        r"\b(our approach|our model|our method|we obtain|we find|experiment shows|baseline comparison)\b",
        r"\b(table \d|figure \d|plot|graph|chart|metric|benchmark|evaluation)\b",
        r"\b(higher|lower|faster|slower|better|worse|significant|marginal|notable)\s+(than|compared)\b",
        r"\b(reduces?|improves?|increases?|decreases?|achieves?|reaches?)\s+\w+\s+by\b",
        r"\b(dramatically|significantly|substantially)\s+(reduc|improv|speed|enhanc)\b",
        r"\b(reducing computational|reducing storage|reducing complexity|reducing overhead)\b",
    ],
    "limitations": [
        r"\b(limitation|drawback|weakness|shortcoming|constraint|challenge|disadvantage)\b",
        r"\b(difficult|fail|cannot|unable|not able|struggle|prone to|susceptible)\b",
        r"\b(however|despite|although|nevertheless|on the other hand|in contrast|yet)\b",
        r"\b(error|noise|overhead|cost|complexity|bottleneck|trade.?off)\b",
        r"\b(only|merely|solely|restricted to|limited to|does not|did not|cannot handle)\b",
    ],
    "research_gaps": [
        r"\b(future work|future research|open problem|open question|research gap|unexplored)\b",
        r"\b(direction|remain|to be investigated|further stud|promising avenue|warrants)\b",
        r"\b(not addressed|not covered|beyond the scope|left for future|not explored|overlooked)\b",
        r"\b(could be extended|can be improved|remains an open|yet to be|still lacks)\b",
        r"\b(recommend|suggest|plan to|intend to)\b",
        r"\bpropose\b.+?\bfuture\b",
    ],
    "conclusion": [
        r"\b(conclude|conclusion|in conclusion|summary|in summary|to summarize|overall)\b",
        r"\b(in this (paper|work|report|project|thesis)|this (paper|work|report|thesis) (has|have|presents|showed))\b",
        r"\b(we have (shown|demonstrated|presented|developed|proposed|achieved|evaluated))\b",
        r"\b(key contribution|main contribution|primary contribution|significant contribution)\b",
        r"\b(thus|therefore|hence|accordingly|as a result|in light of)\b",
    ],
}


# ── Exclusion rules — sentences matching these are REMOVED from a section ───
# Each entry: (section_to_suppress, compiled_pattern)
# section_to_suppress=None means suppress from ALL sections.
_EXCLUSION_RULES = [
    (
        None,
        re.compile(
            r"\b(thank|grateful|gratitude|acknowledge)\b.{0,80}"
            r"(professor|dr\.|department|faculty|supervisor|advisor)",
            re.IGNORECASE,
        ),
    ),
    (
        None,
        re.compile(r"^Q\d+[\)\.]\s+\w", re.IGNORECASE),
    ),
    (
        "research_gaps",
        re.compile(r"^(Figure|Fig\.|Table|Algorithm)\s+[\w\d\.]", re.IGNORECASE),
    ),
    (
        "conclusion",
        re.compile(r"^(Figure|Fig\.|Table|Algorithm)\s+[\w\d\.]", re.IGNORECASE),
    ),
    (
        "research_gaps",
        re.compile(r"\b(eigenvalue|eigenvector|eigenface|eigendecomposition)\b", re.IGNORECASE),
    ),
    (
        "research_gaps",
        re.compile(r"^Solve the\b", re.IGNORECASE),
    ),
    (
        "research_gaps",
        re.compile(r"^\s*(NFR|FR)\d+\s*:", re.IGNORECASE),
    ),
]


# ── Compile positive patterns ────────────────────────────────────────────────
_COMPILED_POSITIVE = {
    sec: [re.compile(p, re.IGNORECASE) for p in pats]
    for sec, pats in _SECTION_RULES.items()
}


def score_sentence(text: str) -> Dict[str, float]:
    """
    Returns a dict of {section: normalised_score} for the given sentence.
    Sections suppressed by exclusion rules receive score 0.0.
    """
    raw: Dict[str, float] = {sec: 0.0 for sec in _SECTION_RULES}

    # 1. Score positive matches
    for sec, patterns in _COMPILED_POSITIVE.items():
        for pat in patterns:
            if pat.search(text):
                raw[sec] += 1.0

    # 2. Apply exclusion rules
    for target_section, excl_pat in _EXCLUSION_RULES:
        if excl_pat.search(text):
            if target_section is None:
                for sec in raw:
                    raw[sec] = 0.0
                break
            else:
                raw[target_section] = 0.0

    # 3. Normalise
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def get_primary_section(text: str) -> str:
    """
    Return the highest-scoring section label for a sentence.
    Falls back to 'introduction' if all scores are zero (fully suppressed).
    """
    scores = score_sentence(text)
    best = max(scores, key=scores.get)
    if scores[best] == 0.0:
        return "introduction"
    return best