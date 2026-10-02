"""Classify sentences into literature survey sections using keyword heuristics."""
from typing import List
from app.metadata.schema import SentenceMeta
from app.sections.keyword_rules import score_sentence, get_primary_section


def classify_sentences(sentences: List[SentenceMeta]) -> None:
    """
    Assign section label and section_score to each sentence in-place.
    Uses keyword rules with positive scoring and exclusion suppression.
    """
    for s in sentences:
        scores = score_sentence(s.text)
        s.section = get_primary_section(s.text)
        s.section_score = scores.get(s.section, 0.0)