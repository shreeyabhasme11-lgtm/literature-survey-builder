"""
Source diversity enforcement — ensures no single paper dominates a section.
At most SOURCE_DIVERSITY_CAP fraction of sentences in any section
can come from one source document.
"""
from collections import defaultdict
from typing import List

from app.metadata.schema import SentenceMeta
from app.config import SOURCE_DIVERSITY_CAP
from app.sections.section_builder import group_by_section


def enforce_diversity(sentences: List[SentenceMeta]) -> List[SentenceMeta]:
    """
    Within each section, caps the fraction of sentences from any single
    source at SOURCE_DIVERSITY_CAP. Lower-scored sentences from
    over-represented sources are dropped.
    Returns a new flattened list of kept sentences across all sections.
    """
    groups = group_by_section(sentences)
    result: List[SentenceMeta] = []

    for section, sents in groups.items():
        # Sort descending by TextRank score so best sentences are kept first
        sents_sorted = sorted(sents, key=lambda s: -s.textrank_score)
        counts: defaultdict = defaultdict(int)
        kept: List[SentenceMeta] = []

        for s in sents_sorted:
            counts[s.doc_id] += 1
            total = len(kept) + 1
            fraction = counts[s.doc_id] / total

            if fraction <= SOURCE_DIVERSITY_CAP or total <= 2:
                kept.append(s)
            else:
                counts[s.doc_id] -= 1  # undo the increment for dropped sentence

        result.extend(kept)

    return result