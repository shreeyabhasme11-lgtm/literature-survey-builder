"""Group sentences by section label."""
from typing import Dict, List
from app.metadata.schema import SentenceMeta
from app.config import SECTION_LABELS


def group_by_section(sentences: List[SentenceMeta]) -> Dict[str, List[SentenceMeta]]:
    """
    Returns a dict mapping each section label to the sentences assigned to it.
    Sentences with unrecognised labels fall into 'introduction'.
    Within each section, sentences are sorted to encourage synthesis:
    topic -> subtopic -> score desc -> source diversification.
    """
    groups: Dict[str, List[SentenceMeta]] = {s: [] for s in SECTION_LABELS}

    for sent in sentences:
        label = sent.section if sent.section in groups else "introduction"
        groups[label].append(sent)

    for section, items in groups.items():
        groups[section] = sorted(
            items,
            key=lambda s: (
                s.topic_id if s.topic_id is not None else 999,
                s.subtopic_cluster if s.subtopic_cluster is not None else 999,
                -s.textrank_score,
                s.doc_id,
                s.page_num,
            ),
        )

    return groups