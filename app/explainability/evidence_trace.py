from typing import List, Dict
from app.metadata.schema import SentenceMeta


def build_traces(sentences: List[SentenceMeta]) -> List[Dict]:
    return [
        {
            "sent_id": s.sent_id,
            "text": s.text,
            "filename": s.filename,
            "page_num": s.page_num,
            "doc_id": s.doc_id,
            "textrank_score": round(s.textrank_score, 6),
            "lexrank_score": round(s.lexrank_score, 6),
            "topic_id": s.topic_id,
            "topic_keywords": s.topic_keywords,
            "section": s.section,
            "section_score": round(s.section_score, 4),
            "coherence_score": round(s.coherence_score, 4),
            "subtopic_cluster": s.subtopic_cluster,
            "ordering_reason": s.ordering_reason,
            "selected": s.selected,
            "redundant": s.redundant,
            "word_count": s.word_count,
        }
        for s in sentences
    ]