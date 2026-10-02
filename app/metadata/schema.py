"""Dataclasses for all pipeline entities."""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DocumentMeta:
    doc_id: str
    filename: str
    filepath: str
    size_bytes: int
    upload_ts: str
    num_pages: int = 0


@dataclass
class PageMeta:
    doc_id: str
    page_num: int
    char_count: int
    ocr_used: bool
    text: str


@dataclass
class SentenceMeta:
    sent_id: str
    doc_id: str
    filename: str
    page_num: int
    text: str
    word_count: int

    # Ranking
    textrank_score: float = 0.0
    lexrank_score: float = 0.0

    # Topics
    topic_id: int = -1
    topic_keywords: str = ""

    # Section classification
    section: str = "introduction"
    section_score: float = 0.0

    # Coherence
    subtopic_cluster: int = -1
    coherence_score: float = 0.0
    ordering_reason: str = ""

    # Flags
    redundant: bool = False
    selected: bool = False

    # New synthesis helpers
    synthesis_role: str = ""
    evidence_group: str = ""
    synthesis_note: str = ""
    supporting_sent_ids: List[str] = field(default_factory=list)