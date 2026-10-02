"""In-memory + disk metadata manager."""
from pathlib import Path
from typing import Dict, List, Optional

from app.metadata.schema import DocumentMeta, PageMeta, SentenceMeta
from app.ingestion.manifest import save_manifest, load_manifest
from app.utils.logger import get_logger

log = get_logger(__name__)


class MetadataManager:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._documents: Dict[str, DocumentMeta] = {}
        self._pages: Dict[str, List[PageMeta]] = {}
        self._sentences: List[SentenceMeta] = []

    # ── Documents ─────────────────────────────────────────────────────────

    def add_document(self, doc: DocumentMeta) -> None:
        self._documents[doc.doc_id] = doc
        self._persist()

    def get_document(self, doc_id: str) -> Optional[DocumentMeta]:
        return self._documents.get(doc_id)

    def all_documents(self) -> List[DocumentMeta]:
        return list(self._documents.values())

    # ── Pages ─────────────────────────────────────────────────────────────

    def set_pages(self, doc_id: str, pages: List[PageMeta]) -> None:
        self._pages[doc_id] = pages
        doc = self._documents.get(doc_id)
        if doc:
            doc.num_pages = len(pages)
            self._persist()

    def get_pages(self, doc_id: str) -> List[PageMeta]:
        return self._pages.get(doc_id, [])

    # ── Sentences ─────────────────────────────────────────────────────────

    def set_sentences(self, sentences: List[SentenceMeta]) -> None:
        self._sentences = sentences

    def all_sentences(self) -> List[SentenceMeta]:
        return self._sentences

    # ── Persistence ───────────────────────────────────────────────────────

    def _persist(self) -> None:
        try:
            save_manifest(list(self._documents.values()), self._data_dir)
        except Exception as e:
            log.warning(f"Manifest save failed: {e}")

    def load_from_disk(self) -> None:
        docs = load_manifest(self._data_dir)
        for d in docs:
            self._documents[d.doc_id] = d
        log.info(f"Loaded {len(docs)} documents from manifest")