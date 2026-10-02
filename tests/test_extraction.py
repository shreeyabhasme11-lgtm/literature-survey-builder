import pytest
from pathlib import Path
from app.metadata.schema import DocumentMeta
from app.extraction.pdf_extractor import extract_document


def test_extract_nonexistent():
    doc = DocumentMeta("test", "fake.pdf", "/tmp/fake.pdf", 0, "2025-01-01T00:00:00Z")
    pages = extract_document(doc)
    assert pages == []