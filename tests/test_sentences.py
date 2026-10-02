from app.metadata.schema import PageMeta
from app.extraction.sentence_splitter import split_sentences

def test_split_basic():
    pages = [PageMeta("d1", 1, 100, False,
        "Machine learning is a field of AI. It enables computers to learn from data.")]
    sents = split_sentences(pages)
    assert len(sents) >= 1
    assert all(s.word_count >= 1 for s in sents)