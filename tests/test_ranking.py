from app.representation.tfidf_vectorizer import build_tfidf
from app.ranking.textrank_ranker import textrank_score
from app.ranking.lexrank_ranker import lexrank_score

TEXTS = [
    "Deep learning models achieve high accuracy on image classification tasks.",
    "Convolutional neural networks are widely used for computer vision.",
    "Recurrent networks process sequential data and time series.",
    "Transfer learning reduces the need for large labeled datasets.",
]

def test_textrank():
    mat, _ = build_tfidf(TEXTS)
    scores = textrank_score(mat)
    assert len(scores) == len(TEXTS)
    assert all(isinstance(s, float) for s in scores)

def test_lexrank():
    mat, _ = build_tfidf(TEXTS)
    scores = lexrank_score(mat)
    assert len(scores) == len(TEXTS)
    assert abs(sum(scores) - 1.0) < 1e-4  # should sum to ~1