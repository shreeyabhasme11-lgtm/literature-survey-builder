from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix
from typing import List
from app.metadata.schema import SentenceMeta
from app.config import REDUNDANCY_THRESHOLD


def filter_redundant(sentences: List[SentenceMeta], matrix: spmatrix, threshold: float = None) -> None:
    """Mark near-duplicate sentences as redundant (in-place)."""
    if threshold is None:
        threshold = REDUNDANCY_THRESHOLD
    sim = cosine_similarity(matrix)
    n = len(sentences)
    selected = set()
    for i in range(n):
        if sentences[i].redundant:
            continue
        for j in selected:
            if sim[i, j] > threshold:
                sentences[i].redundant = True
                break
        if not sentences[i].redundant:
            selected.add(i)