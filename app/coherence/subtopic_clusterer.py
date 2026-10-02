"""
Assign subtopic cluster IDs to sentences within each section using K-Means
on TF-IDF vectors. This groups related sentences so transition templates
fire at meaningful thematic boundaries rather than arbitrarily.
"""
from typing import List, Optional

import numpy as np

from app.metadata.schema import SentenceMeta
from app.config import N_SUBTOPIC_CLUSTERS


def cluster_subtopics(
    sentences: List[SentenceMeta],
    matrix=None,
    n_clusters: int = N_SUBTOPIC_CLUSTERS,
) -> None:
    """Assign subtopic_cluster in-place. Falls back to topic_id if clustering fails."""
    if matrix is None or len(sentences) < n_clusters:
        for s in sentences:
            s.subtopic_cluster = s.topic_id
        return

    try:
        from sklearn.cluster import KMeans

        arr = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        k = min(n_clusters, len(sentences))
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(arr)

        for s, lbl in zip(sentences, labels):
            s.subtopic_cluster = int(lbl)

    except Exception:
        # Fallback: use topic_id as cluster
        for s in sentences:
            s.subtopic_cluster = s.topic_id