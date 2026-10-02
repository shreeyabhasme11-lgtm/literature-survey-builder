"""
Sentence ordering strategies for coherence optimization.
Greedy nearest-neighbour (similarity), topic-sorted, diversity-interleaved,
and chronological orderings are supported.
"""
from typing import List
from collections import defaultdict
import numpy as np

from app.metadata.schema import SentenceMeta


def order_sentences(
    sentences: List[SentenceMeta],
    matrix,
    strategy: str = "similarity",
) -> List[SentenceMeta]:
    """
    Reorder sentences for maximum local coherence within a section.
    strategy: "similarity" | "topic" | "diversity" | "chronological"
    """
    if not sentences:
        return sentences

    if strategy == "similarity" and matrix is not None:
        return _similarity_order(sentences, matrix)
    elif strategy == "topic":
        return sorted(
            sentences,
            key=lambda s: (
                s.topic_id,
                s.subtopic_cluster,
                -s.textrank_score,
                s.doc_id,
            ),
        )
    elif strategy == "diversity":
        return _diversity_order(sentences)
    elif strategy == "chronological":
        return sorted(sentences, key=lambda s: (s.doc_id, s.page_num))
    else:
        return sorted(
            sentences,
            key=lambda s: (
                s.topic_id,
                s.subtopic_cluster,
                -s.textrank_score,
            ),
        )


def _similarity_order(
    sentences: List[SentenceMeta],
    matrix,
) -> List[SentenceMeta]:
    """
    Greedy nearest-neighbour ordering.
    Start from the highest-scored sentence, then always pick the most
    similar unvisited sentence. Maximises local cosine similarity
    between adjacent sentences.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if matrix is None or len(sentences) <= 1:
        return sentences

    try:
        arr = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        n = len(sentences)
        visited = [False] * n
        start = max(range(n), key=lambda i: sentences[i].textrank_score)
        order = [start]
        visited[start] = True

        for _ in range(n - 1):
            last = order[-1]
            sims = cosine_similarity(arr[last].reshape(1, -1), arr).flatten()

            for v in order:
                sims[v] = -1.0

            # Prefer cross-document adjacency when similarity is close
            ranked_idxs = np.argsort(-sims)
            nxt = None
            last_doc = sentences[last].doc_id
            for cand in ranked_idxs:
                if sims[cand] < 0:
                    continue
                if sentences[cand].doc_id != last_doc:
                    nxt = int(cand)
                    break

            if nxt is None:
                nxt = int(np.argmax(sims))

            order.append(nxt)
            visited[nxt] = True

        ordered = [sentences[i] for i in order]
        for i, s in enumerate(ordered):
            if i > 0:
                s.ordering_reason = "similarity-based ordering with source diversification"
            else:
                s.ordering_reason = "highest-ranked sentence starts synthesis chain"
        return ordered

    except Exception:
        return sentences


def _diversity_order(sentences: List[SentenceMeta]) -> List[SentenceMeta]:
    """
    Interleave sentences from different source documents.
    Ensures no two consecutive sentences come from the same paper where possible.
    """
    buckets = defaultdict(list)
    for s in sorted(
        sentences,
        key=lambda x: (
            x.topic_id,
            x.subtopic_cluster,
            -x.textrank_score,
        ),
    ):
        buckets[s.doc_id].append(s)

    result: List[SentenceMeta] = []
    while any(buckets.values()):
        for doc_id in list(buckets.keys()):
            if buckets[doc_id]:
                s = buckets[doc_id].pop(0)
                s.ordering_reason = "source-diversity interleaving"
                result.append(s)

    return result