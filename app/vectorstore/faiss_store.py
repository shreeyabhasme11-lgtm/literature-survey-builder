from pathlib import Path
from typing import List
import numpy as np


class FAISSStore:
    def __init__(self):
        self._index = None
        self._texts: List[str] = []

    def build(self, vectors: np.ndarray, texts: List[str]) -> None:
        import faiss
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._texts = texts

    def search(self, query_vec: np.ndarray, k: int = 5):
        import faiss
        faiss.normalize_L2(query_vec)
        D, I = self._index.search(query_vec, k)
        return [(self._texts[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self._texts)]

    def save(self, path: Path) -> None:
        import faiss
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(path))