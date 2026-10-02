from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from typing import List, Tuple
import numpy as np


def build_tfidf(texts: List[str]) -> Tuple[spmatrix, TfidfVectorizer]:
    vec = TfidfVectorizer(
        max_features=10_000,
        min_df=1,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    matrix = vec.fit_transform(texts)
    return matrix, vec