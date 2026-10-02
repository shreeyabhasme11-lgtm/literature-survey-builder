# app/topics/lda_modeler.py  — sklearn-only version (no gensim)
from typing import List, Tuple
from app.config import NUM_LDA_TOPICS
from app.utils.logger import get_logger

log = get_logger(__name__)


def fit_lda(texts: List[str]) -> Tuple[List[int], List[List[Tuple[str, float]]]]:
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(max_features=5000, stop_words="english", min_df=1)
        dtm = cv.fit_transform(texts)
        vocab = cv.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=NUM_LDA_TOPICS,
            random_state=42,
            max_iter=20,
        )
        doc_topic = lda.fit_transform(dtm)
        assignments = doc_topic.argmax(axis=1).tolist()

        topic_kws = []
        for topic_vec in lda.components_:
            top_idx = topic_vec.argsort()[-8:][::-1]
            topic_kws.append([(vocab[i], float(topic_vec[i])) for i in top_idx])

        log.info(f"sklearn LDA fitted: {NUM_LDA_TOPICS} topics")
        return assignments, topic_kws

    except Exception as e:
        log.warning(f"LDA failed ({e}); assigning topic 0")
        return [0] * len(texts), [[("general", 1.0)]]