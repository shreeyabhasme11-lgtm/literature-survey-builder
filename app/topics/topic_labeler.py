from typing import List, Tuple, Dict


def label_topics(topic_kws: List[List[Tuple[str, float]]]) -> Dict[int, str]:
    """Convert gensim keyword lists to readable label strings."""
    labels = {}
    for i, kws in enumerate(topic_kws):
        top = [w for w, _ in kws[:5]]
        labels[i] = ", ".join(top)
    return labels