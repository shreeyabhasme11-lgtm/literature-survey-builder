from app.sections.keyword_rules import score_sentence

def test_methods_detection():
    scores = score_sentence("We propose a novel deep learning framework for text classification.")
    assert scores["methods"] > 0

def test_limitations_detection():
    scores = score_sentence("However, the model fails to generalize to out-of-domain data.")
    assert scores["limitations"] > 0