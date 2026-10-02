from app.metadata.schema import SentenceMeta
from app.coherence.transition_rules import get_transition

def _make(i): 
    return SentenceMeta(f"s{i}", "d1", "doc.pdf", 1, f"Sentence number {i}.", 4)

def test_transition_returns_string():
    t = get_transition("methods", 0)
    assert isinstance(t, str) and len(t) > 5

def test_transition_cycles():
    t0 = get_transition("findings", 0)
    t3 = get_transition("findings", 3)
    assert isinstance(t3, str)