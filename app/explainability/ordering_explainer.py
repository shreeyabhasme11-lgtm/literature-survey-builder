from app.metadata.schema import SentenceMeta

def explain_ordering(s: SentenceMeta, prev=None) -> str:
    if s.ordering_reason:
        return s.ordering_reason
    if prev is None:
        return "First sentence in section — highest TextRank score."
    return f"Placed after '{prev.sent_id}' by greedy similarity ordering in '{s.section}'."