from app.metadata.schema import SentenceMeta

def explain_score(s: SentenceMeta) -> str:
    level = "high" if s.textrank_score > 0.05 else ("moderate" if s.textrank_score > 0.02 else "low")
    return f"TextRank score {s.textrank_score:.4f} ({level} importance)."