"""Format citation markers for selected sentences."""
from app.metadata.schema import SentenceMeta


def format_citation(s: SentenceMeta) -> str:
    """Returns a bracketed citation string like [DocName, p.4]"""
    name = (
        s.filename
        .replace(".pdf", "")
        .replace("_", " ")
        .strip()
    )
    # Truncate very long names
    if len(name) > 40:
        name = name[:40]
    return f"[{name}, p.{s.page_num}]"