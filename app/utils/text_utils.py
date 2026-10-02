"""Text normalization helpers."""
import re


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to a single space and strip ends."""
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def merge_broken_lines(text: str) -> str:
    """
    Merge lines broken mid-sentence by PDF column extraction.
    A line ending with a lowercase letter or comma followed by a newline
    and a continuation (lowercase or mid-word) is joined with a space.
    """
    # Hyphenated line breaks: "chal-\nlenge" → "challenge"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Soft line breaks: line ends with lowercase/comma, next starts lowercase
    text = re.sub(r"([a-z,;])\n([a-z])", r"\1 \2", text)
    return text


def remove_control_chars(text: str) -> str:
    """Strip non-printable control characters except newlines and tabs."""
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", "", text)


def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix