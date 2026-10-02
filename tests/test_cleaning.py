from app.utils.text_utils import normalize_whitespace, merge_broken_lines

def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"

def test_merge_broken_lines():
    txt = "This is a sentence\nthat continues here."
    merged = merge_broken_lines(txt)
    assert "sentence that continues" in merged