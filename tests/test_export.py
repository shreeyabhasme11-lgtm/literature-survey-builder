import tempfile
from pathlib import Path
from app.export.json_exporter import export_json
from app.export.html_renderer import render_html

def test_json_export():
    traces = [{"sent_id": "s1", "text": "hello", "section": "introduction"}]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trace.json"
        export_json(traces, path)
        assert path.exists()
        assert "hello" in path.read_text()

def test_html_render():
    survey = {"introduction": ["*Transition.*", "This is a test sentence. [Doc, p.1]"]}
    html = render_html(survey, [])
    assert "<h2>" in html
    assert "test sentence" in html
