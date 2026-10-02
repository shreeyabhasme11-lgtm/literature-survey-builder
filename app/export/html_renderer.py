from typing import Dict, List
from app.composition.templates import SURVEY_PREAMBLE, SECTION_HEADER_TEMPLATES


def render_html(survey: Dict[str, List[str]], traces: List[dict]) -> str:
    body_parts = [f"<p>{SURVEY_PREAMBLE}</p>"]
    for section, paragraphs in survey.items():
        header = SECTION_HEADER_TEMPLATES.get(section, f"## {section}\n\n")
        title = header.replace("##", "").replace("\n", "").strip()
        body_parts.append(f"<h2>{title}</h2>")
        for para in paragraphs:
            if para.startswith("*") and para.endswith("*"):
                body_parts.append(f'<p class="transition"><em>{para[1:-1]}</em></p>')
            else:
                body_parts.append(f"<p>{para}</p>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Literature Survey</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 820px; margin: 40px auto;
         font-size: 16px; line-height: 1.75; color: #222; }}
  h1   {{ font-size: 1.8em; border-bottom: 2px solid #01696f; padding-bottom: 8px; }}
  h2   {{ font-size: 1.3em; color: #01696f; margin-top: 2em; }}
  p    {{ text-align: justify; margin: 0.8em 0; }}
  .transition {{ color: #555; font-style: italic; }}
  .citation   {{ color: #01696f; font-size: 0.85em; font-weight: bold; }}
</style>
</head>
<body>
<h1>Explainable Literature Survey</h1>
{"".join(body_parts)}
</body>
</html>"""