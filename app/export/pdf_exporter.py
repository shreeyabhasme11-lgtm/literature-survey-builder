from pathlib import Path
from typing import Dict, List
from app.export.html_renderer import render_html
from app.utils.logger import get_logger

log = get_logger(__name__)


def export_pdf(survey: Dict[str, List[str]], traces: List[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(survey, traces)

    # Resolve lock/permission issues on Windows by finding a unique filename
    actual_path = out_path
    counter = 1
    while True:
        try:
            # Check if we have write permission and the file is not locked
            if actual_path.exists():
                with open(actual_path, "r+b") as f:
                    pass
            else:
                with open(actual_path, "ab") as f:
                    pass
            break
        except PermissionError:
            actual_path = out_path.parent / f"{out_path.stem}_{counter}{out_path.suffix}"
            counter += 1
        except Exception:
            # Let other errors raise during actual writing
            break

    # Try WeasyPrint first, fall back to ReportLab
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(actual_path))
        log.info(f"PDF exported via WeasyPrint → {actual_path}")
        return actual_path
    except Exception as e:
        log.warning(f"WeasyPrint failed ({e}); falling back to ReportLab")

    try:
        _export_reportlab(survey, traces, actual_path)
        return actual_path
    except Exception as e:
        log.error(f"ReportLab export failed: {e}")
        raise


def _export_reportlab(survey: Dict[str, List[str]], traces: List[dict], out_path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                 textColor=colors.HexColor("#01696f"))
    head_style  = ParagraphStyle("Head2", parent=styles["Heading2"],
                                 textColor=colors.HexColor("#01696f"))
    body_style  = ParagraphStyle("Body2", parent=styles["Normal"],
                                 fontSize=11, leading=17)
    italic_style = ParagraphStyle("Italic2", parent=styles["Normal"],
                                  fontSize=10, leading=14,
                                  textColor=colors.HexColor("#555555"),
                                  fontName="Helvetica-Oblique")

    story = [Paragraph("Explainable Literature Survey", title_style), Spacer(1, 16)]

    for section, paragraphs in survey.items():
        story.append(Paragraph(section.replace("_", " ").title(), head_style))
        story.append(Spacer(1, 6))
        for para in paragraphs:
            if para.startswith("*") and para.endswith("*"):
                story.append(Paragraph(para[1:-1], italic_style))
            else:
                story.append(Paragraph(para, body_style))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

    doc.build(story)
    log.info(f"PDF exported via ReportLab → {out_path}")