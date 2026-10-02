"""Page-wise text extraction using PyMuPDF with Tesseract OCR fallback."""
from pathlib import Path
from typing import List

from app.metadata.schema import DocumentMeta, PageMeta
from app.config import OCR_CHAR_THRESHOLD
from app.utils.logger import get_logger

log = get_logger(__name__)


def extract_document(doc: DocumentMeta) -> List[PageMeta]:
    """
    Extract text from every page of a PDF.
    Falls back to OCR for pages with fewer than OCR_CHAR_THRESHOLD characters.
    Returns a list of PageMeta objects (one per page).
    Returns [] only if the file cannot be opened at all.
    """
    path = Path(doc.filepath)
    if not path.exists():
        log.error(f"File not found: {path}")
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.error("PyMuPDF (fitz) not installed. Run: pip install pymupdf")
        return []

    pages: List[PageMeta] = []

    try:
        pdf = fitz.open(str(path))
    except Exception as e:
        log.error(f"Cannot open PDF '{path.name}': {e}")
        return []

    log.info(f"Extracting '{doc.filename}' — {pdf.page_count} pages")

    for page_idx in range(pdf.page_count):
        page_num = page_idx + 1
        try:
            page = pdf[page_idx]
            text = page.get_text("text") or ""
            text = text.strip()
            char_count = len(text)
            ocr_used = False

            # OCR fallback for low-text pages
            if char_count < OCR_CHAR_THRESHOLD:
                ocr_text = _ocr_page(page, doc.filename, page_num)
                if ocr_text:
                    text = ocr_text
                    char_count = len(text)
                    ocr_used = True

            pages.append(PageMeta(
                doc_id=doc.doc_id,
                page_num=page_num,
                char_count=char_count,
                ocr_used=ocr_used,
                text=text,
            ))

        except Exception as e:
            log.warning(f"Error on page {page_num} of '{doc.filename}': {e}")
            # Append empty page so page numbering stays consistent
            pages.append(PageMeta(
                doc_id=doc.doc_id,
                page_num=page_num,
                char_count=0,
                ocr_used=False,
                text="",
            ))

    pdf.close()
    log.info(
        f"'{doc.filename}': extracted {len(pages)} pages, "
        f"{sum(p.char_count for p in pages)} total chars"
    )
    return pages


def _ocr_page(page, filename: str, page_num: int) -> str:
    """
    Render the page to an image and run Tesseract OCR.
    Returns extracted text or empty string on failure.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render at 2x resolution for better OCR accuracy
        mat = page.get_pixmap(dpi=200)
        img_bytes = mat.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang="eng")
        log.info(f"OCR used on '{filename}' p.{page_num} — {len(text)} chars")
        return text.strip()

    except ImportError:
        log.warning("pytesseract or Pillow not installed; skipping OCR")
        return ""
    except Exception as e:
        log.warning(f"OCR failed on '{filename}' p.{page_num}: {e}")
        return ""