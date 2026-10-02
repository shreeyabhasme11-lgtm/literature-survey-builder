from app.utils.logger import get_logger
log = get_logger(__name__)

def ocr_page(pdf_page) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io, os
        from app.config import TESSERACT_CMD
        if os.name == "nt" and TESSERACT_CMD != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        mat = pdf_page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(mat.tobytes("png")))
        return pytesseract.image_to_string(img, lang="eng")
    except ImportError:
        log.warning("pytesseract/Pillow not installed — OCR unavailable")
        return ""
    except Exception as e:
        log.warning(f"OCR failed: {e}")
        return ""