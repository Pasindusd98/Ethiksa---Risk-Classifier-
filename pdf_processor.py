import io
import numpy as np
from .config import OCR_ZOOM, OCR_LANG, OCR_CONF_THRESHOLD

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import cv2
except Exception:
    cv2 = None

def extract_text_from_pdf(pdf_path, zoom=OCR_ZOOM, ocr_lang=OCR_LANG, ocr_conf_threshold=OCR_CONF_THRESHOLD):
    pages = []
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction. Install pymupdf.")
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc.load_page(i)
        txt = page.get_text("text").strip()
        page_info = {"page_num": i, "text": txt, "is_selectable": bool(txt), "ocr_boxes": None}
        if not txt:
            # OCR fallback
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            if Image is None or cv2 is None or pytesseract is None:
                raise RuntimeError("OCR libraries (PIL, cv2, pytesseract) not available. Cannot perform OCR fallback.")

            img = Image.open(io.BytesIO(pix.tobytes()))
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ocr_data = pytesseract.image_to_data(th, lang=ocr_lang, output_type=pytesseract.Output.DICT)
            words = []
            text_acc = []
            n_boxes = len(ocr_data['text'])
            for idx in range(n_boxes):
                w = str(ocr_data['text'][idx]).strip()
                conf_raw = ocr_data['conf'][idx]
                try:
                    conf = float(conf_raw)
                except Exception:
                    try:
                        conf = float(conf_raw) if conf_raw else -1.0
                    except Exception:
                        conf = -1.0
                if w and conf >= ocr_conf_threshold:
                    left = int(ocr_data['left'][idx])
                    top = int(ocr_data['top'][idx])
                    width = int(ocr_data['width'][idx])
                    height = int(ocr_data['height'][idx])
                    words.append({'word': w, 'left': left, 'top': top, 'width': width, 'height': height, 'conf': conf})
                    text_acc.append(w)
            ocr_text = " ".join(text_acc).strip()
            page_info['text'] = ocr_text
            page_info['is_selectable'] = False
            page_info['ocr_boxes'] = words
        pages.append(page_info)
    doc.close()
    return pages
