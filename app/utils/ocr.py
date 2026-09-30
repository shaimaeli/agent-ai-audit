import os
import cv2
import numpy as np
from PIL import Image

from app.config import BASE_DIR


_ocr_instance = None

def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang="fr",
                show_log=False,
                use_gpu=False
            )
        except Exception as e:
            return None
    return _ocr_instance





def is_pdf_scanned(pdf_path):
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    return False
        return True
    except Exception:
        return True


def extract_text_from_scanned_pdf(pdf_path):
    ocr = get_ocr()
    if not ocr:
        return ""

    try:
        import fitz  
        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img)

            result = ocr.ocr(img_array, cls=True)
            if result and result[0]:
                page_text = "\n".join([line[1][0] for line in result[0] if line[1]])
                all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")

        doc.close()
        full_text = "\n\n".join(all_text)
        return full_text

    except Exception as e:
        return ""


def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if len(text.strip()) > 100:
            return text
    except Exception as e:
        print(f"pdfplumber: {e}")

    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if len(text.strip()) > 100:
            return text
    except Exception as e:
        print(f"PyMuPDF: {e}")

    return extract_text_from_scanned_pdf(pdf_path)


def extract_text_from_image(image_input):
    ocr = get_ocr()
    if not ocr:
        return ""

    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return ""
            img = cv2.imread(image_input)
            if img is None:
                return ""
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            return ""

        result = ocr.ocr(img, cls=True)
        if not result or not result[0]:
            print("Aucun texte détecté dans l'image")
            return ""

        text = "\n".join([line[1][0] for line in result[0] if line[1]])
        return text

    except Exception as e:
        return ""


def extract_text_from_file(file_path):
    if not os.path.exists(file_path):
        return ""

    ext = file_path.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext in {"png", "jpg", "jpeg", "tiff", "bmp"}:
        return extract_text_from_image(file_path)
    else:
        return ""