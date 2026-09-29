import os
import cv2
import numpy as np
from PIL import Image

# Import LOCAL uniquement
from app.config import BASE_DIR


# ═══════════════════════════════════════════════════════════════════════════
# PADDLEOCR - Singleton
# ═══════════════════════════════════════════════════════════════════════════

_ocr_instance = None

def get_ocr():
    """Retourne l'instance PaddleOCR unique (lazy loading)."""
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            print("🔄 Chargement PaddleOCR (première fois)...")
            _ocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang="fr",
                show_log=False,
                use_gpu=False
            )
            print("✅ PaddleOCR prêt")
        except Exception as e:
            print(f"❌ Erreur chargement PaddleOCR: {e}")
            return None
    return _ocr_instance


# ═══════════════════════════════════════════════════════════════════════════
# DÉTECTION PDF SCANNÉ
# ═══════════════════════════════════════════════════════════════════════════

def is_pdf_scanned(pdf_path):
    """Détecte si un PDF est scanné (pas de texte natif)."""
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


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION PDF
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_scanned_pdf(pdf_path):
    """Extrait le texte d'un PDF scanné via OCR."""
    ocr = get_ocr()
    if not ocr:
        print("❌ OCR non disponible")
        return ""

    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        all_text = []

        print(f"🔍 OCR sur {len(doc)} pages...")
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
        print(f"✅ OCR PDF : {len(full_text)} caractères sur {len(doc)} pages")
        return full_text

    except Exception as e:
        print(f"❌ Erreur OCR PDF: {e}")
        return ""


def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un PDF (natif ou scanné)."""
    text = ""

    # Étape 1 : pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if len(text.strip()) > 100:
            print(f"✅ pdfplumber : {len(text)} caractères")
            return text
    except Exception as e:
        print(f"⚠️ pdfplumber: {e}")

    # Étape 2 : PyMuPDF fallback
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if len(text.strip()) > 100:
            print(f"✅ PyMuPDF : {len(text)} caractères")
            return text
    except Exception as e:
        print(f"⚠️ PyMuPDF: {e}")

    # Étape 3 : PDF scanné → OCR
    print("📄 PDF probablement scanné → lancement OCR...")
    return extract_text_from_scanned_pdf(pdf_path)


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION IMAGE
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_image(image_input):
    """Extrait le texte d'une image avec PaddleOCR."""
    ocr = get_ocr()
    if not ocr:
        return ""

    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                print(f"❌ Image non trouvée: {image_input}")
                return ""
            img = cv2.imread(image_input)
            if img is None:
                print(f"❌ Impossible de lire l'image: {image_input}")
                return ""
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            print(f"❌ Type non supporté: {type(image_input)}")
            return ""

        result = ocr.ocr(img, cls=True)
        if not result or not result[0]:
            print("⚠️ Aucun texte détecté dans l'image")
            return ""

        text = "\n".join([line[1][0] for line in result[0] if line[1]])
        print(f"✅ Image OCR : {len(text)} caractères")
        return text

    except Exception as e:
        print(f"❌ Erreur OCR image: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_file(file_path):
    """Extrait le texte d'un fichier PDF ou image."""
    if not os.path.exists(file_path):
        print(f"❌ Fichier non trouvé: {file_path}")
        return ""

    ext = file_path.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext in {"png", "jpg", "jpeg", "tiff", "bmp"}:
        return extract_text_from_image(file_path)
    else:
        print(f"❌ Format non supporté: {ext}")
        return ""