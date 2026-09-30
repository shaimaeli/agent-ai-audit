import os
from typing import Dict, Optional

from app.config import OUTPUT_FOLDER


def _find_pdf(doc_type: str, data: Dict) -> Optional[str]:
    safe = (data.get("raison_sociale") or "entreprise") \
        .replace(" ", "_").replace("/", "-")

    exact = os.path.join(OUTPUT_FOLDER, f"{doc_type}_{safe}.pdf")
    if os.path.exists(exact):
        return exact

    prefix = f"{doc_type}_"
    candidates = [
        f for f in os.listdir(OUTPUT_FOLDER)
        if f.startswith(prefix) and f.endswith(".pdf")
    ]
    if candidates:
        candidates.sort(
            key=lambda f: os.path.getmtime(os.path.join(OUTPUT_FOLDER, f)),
            reverse=True
        )
        return os.path.join(OUTPUT_FOLDER, candidates[0])

    return None


def _safe_name(data: Dict) -> str:
    return (data.get("raison_sociale") or "entreprise") \
        .replace(" ", "_").replace("/", "-")


def pdf_to_word(doc_type: str, data: Dict,
                output_path: str = None) -> str:
    from pdf2docx import Converter

    pdf_path = _find_pdf(doc_type, data)
    if not pdf_path:
        raise FileNotFoundError("PDF introuvable.")

    if not output_path:
        output_path = os.path.join(
            OUTPUT_FOLDER, f"{doc_type}_{_safe_name(data)}.docx"
        )

    cv = Converter(pdf_path)
    cv.convert(output_path, start=0, end=None)
    cv.close()

    return output_path