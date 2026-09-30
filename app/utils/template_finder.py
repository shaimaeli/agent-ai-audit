import os
from app.config import DOCS_STAGE_FOLDER

TEMPLATE_PATTERNS = {
    "fiche_signaletique":         ["fiche_signaletique", "fiche_signalitique", "signaletique", "signalitique", "fiche"],
    "acceptation_mission":        ["acceptation", "acceptation_mission", "acceptation de la mission"],
    "maintien_mission":           ["maintien", "maintien_mission", "maintien de la mission"],
    "questionnaire_pri":          ["prise_de_connaissance", "prise de connaissance", "questionnaire_pri", "questionnaire prise de connaissance", "guide de prise de connaissance"],
    "questionnaire_inventaire":   ["inventaire_physique", "inventaire physique", "questionnaire_inventaire", "questionnaire de l inventaire", "l_inventaire"],
    "questionnaire_verification": ["verification_specifique", "verification specifique", "questionnaire_verification", "questionnaire de verification"],
    "questionnaire_evenement":    ["evenement", "poste_cloture", "poste cloture", "questionnaire_evenement", "evenement poste"],
    "questionnaire_fin":          ["fin_de_mission", "fin de mission", "questionnaire_fin", "questionnaire fin de mission"],
}

QUESTIONNAIRE_TYPES = {
    "questionnaire_pri", "questionnaire_inventaire",
    "questionnaire_verification", "questionnaire_evenement", "questionnaire_fin",
    "maintien_mission", "acceptation_mission",
}

GUIDE_TYPES = {"questionnaire_pri"}

STANDARD_TYPES = {"fiche_signaletique", "acceptation_mission", "maintien_mission"}
ALL_VALID_TYPES = QUESTIONNAIRE_TYPES | STANDARD_TYPES


def find_template_pdf(doc_type: str) -> str | None:
    if not os.path.exists(DOCS_STAGE_FOLDER):
        return None
    
    files = [f for f in os.listdir(DOCS_STAGE_FOLDER) if f.lower().endswith(".pdf")]
    if not files:
        return None

    patterns = TEMPLATE_PATTERNS.get(doc_type, [doc_type])

    for fname in sorted(files):
        fn = fname.lower().replace(" ", "_").replace("-", "_").replace("'", "_")
        for p in patterns:
            p_norm = p.lower().replace(" ", "_").replace("-", "_").replace("'", "_")
            if p_norm in fn:
                full_path = os.path.join(DOCS_STAGE_FOLDER, fname)
                return full_path
    return None