import json
import math
import re
import pandas as pd
from typing import Dict, List
from app.config import GROQ_API_KEY, MODEL_GPT
from app.utils.ocr import extract_text_from_file

try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
except ImportError:
    client = None
    print(" Groq non installé")

def clean_nan(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(i) for i in obj]
    return obj


STANDARD_FIELDS = {
    "raison_sociale": "Nom de l'entreprise",
    "activite": "Activité ou nature de l'activité",
    "dirigeants": "Dirigeants (gérant, associé, directeur)",
    "capital": "Capital social (nombre uniquement, sans MAD)",
    "adresse": "Adresse complète du siège social",
    "exercice": "Année de l'exercice (ex: 2026)",
    "banques": "Liste des banques",
    "fiscalite": "Régime fiscal (IS ou IR)",
    "forme_juridique": "Forme juridique (SARL, SA, SNC, etc.)",
    "ice": "ICE - Identifiant Commun Entreprise (15 chiffres)",
    "rc": "RC - Registre Commerce (court, 4-6 chiffres)",
    "if_val": "IF - Identifiant Fiscal (7-8 chiffres)",
    "secteur": "Secteur d'activité",
    "cnss": "Numéro CNSS",
    "telephone": "Numéro de téléphone",
    "fax": "Numéro de fax",
    "email": "Adresse email",
    "ville": "Ville du siège social",
    "date_creation": "Date de création ou première nomination",
    "date_constitution": "Date de constitution",
    "nom_co_auditeur": "Nom du co-auditeur",
    "nature_mission": "Nature de la mission d'audit",
    "autres_missions": "Autres missions réalisées",
    "effectif": "Nombre de salariés",
    "valeur_nominale": "Valeur nominale des parts/actions",
    "nombre_parts": "Nombre de parts",
    "nombre_actions": "Nombre d'actions",
    "groupe_appartenance": "Groupe d'appartenance",
    "nationalite_groupe": "Nationalité du groupe",
    "societe_mere": "Nom de la société mère",
    "conseil_juridique": "Conseil juridique externe",
    "conseil_fiscal": "Conseil fiscal externe",
    "conseil_autres": "Autres conseillers externes",
    "commissaire_1": "1er commissaire aux comptes",
    "commissaire_2": "2ème commissaire aux comptes",
    "auditeurs_contractuels": "Auditeurs contractuels",
    "sites_production": "Sites de production",
    "agences_commerciales": "Agences commerciales",
    "principaux_clients": "Principaux clients",
    "principaux_fournisseurs": "Principaux fournisseurs",
    "principales_banques": "Principales banques",
}


QUESTIONS_MAINTIEN = {
    "q1_relations_familiales": "Avez-vous connaissance de relations familiales ?",
    "q2_honoraires_excessifs": "Le volume des honoraires implique des liens financiers excessifs ?",
    "q3_honoraires_non_audit": "Les honoraires non audit sont-ils supérieurs aux honoraires d'audit ?",
    "q4_remuneration_insuffisante": "La rémunération est-elle jugée insuffisante ?",
    "q5_honoraires_impayes": "Des honoraires significatifs sont-ils impayés ?",
    "q6_litiges": "Existe-t-il un litige susceptible d'affecter l'indépendance ?",
    "q7_conflits_interets": "Existe-t-il des conflits d'intérêts potentiels ?",
    "q8_services_compromettants": "Des missions compromettent l'objectivité ?",
    "q9_mesures_sauvegarde": "Les mesures de sauvegarde sont-elles insuffisantes ?",
    "q10_co_auditeur_risque": "La situation de co-auditeur comporte-t-elle des risques ?",
    "q11_co_auditeur_independant": "Le co-auditeur est-il indépendant ?",
    "q12_autres_elements": "Avez-vous connaissance d'autres éléments ?",
    "q13_difficultes_continuite": "La société connaît-elle des difficultés de continuité ?",
    "q14_procedure_alerte": "La procédure d'alerte a-t-elle été déclenchée ?",
    "q15_fraudes_dirigeants": "Des fraudes ont-elles été commises ?",
    "q16_anomalies_comptables": "Des anomalies comptables significatives ont-elles été constatées ?",
    "q17_revelation_autorites": "Ces fraudes ont-elles été révélées aux autorités ?",
    "q18_infractions_blanchiment": "Des transactions constituent-elles des infractions ?",
    "q19_revelation_droit": "Ces opérations ont-elles été révélées à qui de droit ?",
    "q20_pratiques_non_conformes": "La société utilise-t-elle des pratiques non conformes ?",
    "q21_donnees_erronees": "La société omet-elle des données induisant en erreur ?",
    "q22_opinions_reserves": "Des opinions avec réserves ont-elles été émises ?",
}

QUESTIONS_ACCEPTATION = {
    "q1_relations_familiales": "Relations familiales, personnelles ou financières ?",
    "q2_prestations_incompatibles": "Des prestations incompatibles avec l'audit ?",
    "q3_conflits_interets": "Conflits d'intérêts avec un autre client ?",
    "q4_litiges": "Litige susceptible d'affecter l'indépendance ?",
    "q5_competences_techniques": "Des compétences techniques particulières nécessaires ?",
    "q6_societe_nouvelle": "Société nouvelle ?",
    "q7_non_renouvellement": "Non renouvellement d'un confrère ?",
    "q8_appel_offre": "Appel d'offre ?",
    "q9_demission": "Démission, empêchement ou décès d'un confrère ?",
    "q10_contacts_precedent": "Contacts avec l'auditeur précédent ?",
    "q11_limitation_controles": "Limitation des contrôles ?",
    "q12_honoraires_insuffisants": "Honoraires insuffisants ou impayés ?",
    "q13_independance_precedent": "Problèmes d'indépendance ?",
    "q14_anomalies_fraudes": "Anomalies comptables résultant de fraudes ?",
    "q15_autres_obstacles": "Autres obstacles ?",
    "q16_budget_suffisant": "Le budget d'honoraires est-il suffisant ?",
}

QUESTIONS_EVENEMENTS = {
    "q1_pv_conseil": "A-t-on examiné les registres des PV ?",
    "q2_assemblees": "Des assemblées sans PV enregistré ?",
    "q3_reunions_comite": "Même démarche pour les comités de direction ?",
    "q4_situations_intermediaires": "Des situations intermédiaires établies ?",
    "q5_principes_comptables": "Établies selon des principes comptables identiques ?",
    "q6_comparaison_cloture": "Comparaison avec les comptes de clôture ?",
    "q7_comparaison_budget": "Comparaison avec le budget ?",
    "q8_modifications_ventes": "Modifications significatives des ventes ?",
    "q9_modifications_charges": "Modifications significatives des charges ?",
    "q10_modifications_marge": "Modifications significatives de la marge ?",
    "q11_pertes_exceptionnelles": "Pertes et profits exceptionnels ?",
    "q12_modifications_capital": "Modifications du capital ?",
    "q13_fonds_roulement": "Modification significative du fonds de roulement ?",
    "q14_passifs_eventuels": "Passifs éventuels devenus certains ?",
    "q15_proces_litiges": "Procès ou litiges nés après clôture ?",
    "q16_controles_fiscaux": "Contrôles fiscaux après clôture ?",
    "q17_immobilisations": "Modifications significatives des immobilisations ?",
    "q18_prises_participation": "Prises de participation significatives ?",
    "q19_moins_value": "Moins-value sur cession d'immobilisations ?",
    "q20_reserves": "La moins-value résulte-t-elle des réserves ?",
    "q21_pertes_stock": "Pertes non provisionnées sur stock ?",
    "q22_pertes_creances": "Pertes non provisionnées sur créances ?",
    "q23_augmentation_stock": "Augmentation anormale des stocks ?",
    "q24_augmentation_creances": "Augmentation anormale des créances ?",
    "q25_situations_intermediaires_comparaison": "Comparaisons avec situations intermédiaires ?",
}

QUESTIONS_INVENTAIRE = {
    "q1_techniques_comptage": "Les techniques de comptage sont-elles bien définies ?",
    "q2_probleme_tare": "A-t-on appréhendé le problème de la tare ?",
    "q3_verification_qualite": "Vérification de la qualité des articles ?",
    "q4_stocks_deprecier": "Les stocks à déprécier sont-ils recensés ?",
    "q5_mouvements_inventaire": "Des mouvements pendant l'inventaire ?",
    "q6_mesures_doubles": "Mesures contre doubles comptages ?",
    "q7_bons_entree_sortie": "Entrées/sorties au vu des bons ?",
    "q8_releve_bons": "Relevé des numéros de bons ?",
    "q9_determination_propriete": "Détermination de la propriété ?",
    "q10_livraisons_fournisseurs": "Livraisons pendant l'inventaire ?",
    "q11_identification_livraisons": "Identification des livraisons ?",
    "q12_expeditions_clients": "Expéditions pendant l'inventaire ?",
    "q13_identification_expeditions": "Identification des expéditions ?",
    "q14_consignation": "Traitement des articles en consignation ?",
    "q15_marchandises_tiers": "Marchandises détenues par des tiers ?",
    "q16_demandes_confirmation": "Demandes de confirmation ?",
    "q17_rapprochement_inventaire": "Rapprochement bons/inventaire permanent ?",
    "q18_comptages_inventaire_permanent": "Comptages comparés par personnes différentes ?",
    "q19_ecarts_quantites": "Raisons des écarts quantités ?",
    "q20_enquete_prealable": "Enquête préalable aux corrections ?",
    "q21_liste_ecarts_direction": "Liste des écarts communiquée ?",
}


def extract_with_regex(text: str) -> Dict:
    result = {}
    if not text:
        return result

    ice_match = re.search(r'ICE[:\s]+(\d{15})', text, re.I)
    if not ice_match:
        ice_match = re.search(r'\b(\d{15})\b', text)
    if ice_match:
        result["ice"] = ice_match.group(1)

    if_match = re.search(r'IF[:\s]+(\d{7,8})', text, re.I)
    if if_match:
        result["if_val"] = if_match.group(1)

    rc_match = re.search(r'R\.?C\.?[:\s]+(\d{4,6})', text, re.I)
    if rc_match:
        result["rc"] = rc_match.group(1)

    cnss_match = re.search(r'CNSS[:\s]+(\d+)', text, re.I)
    if cnss_match:
        result["cnss"] = cnss_match.group(1)

    cap_match = re.search(r'capital.*?(\d[\d\s,.]+)', text, re.I)
    if cap_match:
        cap_clean = cap_match.group(1).replace(" ", "").replace(",", ".")
        try:
            result["capital"] = float(cap_clean)
        except:
            pass

    tel_match = re.search(r'téléphone[:\s]+([+\d\s.-]+)', text, re.I)
    if tel_match:
        result["telephone"] = tel_match.group(1).strip()

    email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
    if email_match:
        result["email"] = email_match.group(1)

    villes = ["tanger", "casablanca", "rabat", "marrakech", "fès", "fez",
              "agadir", "oujda", "tétouan", "tetouan", "safi", "kenitra", "meknes"]
    text_lower = text.lower()
    for v in villes:
        if v in text_lower:
            result["ville"] = v.capitalize()
            break

    eff_match = re.search(r'effectif[:\s]+(\d+)', text, re.I)
    if eff_match:
        result["effectif"] = int(eff_match.group(1))

    rs_match = re.search(r'(?:dénomination|nom du client|raison sociale)[:\s]+([A-Z][A-Za-z\s\.\-]+(?:SARL|SA|SNC|SAS))', text, re.I)
    if rs_match:
        result["raison_sociale"] = rs_match.group(1).strip()

    fj_match = re.search(r'forme juridique[:\s]+(SARL|SA|SNC|SAS)', text, re.I)
    if fj_match:
        result["forme_juridique"] = fj_match.group(1).upper()

    date_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
    if date_match:
        result["date_creation"] = date_match.group(1)

    return result


def extract_entreprise_data_from_text(text: str, doc_type: str = "fiche_signaletique") -> Dict:
    if not text or not text.strip():
        print("Texte vide ")
        return {}

    result = _extract_with_groq(text, doc_type)

    if not result:
        print("Fallback extraction regex...")
        result = extract_with_regex(text)

    result = _post_process_extraction(result, doc_type)
    return result


def _extract_with_groq(text: str, doc_type: str) -> Dict:
    if not client:
        print("Groq non disponible")
        return {}

    if doc_type in ["maintien_mission", "acceptation_mission"]:
        fields_desc = "\n".join([f'    "{k}": ""' for k in STANDARD_FIELDS.keys()])
        questions = QUESTIONS_MAINTIEN if doc_type == "maintien_mission" else QUESTIONS_ACCEPTATION
        questions_desc = "\n".join([f'    "{k}": "OUI ou NON"' for k in questions.keys()])

        prompt = f"""
Tu es un expert en commissariat aux comptes au Maroc.
Analyse le texte suivant et extrais TOUTES les informations disponibles.

Retourne UNIQUEMENT un JSON avec ces clés exactes :
{{
{fields_desc}
    "questions": {{
{questions_desc}
    }},
    "commentaires": {{}},
    "decision": "",
    "mesures_sauvegarde": "",
    "nom_associé": "",
    "date_decision": ""
}}

RÈGLES STRICTES :
1. "rc" = Numéro Registre Commerce (court, 4-6 chiffres)
2. "ice" = Identifiant Commun Entreprise (15 chiffres)
3. "if_val" = Identifiant Fiscal (7-8 chiffres)
4. "capital" = Nombre uniquement, sans "MAD". Mets 0 si non trouvé.
5. Pour les questions OUI/NON : cherche la réponse EXPLICITE dans le texte
6. Si une valeur n'est pas trouvée, mets "" ou [] ou 0 selon le type
7. Pour "dirigeants" cherche : gérant, associé, directeur, responsable
8. "nom_associé" = nom de l'associé responsable de la décision
9. "decision" = MAINTIEN ou NON MAINTIEN ou ACCEPTATION ou REFUS
10. "mesures_sauvegarde" = liste des mesures de sauvegarde

TEXTE :
{text}

IMPORTANT : Retourne uniquement le JSON valide, sans explication.
"""
    elif doc_type == "questionnaire_evenement":
        questions_desc = "\n".join([f'    "{k}": "OUI ou NON ou N.A"' for k in QUESTIONS_EVENEMENTS.keys()])

        prompt = f"""
Tu es un expert en commissariat aux comptes au Maroc.
Analyse le texte suivant et extrais les réponses aux questions sur les événements postérieurs à la clôture.

Retourne UNIQUEMENT un JSON avec ces clés exactes :
{{
    "raison_sociale": "",
    "exercice": "",
    "questions": {{
{questions_desc}
    }},
    "commentaires": {{}},
    "traitement_evenements": {{
        "q1_renseignements_complementaires": "OUI ou NON ou N.A",
        "q2_modification_comptes": "OUI ou NON ou N.A",
        "q3_reserve_rapport": "OUI ou NON ou N.A",
        "q4_continuite_exploitation": "OUI ou NON ou N.A",
        "q5_situation_non_existante": "OUI ou NON ou N.A",
        "q6_continuite_affectee": "OUI ou NON ou N.A",
        "q7_description_evenements": "OUI ou NON ou N.A",
        "q8_mention_appropriee": "OUI ou NON ou N.A"
    }}
}}

TEXTE :
{text}

IMPORTANT : Retourne uniquement le JSON valide.
"""
    elif doc_type == "questionnaire_inventaire":
        questions_desc = "\n".join([f'    "{k}": "OUI ou NON"' for k in QUESTIONS_INVENTAIRE.keys()])

        prompt = f"""
Tu es un expert en commissariat aux comptes au Maroc.
Analyse le texte suivant et extrais les réponses sur l'évaluation de l'inventaire physique.

Retourne UNIQUEMENT un JSON avec ces clés exactes :
{{
    "raison_sociale": "",
    "exercice": "",
    "questions": {{
{questions_desc}
    }},
    "commentaires": {{}},
    "avis_procedures": "",
    "avis_execution": "",
    "suggestions": ""
}}

TEXTE :
{text}

IMPORTANT : Retourne uniquement le JSON valide.
"""
    else:
        fields_desc = "\n".join([f'    "{k}": ""' for k in STANDARD_FIELDS.keys()])

        prompt = f"""
Tu es un expert en commissariat aux comptes au Maroc.
Analyse le texte suivant et extrais TOUTES les informations disponibles.

Retourne UNIQUEMENT un JSON avec ces clés exactes :
{{
{fields_desc}
}}

RÈGLES STRICTES :
1. "rc" = Numéro Registre Commerce (court, 4-6 chiffres)
2. "ice" = Identifiant Commun Entreprise (15 chiffres)
3. "if_val" = Identifiant Fiscal (7-8 chiffres)
4. "capital" = Nombre uniquement, sans "MAD". Mets 0 si non trouvé.
5. "dirigeants" = liste ["Nom1", "Nom2"]
6. "banques" = liste ["Banque1", "Banque2"]
7. Si une valeur n'est pas trouvée, mets "" ou [] ou 0 selon le type
8. Pour "dirigeants" cherche : gérant, associé, directeur, responsable
9. Pour "date_creation" cherche : date de nomination, date de constitution, date création
10. Cherche ces informations partout dans le texte

TEXTE :
{text}

IMPORTANT : Retourne uniquement le JSON valide, sans explication.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
        result = json.loads(raw) if isinstance(raw, str) else raw
        print(f"Groq extraction OK : {len(str(result))} caractères")
        return result

    except Exception as e:
        print(f"Erreur Groq extraction: {e}")
        return {}


def _post_process_extraction(data: Dict, doc_type: str) -> Dict:
    if not data:
        return {}

    if "questions" in data and isinstance(data["questions"], dict):
        for q_key, q_val in data["questions"].items():
            data[q_key] = q_val

    text_decision = str(data.get("decision", ""))
    if "MAINTIEN" in text_decision.upper():
        data["decision"] = "MAINTIEN"
    elif "NON MAINTIEN" in text_decision.upper():
        data["decision"] = "NON MAINTIEN"

    mesures = str(data.get("mesures_sauvegarde", ""))
    if mesures.startswith("- "):
        data["mesures_sauvegarde"] = mesures.replace("- ", "").strip()

    rc = str(data.get("rc", "")).strip()
    ice = str(data.get("ice", "")).strip()
    if_val = str(data.get("if_val", "")).strip()

    if rc and len(rc.replace(" ", "")) >= 15:
        if not ice:
            data["ice"] = rc
        if len(rc.replace(" ", "")) > 15:
            data["rc"] = ""

    if if_val and len(if_val.replace(" ", "")) >= 15:
        if not data.get("ice"):
            data["ice"] = if_val
        data["if_val"] = ""

    if ice and len(ice.replace(" ", "")) < 10:
        if not rc:
            data["rc"] = ice
        data["ice"] = ""

    capital = data.get("capital", 0)
    if isinstance(capital, str):
        capital_clean = capital.replace("MAD", "").replace(" ", "").replace(",", ".")
        try:
            data["capital"] = float(capital_clean) if capital_clean else 0
        except:
            data["capital"] = 0

    if not data.get("ville") and data.get("adresse"):
        adresse = str(data["adresse"]).lower()
        villes = ["tanger", "casablanca", "rabat", "marrakech", "fès", "fez",
                  "agadir", "oujda", "tétouan", "tetouan", "safi", "kenitra", "meknes"]
        for v in villes:
            if v in adresse:
                data["ville"] = v.capitalize()
                break

    if not data.get("exercice"):
        date_creation = str(data.get("date_creation", ""))
        year_match = re.search(r'(20\d{2})', date_creation)
        if year_match:
            data["exercice"] = year_match.group(1)

    for key in data:
        if key.startswith("q") and isinstance(data[key], str):
            val = data[key].strip().upper()
            if val in ["OUI", "YES", "1", "TRUE"]:
                data[key] = "OUI"
            elif val in ["NON", "NO", "0", "FALSE"]:
                data[key] = "NON"

    return data


def extract_data_from_files(file_paths: List[str], doc_type: str = "fiche_signaletique") -> Dict:
    all_text = ""
    raw_texts = []
    financial_data = {}

    for file_path in file_paths:
        ext = file_path.lower()
        if ext.endswith((".pdf", ".png", ".jpg", ".jpeg")):
            extracted = extract_text_from_file(file_path)
            print("=" * 50)
            print(f"TEXTE OCR DE : {file_path}")
            print(extracted[:800] if extracted else "VIDE")
            print("=" * 50)
            all_text += extracted + "\n\n"
            raw_texts.append(extracted)

        elif ext.endswith((".xlsx", ".xls")):
            financial_data.update(extract_financial_data_from_excel(file_path))
        elif ext.endswith(".txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    structured_text = f.read()
                print(f"TXT : {file_path}")
                all_text += structured_text + "\n\n"
                raw_texts.append(structured_text)
            except Exception as e:
                print(f"Erreur TXT: {e}")

    # Extraction LLM
    entreprise_data = extract_entreprise_data_from_text(all_text, doc_type)

    # Stocker le texte brut
    entreprise_data["_raw_texts"] = raw_texts
    entreprise_data["_full_text"] = all_text

    # Regex fallback
    entreprise_data = _fill_missing_with_regex(entreprise_data, all_text)

    print("APRÈS REGEX :")
    for k in ["telephone", "email", "capital", "adresse", "fax"]:
        print(f"   {k} = '{entreprise_data.get(k, 'ABSENT')}'")

    for key, value in financial_data.items():
        if key not in entreprise_data or not entreprise_data[key]:
            entreprise_data[key] = value

    print("DONNÉES FINALES EXTRAITES :")
    for k, v in entreprise_data.items():
        if v and str(v).lower() not in ("none", "null", "n/a", "na", "", "0", "0.0", "[]"):
            v_str = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
            print(f"{k}: {v_str}")

    missing = [k for k in STANDARD_FIELDS.keys() if not entreprise_data.get(k)]
    if missing:
        print(f"Manquants: {', '.join(missing[:10])}")

    return entreprise_data


def _fill_missing_with_regex(data: Dict, text: str) -> Dict:
    if not text:
        return data

    def _missing(key):
        v = str(data.get(key, "")).strip()
        return not v or v.lower() in ("", "none", "null", "n/a", "na", "0", "0.0", "[]")

    if _missing("telephone"):
        m = re.search(r'(?:t[eé]l[eé]phone|t[eé]l|phone)[^\d+]*([+\d][\d\s.\-]{7,})', text, re.I)
        if m:
            data["telephone"] = m.group(1).strip()

    if _missing("fax"):
        m = re.search(r'fax[^\d+]*([+\d][\d\s.\-]{7,})', text, re.I)
        if m:
            data["fax"] = m.group(1).strip()

    if _missing("email"):
        m = re.search(r'([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})', text)
        if m:
            data["email"] = m.group(1).strip()

    if _missing("capital"):
        m = re.search(r'capital\s*(?:social)?\s*[:\s]*([0-9][\d\s,\.]+)\s*(?:MAD|DH|dirhams)?', text, re.I)
        if m:
            cap = m.group(1).replace(" ", "").replace(",", ".")
            try:
                data["capital"] = float(cap)
            except:
                pass

    if _missing("adresse"):
        m = re.search(r'(?:si[eè]ge\s*social|adresse)[^\n:]*[:\s]+([^\n]+)', text, re.I)
        if m:
            data["adresse"] = m.group(1).strip()

    if _missing("ice"):
        m = re.search(r'ICE[^\d]*(\d{15})', text, re.I)
        if not m:
            m = re.search(r'\b(\d{15})\b', text)
        if m:
            data["ice"] = m.group(1)

    if _missing("rc"):
        m = re.search(r'R\.?C\.?\s*[:\s]+(\d{4,6})\b', text, re.I)
        if m:
            data["rc"] = m.group(1)

    if _missing("if_val"):
        m = re.search(r'\bIF\s*[:\s]+(\d{7,8})\b', text, re.I)
        if m:
            data["if_val"] = m.group(1)

    if _missing("cnss"):
        m = re.search(r'CNSS[^\d]*(\d+)', text, re.I)
        if m:
            data["cnss"] = m.group(1)

    if _missing("effectif"):
        m = re.search(r'effectif[^\d]*(\d+)', text, re.I)
        if m:
            data["effectif"] = m.group(1)

    if _missing("ville") and data.get("adresse"):
        for v in ["Tanger","Casablanca","Rabat","Marrakech","Agadir",
                  "Oujda","Tétouan","Safi","Kénitra","Meknès","Fès"]:
            if v.lower() in str(data["adresse"]).lower():
                data["ville"] = v
                break

    if _missing("date_creation"):
        m = re.search(r'(?:nomination|constitution|cr[eé]ation)[^\d]*(\d{2}/\d{2}/\d{4})', text, re.I)
        if m:
            data["date_creation"] = m.group(1)

    if _missing("nom_co_auditeur"):
        m = re.search(r'co[-\s]?auditeur[^\n:]*[:\s]+([^\n]+)', text, re.I)
        if m:
            data["nom_co_auditeur"] = m.group(1).strip()

    if _missing("nature_mission"):
        m = re.search(r'nature\s+de\s+la\s+mission[^\n:]*[:\s]+([^\n]+)', text, re.I)
        if m:
            data["nature_mission"] = m.group(1).strip()

    return data


def extract_financial_data_from_excel(file_path: str) -> Dict:
    try:
        df_raw = pd.read_excel(file_path, sheet_name=None, header=None)
        financial_data = {}

        print("FEUILLES EXCEL :", list(df_raw.keys()))

        for sheet_name, df in df_raw.items():
            df_clean = df.dropna(how="all").fillna("")
            sheet_text = f"=== Feuille : {sheet_name} ===\n"
            sheet_text += df_clean.to_string(index=False, header=False)

            print(f"\nCONTENU '{sheet_name}' (aperçu) :\n{sheet_text[:500]}")
            print("-" * 40)

            result = _analyze_sheet_with_ai(sheet_name, sheet_text)

            if result and result.get("lignes"):
                financial_data["bilan_ia"] = result

                if result.get("raison_sociale"):
                    financial_data["raison_sociale"] = result["raison_sociale"]
                if result.get("exercice"):
                    financial_data["exercice"] = result["exercice"]
                if result.get("capital") and result["capital"] > 0:
                    financial_data["capital"] = result["capital"]

                lignes_actif = [l for l in result["lignes"] if l.get("categorie") == "actif"]
                lignes_passif = [l for l in result["lignes"] if l.get("categorie") == "passif"]

                financial_data["bilan"] = [
                    {"Poste": l["poste"], "Montant_N": l.get("montant_n", 0), "Montant_N1": l.get("montant_n1", 0)}
                    for l in lignes_actif
                ]
                financial_data["bilan_passif"] = [
                    {"Poste": l["poste"], "Montant_N": l.get("montant_n", 0)}
                    for l in lignes_passif
                ]
                financial_data["total_actif"] = result.get("totaux", {}).get("total_actif", 0)

                print(f"bilan_ia créé avec {len(result['lignes'])} postes")
            else:
                print(f"Aucune ligne extraite pour '{sheet_name}'")

        return clean_nan(financial_data)

    except Exception as e:
        print(f"Erreur lecture Excel: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _analyze_sheet_with_ai(sheet_name: str, sheet_text: str) -> Dict:
    if not client:
        return {}

    prompt = f"""
    Tu es un expert-comptable et auditeur au Maroc.
    Voici le contenu brut d'une feuille Excel nommée "{sheet_name}" :

    {sheet_text}

    Analyse ce contenu et retourne un JSON avec exactement cette structure :
    {{
        "type_document": "bilan",
        "structure": "actif_passif",
        "raison_sociale": "",
        "exercice": "",
        "capital": 0.0,
        "lignes": [
            {{
                "poste": "Immobilisations corporelles",
                "categorie": "actif",
                "montant_n": 2800000.0,
                "montant_n1": 0.0,
                "variation_pct": 0.0
            }}
        ],
        "totaux": {{
            "total_actif": 0.0,
            "total_passif": 0.0,
            "total_produits": 0.0,
            "total_charges": 0.0,
            "resultat_net": 0.0
        }},
        "alertes": []
    }}

    RÈGLES IMPORTANTES :
    - type_document : "bilan", "compte_resultat", "grand_livre", "balance" ou "autre"
    - categorie : "actif", "passif", "produit", "charge" ou "autre"
    - Extrais TOUS les postes avec leurs montants RÉELS
    - Ne mets JAMAIS 0 si le montant est présent dans le texte
    - Calcule les totaux actif et passif
    - Détecte les alertes : ratios anormaux, concentrations, valeurs suspectes
    - Retourne uniquement le JSON, sans explication
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
        result = json.loads(raw) if isinstance(raw, str) else raw
        print(f"IA analysé '{sheet_name}' : {result.get('type_document')}")
        return result
    except Exception as e:
        print(f"Erreur analyse IA feuille '{sheet_name}': {e}")
        return {}