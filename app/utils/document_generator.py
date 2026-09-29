import os
import shutil
import json
import time
import hashlib
import threading
from typing import Dict
import fitz
from app.utils.template_finder import (
    find_template_pdf,
    QUESTIONNAIRE_TYPES,
    GUIDE_TYPES,
    STANDARD_TYPES,
    ALL_VALID_TYPES,
)

from app.config import OUTPUT_FOLDER, BASE_DIR, GROQ_API_KEY, MODEL_GPT, LLM_CACHE_FILE, DOCS_STAGE_FOLDER
from groq import Groq


client = Groq(api_key=GROQ_API_KEY)

_groq_lock     = threading.Lock()
_groq_last_call = 0
MIN_DELAY      = 2.5
_llm_cache     = {}
_cache_dirty   = False

def _load_cache():
    global _llm_cache
    if os.path.exists(LLM_CACHE_FILE):
        try:
            with open(LLM_CACHE_FILE, 'r', encoding='utf-8') as f:
                _llm_cache = json.load(f)
            print(f"Cache : {len(_llm_cache)} entrées")
        except Exception:
            _llm_cache = {}

def _save_cache():
    global _cache_dirty
    if not _cache_dirty:
        return
    try:
        with open(LLM_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_llm_cache, f, ensure_ascii=False, indent=2)
        _cache_dirty = False
    except Exception as e:
        print(f"Cache save : {e}")

def _cache_key(doc_type, flat_data, pfx=""):
    s = json.dumps(sorted(flat_data.items()), ensure_ascii=False)
    h = hashlib.md5(s.encode()).hexdigest()[:12]
    return f"{pfx}{doc_type}_{h}"

_load_cache()

def _groq(prompt, system, max_wait=30):
    global _groq_last_call, _cache_dirty
    deadline = time.time() + max_wait * 60
    attempt  = 0
    while time.time() < deadline:
        with _groq_lock:
            gap = time.time() - _groq_last_call
            if gap < MIN_DELAY:
                time.sleep(MIN_DELAY - gap)
            try:
                attempt += 1
                print(f"Groq tentative {attempt}...")
                r = client.chat.completions.create(
                    model=MODEL_GPT,
                    messages=[{"role":"system","content":system},
                               {"role":"user",  "content":prompt}],
                    temperature=0.0, max_tokens=4000,
                    response_format={"type":"json_object"}
                )
                _groq_last_call = time.time()
                raw = r.choices[0].message.content
                return json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                _groq_last_call = time.time()
                if "429" in str(e) or "rate_limit" in str(e):
                    delay = min(5 * (2**attempt), 300)
                    print(f"Rate limit — attente {delay}s")
                    time.sleep(delay)
                else:
                    print(f"Groq : {e}")
                    return None
    return None



def _safe_name(data):
    return data.get("raison_sociale","entreprise").replace(" ","_").replace("/","-")


def _flatten_data(data: Dict) -> Dict:
    dirs = data.get("dirigeants", [])
    dirs_str = ", ".join(dirs) if isinstance(dirs, list) else str(dirs or "")
    bnq  = data.get("banques", [])
    bnq_str  = ", ".join(bnq)  if isinstance(bnq,  list) else str(bnq  or "")
    rsq  = data.get("risques", [])
    rsq_str  = " | ".join(rsq) if rsq else ""

    flat = dict(data)
    flat.update({
        "dirigeants_str":    dirs_str,
        "banques_str":       bnq_str,
        "risques_str":       rsq_str,
        "decision_maintien": data.get("decision", "MAINTIEN"),
        "nature_mission":    flat.get("nature_mission") or flat.get("activite","Audit légal"),
        "autres_missions":   flat.get("autres_missions") or flat.get("secteur",""),
        "nom_co_auditeur":   flat.get("nom_co_auditeur",""),
        "nom_associé":       flat.get("nom_associé",""),
        "date_decision":     flat.get("date_decision",""),
        "mesures_sauvegarde": flat.get("mesures_sauvegarde",""),
    })

    for a, b in [("date_creation","date_constitution"),("date_creation","date_premiere_nomination"),
                 ("ice","ice_val"),("if_val","if"),
                 ("raison_sociale","nom_du_client"),("activite","nature_activite"),
                 ("adresse","siege_social")]:
        if flat.get(a) and not flat.get(b): flat[b] = flat[a]
        if flat.get(b) and not flat.get(a): flat[a] = flat[b]

    if flat.get("principales_banques") and not flat.get("banques_str"):
        flat["banques_str"] = flat["principales_banques"]

    if not flat.get("ville") and flat.get("adresse"):
        for v in ["tanger","casablanca","rabat","marrakech","agadir","fès","fez",
                  "oujda","tétouan","tetouan","safi","kenitra","meknes"]:
            if v in str(flat["adresse"]).lower():
                flat["ville"] = v.capitalize(); break

    for k in list(flat.keys()):
        v = flat[k]
        if isinstance(v, list):
            if v and all(str(i).strip() for i in v):
                flat[k] = ", ".join(str(i) for i in v if str(i).strip())
            else:
                flat[k] = ""

    for q_key in list(data.keys()):
        if q_key.startswith("q") and "_" in q_key:
            flat[q_key] = data.get(q_key, "NON")

    return flat


INVALID_VALUES = {"", "none", "null", "n/a", "na", "0", "0.0",
                  "non précisé", "non précisée", "non precisé",
                  "non precisee", "undefined", "false", "inconnu",
                  "[]", "[ ]", "vide", "non renseigné", "à compléter"}


def _get_val(key: str, flat: Dict) -> str:
    v = str(flat.get(key, "")).strip()
    if v in ("[]", "[ ]", "['']", '[""]', "None", "nan", "NaN"):
        return ""
    return "" if v.lower() in INVALID_VALUES else v


def _extract_spans(pdf_path: str) -> list:
    spans = []
    doc   = fitz.open(pdf_path)
    for pn, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0: continue
            for line in block["lines"]:
                for sp in line["spans"]:
                    t = sp.get("text","").strip()
                    if t:
                        spans.append({
                            "page": pn, "text": t,
                            "x0": round(sp["bbox"][0],1), "y0": round(sp["bbox"][1],1),
                            "x1": round(sp["bbox"][2],1), "y1": round(sp["bbox"][3],1),
                            "size": round(sp.get("size",8),1),
                        })
    doc.close()
    spans.sort(key=lambda s:(s["page"],round(s["y0"]),s["x0"]))
    return spans


def _extract_rects(pdf_path: str) -> list:
    rects = []
    doc   = fitz.open(pdf_path)
    for pn, page in enumerate(doc):
        for path in page.get_drawings():
            r = path.get("rect")
            if not r: continue
            w, h = r[2]-r[0], r[3]-r[1]
            if w > 15 and h > 3:
                rects.append({
                    "page": pn,
                    "x0": round(r[0],1), "y0": round(r[1],1),
                    "x1": round(r[2],1), "y1": round(r[3],1),
                    "w": round(w,1),     "h":  round(h,1),
                })
    doc.close()
    return rects


def _find_rect_for_label(span: dict, rects: list) -> dict | None:
    candidates = []
    page_width = 595.0

    for r in rects:
        if r["page"] != span["page"]: continue

        v_overlap = min(span["y1"], r["y1"]) - max(span["y0"], r["y0"])
        if v_overlap < 1: continue

        if r["x0"] < span["x1"] - 3: continue

        if r["h"] < 4 or r["h"] > 50: continue

        if r["w"] > page_width * 0.95: continue

        candidates.append(r)

    if not candidates: return None
    return min(candidates, key=lambda r: r["x0"])

def _detect_cols(spans: list) -> tuple:
    oui_x, com_x = 312.0, 434.0
    
    for s in spans:
        t = s["text"].strip().upper().replace(" ","")
        
        if ("OUI" in t and "NON" in t):
            oui_x = s["x0"]
        elif t in ("COMMENTAIRE","COMMENTAIRES","OBSERVATION"):
            com_x = s["x0"]
        
        elif t == "RÉPONSES" or t == "REPONSES":
            oui_x = s["x0"]   
            com_x = s["x0"] + 122  

    return oui_x, com_x


LABEL_MAP = [
    ("dénomination",                    "raison_sociale"),
    ("nom du client",                   "raison_sociale"),
    ("raison sociale",                  "raison_sociale"),
    ("forme juridique",                 "forme_juridique"),
    ("capital social",                  "capital"),
    ("n° d'identification fiscal",      "if_val"),
    ("numéro d'identification fiscal",  "if_val"),
    ("identification fiscal",           "if_val"),
    ("n° if",                           "if_val"),
    ("if :",                            "if_val"),
    ("n° r.c",                          "rc"),
    ("n° rc",                           "rc"),
    ("registre commerce",               "rc"),
    ("n° ice",                          "ice"),
    ("ice :",                           "ice"),
    ("n° cnss",                         "cnss"),
    ("cnss :",                          "cnss"),
    ("siège social",                    "adresse"),
    ("adresse",                         "adresse"),
    ("nature de l'activité",            "activite"),
    ("nature de l activité",            "activite"),
    ("activité",                        "activite"),
    ("téléphone",                       "telephone"),
    ("telephone",                       "telephone"),
    ("fax",                             "fax"),
    ("email",                           "email"),
    ("e-mail",                          "email"),
    ("ville",                           "ville"),
    ("effectif",                        "effectif"),
    ("date de sa constitution",         "date_creation"),
    ("date de constitution",            "date_creation"),
    ("date de création",                "date_creation"),
    ("date de première nomination",     "date_creation"),
    ("date de premiére nomination",     "date_creation"),
    ("date de premiere nomination",     "date_creation"),
    ("valeur nominale",                 "valeur_nominale"),
    ("valeur nominale des parts",       "valeur_nominale"),
    ("nombre de parts",                 "nombre_parts"),
    ("nombre d'actions",                "nombre_actions"),
    ("groupe d'appartenance",           "groupe_appartenance"),
    ("groupe d appartenance",           "groupe_appartenance"),
    ("nationalité du groupe",           "nationalite_groupe"),
    ("nationalite du groupe",           "nationalite_groupe"),
    ("nom de la société mère",          "societe_mere"),
    ("nom de la societe mere",          "societe_mere"),
    ("nom du co-auditeur",              "nom_co_auditeur"),
    ("nom du co auditeur",              "nom_co_auditeur"),
    ("nature de la mission",            "nature_mission"),
    ("réalisation d'autres mission",    "autres_missions"),
    ("realisation d autres mission",    "autres_missions"),
    ("poursuite de la mission",         "decision_maintien"),
    ("nom de l'associé",                "nom_associé"),
    ("nom de l associe",                "nom_associé"),
    ("date de la décision",             "date_decision"),
    ("date de la decision",             "date_decision"),
    ("mesures de sauvegarde",           "mesures_sauvegarde"),
    ("conseil juridique",               "conseil_juridique"),
    ("conseil fiscal",                  "conseil_fiscal"),
    ("autres conseillers",              "conseil_autres"),
    ("autres conseil",                  "conseil_autres"),
    ("1er commissaire aux comptes",     "commissaire_1"),
    ("1er commissaire",                 "commissaire_1"),
    ("premier commissaire",             "commissaire_1"),
    ("2ème commissaire aux comptes",    "commissaire_2"),
    ("2ème commissaire",                "commissaire_2"),
    ("2eme commissaire",                "commissaire_2"),
    ("deuxième commissaire",            "commissaire_2"),
    ("deuxieme commissaire",            "commissaire_2"),
    ("auditeurs contractuels",          "auditeurs_contractuels"),
    ("auditeurs contractuel",           "auditeurs_contractuels"),
    ("sites de production",             "sites_production"),
    ("site de production",              "sites_production"),
    ("agences commerciales",            "agences_commerciales"),
    ("agence commerciale",              "agences_commerciales"),
    ("principaux clients",              "principaux_clients"),
    ("principal client",                "principaux_clients"),
    ("principaux fournisseurs",         "principaux_fournisseurs"),
    ("principal fournisseur",           "principaux_fournisseurs"),
    ("principales banques",             "principales_banques"),
    ("principale banque",               "principales_banques"),
    ("banques",                         "banques_str"),
    ("banque",                          "banques_str"),
]

OUI_NON_DEFAULTS = {
    "relations familiales":           ("NON", "Aucun lien détecté"),
    "liens financiers excessifs":     ("NON", "Honoraires proportionnés"),
    "supérieurs aux honoraires":      ("NON", "Audit > non-audit"),
    "rémunération des temps":         ("NON", "Taux conformes"),
    "honoraires d'un montant":        ("NON", "Honoraires à jour"),
    "litige opposant":                ("NON", "Aucun litige"),
    "conflits d'intérêts":            ("NON", "Aucun conflit"),
    "compromettre l'objectivité":     ("NON", "Services autorisés"),
    "mesures de sauvegarde":          ("NON", "N/A"),
    "indépendant de notre cabinet":   ("OUI", "Cabinets distincts"),
    "autres éléments susceptibles":   ("NON", "RAS"),
    "difficultés sérieuses":          ("NON", "Situation saine"),
    "procédure d'alerte":             ("NON", "Non déclenchée"),
    "fraudes ont-elles été commises": ("NON", "Aucune fraude"),
    "anomalies comptables":           ("NON", "Comptes conformes"),
    "révélation au procureur":        ("NON", "N/A"),
    "blanchiment":                    ("NON", "Transactions normales"),
    "révélation à qui de droit":      ("NON", "N/A"),
    "pratiques non conformes":        ("NON", "CGNC respecté"),
    "données chiffrées":              ("NON", "Données fiables"),
    "opinions avec réserves":         ("NON", "Opinion sans réserve"),
    "techniques de comptage":         ("OUI", "Procédures définies"),
    "problème de la tare":            ("OUI", "Tare appréhendée"),
    "vérification de la qualité":     ("OUI", "Contrôle qualité OK"),
    "stocks à déprécier":             ("OUI", "Recensés et provisionnés"),
    "mouvements pendant":             ("NON", "Stocks gelés"),
    "doubles comptages":              ("OUI", "Mesures prises"),
    "entrées/sorties au vu":          ("OUI", "Bons vérifiés"),
    "relevé des numéros":             ("OUI", "Numéros relevés"),
    "détermination de la propriété":  ("OUI", "Propriété vérifiée"),
    "livraisons pendant":             ("NON", "Aucune livraison"),
    "expéditions pendant":            ("NON", "Aucune expédition"),
    "consignation":                   ("OUI", "Articles identifiés"),
    "marchandises détenues":          ("NON", "Pas de stock tiers"),
    "demandes de confirmation":       ("OUI", "Confirmations envoyées"),
    "rapprochement":                  ("OUI", "Rapprochement effectué"),
    "comptages comparés":             ("OUI", "Double comptage OK"),
    "raisons des écarts":             ("OUI", "Écarts justifiés"),
    "enquête préalable":              ("OUI", "Enquête réalisée"),
    "liste des écarts":               ("OUI", "Liste communiquée"),
    "registres des pv":               ("OUI", "PV examinés"),
    "assemblées sans pv":             ("NON", "Tous PV enregistrés"),
    "comités de direction":           ("OUI", "Démarche identique"),
    "situations intermédiaires":      ("OUI", "Situations établies"),
    "principes comptables identiques":("OUI", "Principes cohérents"),
    "comparaison avec les comptes":   ("OUI", "Comparaison effectuée"),
    "comparaison avec le budget":     ("OUI", "Comparaison OK"),
    "modifications significatives des ventes": ("NON", "Ventes stables"),
    "modifications significatives des charges":("NON", "Charges stables"),
    "modifications significatives de la marge":("NON", "Marge stable"),
    "pertes et profits exceptionnels":("NON", "Aucune perte exceptionnelle"),
    "modifications du capital":       ("NON", "Capital inchangé"),
    "fonds de roulement":             ("NON", "FR stable"),
    "passifs éventuels":              ("NON", "Pas de passifs éventuels"),
    "procès ou litiges":              ("NON", "Aucun litige post-clôture"),
    "contrôles fiscaux":              ("NON", "Aucun contrôle fiscal"),
    "modifications significatives des immobilisations":("NON","Immobilisations stables"),
    "prises de participation":        ("NON", "Aucune prise participation"),
    "moins-value":                    ("NON", "Aucune moins-value"),
    "pertes non provisionnées sur stock":("NON","Stocks correctement évalués"),
    "pertes non provisionnées sur créances":("NON","Créances correctement évaluées"),
    "augmentation anormale des stocks":("NON","Stocks normaux"),
    "augmentation anormale des créances":("NON","Créances normales"),
    "entités consolidées":            ("NON", "Aucune relation détectée"),
    "d'audit ?":                      ("NON", "Honoraires proportionnés"),
    "honoraires d'audit ?":           ("NON", "Audit > non-audit"),
    "sans espoir d'évolution ?":      ("NON", "Taux conformes"),
    "sont-ils impayés ?":             ("NON", "Honoraires à jour"),
    "notre indépendance ?":           ("NON", "Aucun litige"),
    "d'un autre client ?":            ("NON", "Aucun conflit"),
    "du cabinet ?":                   ("NON", "Services autorisés"),
    "et indépendance ?":              ("NON", "N/A"),
    "sur les comptes ?":              ("NON", "Aucun risque"),
    "de notre cabinet ?":             ("OUI", "Cabinets distincts"),
    "décrivez ces situations en":     ("NON", "RAS"),
    "continuité d'exploitation ?":    ("NON", "Situation saine"),
    "a t-elle été déclenchée ?":      ("NON", "Non déclenchée"),
    "détournements d'actifs notamment?": ("NON", "Aucune fraude"),
    "été constatées ?":               ("NON", "Comptes conformes"),
    "de la république":               ("NON", "N/A"),
    "d'argent criminel ?":            ("NON", "Transactions normales"),
    "qui de droit ?":                 ("NON", "N/A"),
    "fidèle des comptes ?":           ("NON", "CGNC respecté"),
    "des comptes ?":                  ("NON", "Données fiables"),
    "dans les rapports ?":            ("NON", "Opinion sans réserve"),
    "(société-mère et filiales intégrées) ?":     ("NON", "Aucune relation"),
    "situations incompatibles avec la mission d'audit ?": ("NON", "Aucune prestation incompatible"),
    "cabinet ou d'un membre du réseau chez un autre client ?": ("NON", "Aucun conflit"),
    "réseau au client susceptible d'affecter notre indépendance ?": ("NON", "Aucun litige"),
    "particulières que le cabinet doit acquérir ou renforcer ?": ("NON", "Compétences disponibles"),
    "société nouvelle":                           ("NON", "Société existante"),
    "non renouvellement d'un confrère":           ("NON", "Premier mandat"),
    "appel d'offre":                              ("NON", "Nomination directe"),
    "démission, empêchement ou décés d'un confrère": ("NON", "Sans objet"),
    "limitation des contrôles":                   ("NON", "Accès complet"),
    "honoraires insuffisants ou impayés":         ("NON", "Honoraires conformes"),
    "indépendance":                               ("OUI", "Indépendance confirmée"),
    "anomalies comptables résultant de fraudes ou d'erreurs": ("NON", "Aucune anomalie"),
    "autres obstacles à la mission":              ("NON", "Aucun obstacle"),
    "motifs de la démission le cas échéant":      ("NON", "Sans objet"),
    "société, des risques et de l'environnement de contrôle ?": ("OUI", "Budget suffisant"),
    "contrôle interne":                           ("Faible", "Contrôle interne satisfaisant"),
    "principes comptables et information financière": ("Faible", "Principes conformes CGNC"),
    "réputation et intégrité des dirigeants":     ("Faible", "Réputation positive"),
    "compétence et stabilité du management":      ("Faible", "Management stable"),
    "clarté de la situation juridique":           ("Faible", "Situation claire"),
    "clarté de la situation fiscale":             ("Faible", "Fiscalité conforme"),
    "relations entre les dirigeants":             ("Faible", "Relations harmonieuses"),
    "relations avec les autorités de contrôle":   ("Faible", "Relations normales"),
    "respect des textes et réglements":           ("Faible", "Conformité vérifiée"),
    "situation financière, rentabilité":          ("Faible", "Situation saine"),
    "sécurité, stabilité du secteur":             ("Faible", "Secteur stable"),
    "perspectives du marché":                     ("Faible", "Perspectives positives"),
    # Inventaire physique — fins de questions
    "couverts ?":                         ("OUI", "Tous stocks couverts"),
    "facilité ?":                         ("OUI", "Rangement adéquat"),
    "l'identification des stocks est-elle faite ?": ("OUI", "Stocks identifiés"),
    "des quantités théoriques)":          ("OUI", "Comptage en aveugle"),
    "les écarts sont-ils analysés ?":     ("OUI", "Écarts analysés"),
    "les stocks comptés sont-ils clairement identifiés ?": ("OUI", "Stocks marqués"),
    "physique et l'inventaire permanent sont-ils analysés ?": ("OUI", "Écarts analysés"),
    "dans les équipes de comptage ?":     ("OUI", "Indépendance assurée"),
    "sur le travail qu'elles devaient effectuer ?": ("OUI", "Instructions données"),
    "stocks ?":                           ("OUI", "Compétences vérifiées"),
    "elles bien définies ?":              ("OUI", "Techniques définies"),
    "emballages, échantillonnage des liquides et produits chimiques)": ("OUI", "Qualité vérifiée"),
    "d'inventaire ?":                     ("NON", "Stocks gelés"),
    "omissions ? si oui, lesquelles ?":   ("OUI", "Mesures prises"),
    "sortie ?":                           ("OUI", "Bons vérifiés"),
    "opérations d'inventaire.":           ("OUI", "Numéros relevés"),
    "inventoriés ?":                      ("OUI", "Propriété déterminée"),
    "des fournisseurs pendant l'inventaire ?": ("NON", "Aucune livraison"),
    "ces livraisons ? si oui, lesquelles ?": ("NON", "N/A"),
    "l'inventaire ?":                     ("NON", "Aucune expédition"),
    "ces expéditions ? si oui, lesquelles ?": ("NON", "N/A"),
    "comment a-t-on traité les articles en consignation ?": ("OUI", "Articles identifiés"),
    "oui, comment ?":                     ("NON", "Pas de stock tiers"),
    "a-t-on fait des demandes de confirmation ?": ("OUI", "Confirmations envoyées"),
    "permanent ?":                        ("OUI", "Rapprochement effectué"),
    "des personnes différentes des magasiniers et des équipes de comptage ?": ("OUI", "Double comptage OK"),
    "inventaire permanent.":              ("OUI", "Écarts justifiés"),
    "substantielle de l'inventaire permanent ?": ("OUI", "Enquête réalisée"),
    "la liste des écarts est-elle communiquée à la direction ?": ("OUI", "Liste communiquée"),
}


def _build_insertions(spans: list, rects: list,
                       flat_data: Dict, doc_type: str) -> list:
    if doc_type == "questionnaire_pri":
        return _build_pri_insertions(spans, rects, flat_data)
    if doc_type == "questionnaire_verification":
        return _build_verification_insertions(spans, rects, flat_data)
    if doc_type == "questionnaire_fin":
        return _build_fin_insertions(spans, rects, flat_data)
    return _build_standard_insertions(spans, rects, flat_data, doc_type)


def _build_fin_insertions(spans: list, rects: list, flat_data: Dict) -> list:
    insertions = []
    used = set()

    oui_x = 344.0
    com_x = 448.0

    for s in spans:
        t = s["text"].strip().upper().replace(" ","")
        if "OUI" in t and "NON" in t and "N.A" in t:
            oui_x = s["x0"]
        elif t in ("COMMENTAIRE", "COMMENTAIRES"):
            com_x = s["x0"]

    print(f"Colonnes fin : Oui/Non=x{oui_x:.0f}, Com=x{com_x:.0f}")

    header_ys = set()
    for s in spans:
        t = s["text"].strip().upper().replace(" ","")
        if "OUI" in t and "NON" in t:
            header_ys.add((s["page"], round(s["y0"])))

    full_text = flat_data.get("_full_text", "")

    IGNORE = {"questionnaire de fin", "cabinet el housny", "fait par",
              "mission:", "exercice du:", "b_qfm", "1 / 2", "2 / 2",
              "oui/non/n.a", "oui / non", "commentaire", "commentaires"}

    for span in spans:
        txt  = span["text"].strip()
        tlow = txt.lower()
        page = span["page"]
        y0   = span["y0"]
        size = span["size"]

        if any(h in tlow for h in IGNORE):
            continue

        if (page, round(y0)) in header_ys:
            continue

        import re
        if re.match(r'^\d+[\.\d]*\s+\w', txt):
            continue
        
        clean = tlow.strip("- •0123456789./: ")
        if len(clean) < 4:
            continue

        line_key = (page, round(y0))
        if line_key in used:
            continue
        used.add(line_key)

        observation = _find_fin_observation(tlow, full_text)

        insertions.append({
            "page":  page,
            "x":     oui_x + 3,
            "y":     y0 + size * 0.80,
            "value": "Oui",
            "size":  size * 0.88,
            "max_x": com_x - 2,
            "type":  "oui_non",
        })

        if observation:
            insertions.append({
                "page":  page,
                "x":     com_x + 3,
                "y":     y0 + size * 0.80,
                "value": observation[:35],
                "size":  max(5.5, size * 0.75),
                "max_x": 541.0,
                "type":  "commentaire",
            })

    print(f"   → {len(insertions)} insertions fin de mission")
    return insertions


def _find_fin_observation(label: str, full_text: str) -> str:
    OBS_MAP = {
        "continuité d'exploitation":     "Risques évalués",
        "concordance":                    "Opinions cohérentes",
        "orientations":                   "Intégrées au programme",
        "feuilles maîtresses":            "Rapprochées et validées",
        "contrôles programmés":           "Exécutés",
        "documentés":                     "Travaux référencés",
        "points en suspens":              "Traités",
        "conclusions":                    "Conformes aux éléments",
        "note de synthèse":               "Préparée et validée",
        "ajustements":                    "Communiqués à la direction",
        "comptes complets":               "Cohérents",
        "etic":                           "Examiné, conforme",
        "vérifications spécifiques":      "Réalisées",
        "communications sur les contrôle":"Adressées à la direction",
        "irrégularités":                  "Aucune irrégularité",
        "rapport général":                "Préparé et cohérent",
        "assertions":                     "Couvertes",
        "opinion":                        "Préparée",
        "états de synthèse":              "Contrôlés et conformes",
        "événements post":                "Examinés",
        "contrôles substantifs":          "Réalisés",
        "dossier":                        "Mis à jour",
        "examen analytique":              "Effectué",
        "temps passés":                   "Comparés au budget",
        "écarts":                         "Analysés et justifiés",
        "revue qualité":                  "Procédures appliquées",
        "dossier permanent":              "Mis à jour",
        "n+1":                            "Points recensés",
        "rapport spécial":                "Préparé",
        "identification des risques":     "Conforme aux normes",
        "plan de mission":                "Validé",
        "lettre de mission":              "Signée",
        "connaissance":                   "Actualisée",
        "tests de conformité":            "Réalisés",
        "tests de procédures":            "Réalisés",
        "approche d'audit":               "Adaptée aux risques",
        "programme de contrôle":          "Défini et exécuté",
        "revue réciproque":               "Réalisée",
        "compte rendu écrit":             "Préparé et transmis",
        "compte rendu oral":              "Note au dossier",
        "réunion de clôture":             "Tenue avec la direction",
        "recommandations":                "Formulées",
        "présence":                       "Convoqué à l'AG",
        "assemblée":                      "Rapports présentés",
        "prise de connaissance":          "Réalisée",
        "contrôle interne":               "Évalué",
        "révision des comptes":           "Réalisée",
        "finalisation":                   "Terminée",
        "co-commissariat":                "N/A",
        "revue des feuilles":             "Revue effectuée",
        "mise à jour":                    "Complétée",
        "rapprochement":                  "Effectué",
        "analyse":                        "Réalisée",
    }

    for keyword, obs in OBS_MAP.items():
        if keyword in label:
            return obs
    return ""


def _build_verification_insertions(spans: list, rects: list, flat_data: Dict) -> list:
    insertions = []
    used = set()

    fait_x = 319.0
    obs_x  = 432.0

    for s in spans:
        t = s["text"].strip()
        tu = t.upper().replace(" ", "")
        if "FAIT" in tu and "N.A" in tu:
            fait_x = s["x0"]
        elif tu in ("OBSERVATIONS", "OBSERVATION"):
            obs_x = s["x0"]

    print(f"Colonnes : Fait=x{fait_x:.0f}, Obs=x{obs_x:.0f}")

    full_text = flat_data.get("_full_text", "")
    section_map = _parse_verification_report(full_text)

    header_ys = set()
    for s in spans:
        if "fait ou n.a" in s["text"].lower():
            header_ys.add((s["page"], round(s["y0"])))

    for span in spans:
        txt  = span["text"].strip()
        tlow = txt.lower()
        page = span["page"]
        y0   = span["y0"]
        size = span["size"]

        if "fait ou n.a" in tlow or tlow in ("observations", "observation"):
            continue

        if (page, round(y0)) in header_ys:
            continue

        if len(tlow.strip("- •0123456789./")) < 5:
            continue

        if any(h in tlow for h in ["questionnaire des vérifications",
                                    "cabinet el housny", "fait par",
                                    "mission:", "exercice du:", "b_qvs"]):
            continue

        line_key = (page, round(y0))
        if line_key in used:
            continue
        used.add(line_key)

        observation = _find_observation(tlow, section_map)

        insertions.append({
            "page":  page,
            "x":     fait_x + 3,
            "y":     y0 + size * 0.80,
            "value": "Fait",
            "size":  size * 0.88,
            "max_x": obs_x - 2,
            "type":  "oui_non",
        })

        if observation:
            insertions.append({
                "page":  page,
                "x":     obs_x + 3,
                "y":     y0 + size * 0.80,
                "value": observation[:35],
                "size":  max(5.5, size * 0.75),
                "max_x": 541.0,
                "type":  "commentaire",
            })

    print(f" {len(insertions)} insertions vérification")
    return insertions


def _parse_verification_report(text: str) -> dict:
    if not text:
        return {}

    sections = {}
    current_section = None
    current_lines = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line and line[0].isdigit() and "." in line[:3]:
            if current_section and current_lines:
                sections[current_section] = " ".join(current_lines[:3])
            current_section = line.lower()
            current_lines = []
        elif current_section:
            current_lines.append(line)

    if current_section and current_lines:
        sections[current_section] = " ".join(current_lines[:3])

    return sections


def _find_observation(label: str, section_map: dict) -> str:
    OBSERVATION_MAP = {
        "règles légales":       "Conformité vérifiée",
        "statut":               "Statuts conformes",
        "administrateur":       "Mandats vérifiés",
        "registre":             "Registres à jour",
        "pv":                   "PV disponibles",
        "extrait":              "Documents obtenus",
        "récépissé":            "Dépôt vérifié",
        "convention":          "Conventions examinées",
        "lettre circulaire":    "Lettre envoyée",
        "rapport spécial":      "Rapport établi",
        "tableau de suivi":     "Tableau établi",
        "interdit":             "Aucune convention interdite",
        "emprunt":              "Aucun emprunt dirigeant",
        "découvert":            "Aucun découvert",
        "cautionnement":        "Aucune caution irrégulière",
        "rapport de gestion":   "Rapport conforme",
        "sincérité":            "Informations cohérentes",
        "actionnaire":          "Droits respectés",
        "convocation":          "Délais respectés",
        "dividende":            "Répartition conforme",
        "filiale":              "Aucune acquisition",
        "participation":        "Participations conformes",
        "irrégularité":         "Aucune irrégularité",
        "délit":                "Aucun fait délictueux",
        "image fidèle":         "États conformes",
        "action de garantie":   "Actions conformes",
        "égalité":              "Égalité assurée",
        "droit de vote":        "Droits préservés",
    }

    for keyword, obs in OBSERVATION_MAP.items():
        if keyword in label:
            return obs

    return ""


def _build_pri_insertions(spans: list, rects: list, flat_data: Dict) -> list:
    raw_texts = flat_data.get("_raw_texts", [])
    full_text = flat_data.get("_full_text", "")

    if not full_text:
        full_text = _build_reference_text_from_data(flat_data)

    qa_map = _parse_reference_text_to_qa(full_text)

    if not qa_map:
        qa_map = _build_default_qa_map(flat_data)

    print(f"Q&A map : {len(qa_map)} paires trouvées")

    insertions = []
    used_lines = set()

    for span in spans:
        txt = span["text"].strip()
        tlow = txt.lower()

        if ":" not in txt:
            continue

        label_part = tlow.split(":")[0].strip()
        label_clean = label_part.lstrip("- •").strip()

        best_answer = None
        best_score = 0

        for map_label, answer in qa_map.items():
            if not answer or str(answer).strip().lower() in INVALID_VALUES:
                continue

            map_clean = map_label.lower().strip()
            map_words = set(map_clean.split())
            label_words = set(label_clean.split())

            if not map_words:
                continue

            common = map_words & label_words
            score = len(common) / max(len(map_words), 1)

            if label_clean.startswith(map_clean[:20]):
                score += 0.3
            if map_clean in label_clean:
                score += 0.2

            if score > best_score and score > 0.35:  
                best_score = score
                best_answer = answer

        if not best_answer:
            continue

        line_key = (span["page"], round(span["y0"] / 2) * 2)
        if line_key in used_lines:
            continue
        used_lines.add(line_key)

        colon_idx = txt.find(":")
        char_width = span["size"] * 0.5
        x_insert = span["x0"] + (colon_idx + 1) * char_width + 3

        if colon_idx == len(txt) - 1:
            x_insert = span["x1"] + 3

        y_insert = span["y0"] + span["size"] * 0.85

        clean_zone = {
            "x0": x_insert - 2,
            "y0": span["y0"] - 1,
            "x1": 545,
            "y1": span["y1"] + 2,
        }

        display_answer = str(best_answer).strip()
        if len(display_answer) > 100:
            display_answer = display_answer[:97] + "…"

        insertions.append({
            "page": span["page"],
            "x": x_insert,
            "y": y_insert,
            "value": display_answer,
            "size": span["size"] * 0.85,
            "max_x": 540,
            "type": "guide_clean",
            "clean_zone": clean_zone,
        })

    print(f"   → {len(insertions)} insertions PRI trouvées")
    for ins in insertions[:5]:
        print(f"      p{ins['page']} x={ins['x']:.0f} y={ins['y']:.0f} → '{ins['value'][:40]}'")
    if len(insertions) > 5:
        print(f"      ... et {len(insertions)-5} autres")

    return insertions


def _parse_reference_text_to_qa(text: str) -> Dict:
    if not text:
        return {}

    qa_map = {}
    lines = text.split("\n")
    current_question = None
    current_answer_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_question and current_answer_lines:
                answer = " ".join(current_answer_lines).strip()
                if answer and answer.lower() not in INVALID_VALUES:
                    qa_map[current_question] = answer
                current_question = None
                current_answer_lines = []
            continue

        if ":" in line and not line.startswith("*") and not line.startswith("-"):
            if current_question and current_answer_lines:
                answer = " ".join(current_answer_lines).strip()
                if answer:
                    qa_map[current_question] = answer

            parts = line.split(":", 1)
            current_question = parts[0].strip().lstrip("- •").strip()
            current_answer_lines = [parts[1].strip()] if len(parts) > 1 else []

        elif current_question and (line.startswith("*") or line.startswith("-") or line.startswith("  ")):
            current_answer_lines.append(line.lstrip("*- ").strip())

        elif line.startswith("*") or line.startswith("-"):
            content = line.lstrip("*- ").strip()
            if ":" in content:
                if current_question and current_answer_lines:
                    answer = " ".join(current_answer_lines).strip()
                    if answer:
                        qa_map[current_question] = answer

                parts = content.split(":", 1)
                current_question = parts[0].strip()
                current_answer_lines = [parts[1].strip()] if len(parts) > 1 else []
            elif current_question:
                current_answer_lines.append(content)

    if current_question and current_answer_lines:
        answer = " ".join(current_answer_lines).strip()
        if answer:
            qa_map[current_question] = answer

    print(f"Q&A parse : {len(qa_map)} paires trouvées")
    return qa_map


def _build_reference_text_from_data(flat_data: Dict) -> str:
    lines = []

    if flat_data.get("activite"):
        lines.append(f"Nature du secteur d'activité : {flat_data['activite']}")
    if flat_data.get("ville"):
        lines.append(f"Implantation géographique : {flat_data['ville']}")
    if flat_data.get("forme_juridique"):
        lines.append(f"Forme actuelle : {flat_data['forme_juridique']}")
    if flat_data.get("dirigeants_str"):
        lines.append(f"Liste des dirigeants : {flat_data['dirigeants_str']}")
    if flat_data.get("capital"):
        lines.append(f"Capital social : {flat_data['capital']} MAD")
    if flat_data.get("adresse"):
        lines.append(f"Siège social : {flat_data['adresse']}")
    if flat_data.get("exercice"):
        lines.append(f"Exercice : {flat_data['exercice']}")
    if flat_data.get("email"):
        lines.append(f"Email : {flat_data['email']}")
    if flat_data.get("telephone"):
        lines.append(f"Téléphone : {flat_data['telephone']}")

    return "\n".join(lines)


def _build_default_qa_map(flat_data: Dict) -> Dict:
    return {
        "nature du secteur d'activité": flat_data.get("activite", "À compléter"),
        "présentation générale": flat_data.get("resume", "À compléter"),
        "implantation géographique": flat_data.get("ville", "À compléter"),
        "forme actuelle": flat_data.get("forme_juridique", "À compléter"),
        "liste des dirigeants": flat_data.get("dirigeants_str", "À compléter"),
        "capital social": str(flat_data.get("capital", "À compléter")),
        "siège social": flat_data.get("adresse", "À compléter"),
        "exercice": flat_data.get("exercice", "À compléter"),
        "email": flat_data.get("email", "À compléter"),
        "téléphone": flat_data.get("telephone", "À compléter"),
        "fax": flat_data.get("fax", "À compléter"),
        "ice": flat_data.get("ice", "À compléter"),
        "rc": flat_data.get("rc", "À compléter"),
        "if": flat_data.get("if_val", "À compléter"),
    }


def _build_standard_insertions(spans: list, rects: list,
                                flat_data: Dict, doc_type: str) -> list:
    insertions = []
    used       = set()
    is_q       = doc_type in QUESTIONNAIRE_TYPES
    oui_x, com_x = _detect_cols(spans)

    PURE_HEADERS = {"nom", "fonction", "commentaire", "observation",
                    "oui / non", "oui/non"}
    AMBIGUOUS = {"email", "e-mail", "téléphone", "telephone"}

    for span in spans:
        txt  = span["text"].strip()
        tlow = txt.lower().rstrip(" :")
        if len(tlow) < 2: continue

        page = span["page"]
        y0   = span["y0"]
        size = span["size"]

        if tlow in PURE_HEADERS:
            continue

        if tlow in AMBIGUOUS:
            same_y = [
                s["text"].lower().strip().rstrip(" :")
                for s in spans
                if abs(s["y0"] - y0) < 3
                and s["page"] == page
                and s["text"].strip() != txt
            ]
            if any(h in same_y for h in {"nom", "fonction", "commentaire"}):
                continue

        for label_frag, data_key in LABEL_MAP:
            if label_frag not in tlow:
                continue

            y_key      = (page, round(y0), label_frag)
            data_y_key = (page, round(y0), data_key)
            if y_key in used or data_y_key in used:
                break

            value = _get_val(data_key, flat_data)
            if not value:
                break

            rect = _find_rect_for_label(span, rects)
            if rect:
                x     = rect["x0"] + 3
                y     = rect["y0"] + size * 0.80
                max_x = rect["x1"] - 2
            else:
                x     = span["x1"] + 5
                y     = y0 + size * 0.80
                max_x = 540.0

            used.add(y_key)
            used.add(data_y_key)
            insertions.append({
                "page": page, "x": x, "y": y,
                "value": value, "size": size,
                "max_x": max_x, "type": "identification",
            })
            break

        if is_q and "?" in txt:
            oui_key = (page, round(y0), "oui")
            if oui_key in used:
                continue

            answer, comment = "NON", ""
            for frag, (ans, com) in OUI_NON_DEFAULTS.items():
                if frag in tlow:
                    answer, comment = ans, com
                    break

            used.add(oui_key)
            insertions.append({
                "page": page, "x": oui_x + 3, "y": y0 + size * 0.80,
                "value": answer, "size": size,
                "max_x": com_x - 2, "type": "oui_non",
            })
            if comment:
                insertions.append({
                    "page": page, "x": com_x + 3, "y": y0 + size * 0.80,
                    "value": comment, "size": max(6.0, size * 0.80),
                    "max_x": 541.0, "type": "commentaire",
                })

    if is_q:
        insertions = _llm_add_missing_comments(
            spans, flat_data, doc_type, insertions, oui_x, com_x
        )

    print(f"   → {len(insertions)} insertions")
    for ins in insertions:
        print(f"      [{ins.get('type','id')}] "
              f"p{ins['page']} x={ins['x']:.0f} y={ins['y']:.0f} "
              f"→ '{ins['value'][:35]}'")
    return insertions


def _llm_add_missing_comments(spans, flat_data, doc_type,
                               existing, oui_x, com_x) -> list:
    answered_y = {round(i["y"]) for i in existing if i.get("type") == "oui_non"}
    has_com_y  = {round(i["y"]) for i in existing if i.get("type") == "commentaire"}
    need_com   = [
        {"y": s["y0"], "page": s["page"], "text": s["text"][:80], "size": s["size"]}
        for s in spans
        if "?" in s["text"]
        and round(s["y0"] + s["size"] * 0.80) in answered_y
        and round(s["y0"] + s["size"] * 0.80) not in has_com_y
    ]
    if not need_com: return existing

    ent = {k: v for k, v in flat_data.items()
           if v and k in ["raison_sociale","activite","risques_str","dirigeants_str","exercice"]}

    prompt = f"""CAC Maroc. Pour chaque question, commentaire professionnel max 35 chars.
Questions : {json.dumps(need_com, ensure_ascii=False)}
Entreprise : {json.dumps(ent, ensure_ascii=False)}
JSON : {{"comments":[{{"y":266.0,"page":0,"text":"Commentaire"}}]}}"""

    try:
        r = _groq(prompt, "Expert CAC Maroc. JSON valide uniquement.")
        if r:
            for c in r.get("comments", []):
                t = str(c.get("text","")).strip()
                if t and t.lower() not in INVALID_VALUES:
                    existing.append({
                        "page": int(c.get("page",0)),
                        "x": com_x + 3,
                        "y": float(c.get("y",0)) + 7.8 * 0.80,
                        "value": t, "size": 6.5,
                        "max_x": 541.0, "type": "commentaire",
                    })
            print(f"   LLM comments : +{len(r.get('comments',[]))}")
    except Exception as e:
        print(f"LLM comments : {e}")
    return existing



def _write_on_pdf(template_path: str, insertions: list, output_path: str):
    FORBIDDEN = {"non précisé","non précisée","non precisé","non precisee",
                 "n/a","inconnu","inconnue","à compléter","undefined","none","null"}

    doc = fitz.open(template_path)

    page_insertions = {}
    for ins in insertions:
        pn = int(ins["page"])
        if pn not in page_insertions:
            page_insertions[pn] = []
        page_insertions[pn].append(ins)

    for pn, ins_list in page_insertions.items():
        if pn >= len(doc):
            continue

        page = doc[pn]

        ins_list.sort(key=lambda i: i["y"])

        for ins in ins_list:
            if "clean_zone" not in ins:
                continue

            zone = ins["clean_zone"]
            rect = fitz.Rect(zone["x0"], zone["y0"], zone["x1"], zone["y1"])

            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)

        for ins in ins_list:
            x = float(ins["x"])
            y = float(ins["y"])
            value = str(ins["value"]).strip()
            size = float(ins.get("size", 7.8))
            max_x = float(ins.get("max_x", 540))
            itype = ins.get("type", "identification")

            if not value or value.lower() in FORBIDDEN:
                continue

            if itype == "guide_clean":
                fs, color = size, (0, 0, 0)  # Noir standard
            elif itype == "oui_non":
                fs, color = size * 0.95, (0, 0, 0)
            elif itype == "commentaire":
                fs, color = max(5.5, size * 0.78), (0.3, 0.3, 0.3)
            else:
                fs, color = size * 0.88, (0, 0, 0)

            avail = max_x - x - 2
            if avail <= 4:
                continue

            try:
                w = fitz.get_text_length(value, fontname="helv", fontsize=fs)
            except:
                w = len(value) * fs * 0.5

            if w > avail:
                while len(value) > 2:
                    value = value[:-1]
                    try:
                        w = fitz.get_text_length(value + "…", fontname="helv", fontsize=fs)
                    except:
                        w = len(value) * fs * 0.5
                    if w <= avail:
                        value += "…"
                        break

            page.insert_text((x, y), value, fontsize=fs, color=color, fontname="helv")

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"PDF généré : {output_path}")



def _generate_document(doc_type: str, data: Dict) -> str:
    if doc_type not in ALL_VALID_TYPES:
        print(f"Type invalide : '{doc_type}' — valides : {sorted(ALL_VALID_TYPES)}")
        return None

    safe        = _safe_name(data)
    output_path = os.path.join(OUTPUT_FOLDER, f"{doc_type}_{safe}.pdf")
    flat        = _flatten_data(data)

    print(f" Données disponibles :")
    for k,v in flat.items():
        if v and str(v).lower() not in ("none","null","n/a","na",""):
            print(f"   {k}: {v}")

    tpl = find_template_pdf(doc_type)
    if not tpl:
        print(f"ERREUR CRITIQUE : Aucun modèle PDF trouvé pour '{doc_type}'")
        print(f"   Placez un modèle dans : {DOCS_STAGE_FOLDER}")
        return None

    print(f" [{doc_type}] {os.path.basename(tpl)}")
    spans = _extract_spans(tpl)
    rects = _extract_rects(tpl)
    print(f"   {len(spans)} spans | {len(rects)} rects")

    insertions = _build_insertions(spans, rects, flat, doc_type)

    if not insertions:
        print(" Aucun champ détecté — copie brute du modèle (non rempli)")
        shutil.copy(tpl, output_path)
        return output_path

    _write_on_pdf(tpl, insertions, output_path)
    return output_path


def generate_fiche_signaletique(data):      return _generate_document("fiche_signaletique",         data)
def generate_acceptation_mission(data):     return _generate_document("acceptation_mission",         data)
def generate_maintien_mission(data):        return _generate_document("maintien_mission",             data)
def generate_questionnaire_pri(data):       return _generate_document("questionnaire_pri",            data)
def generate_questionnaire_inventaire(data):return _generate_document("questionnaire_inventaire",     data)
def generate_questionnaire_verification(data):return _generate_document("questionnaire_verification", data)
def generate_questionnaire_evenement(data): return _generate_document("questionnaire_evenement",      data)
def generate_questionnaire_fin(data):       return _generate_document("questionnaire_fin",            data)