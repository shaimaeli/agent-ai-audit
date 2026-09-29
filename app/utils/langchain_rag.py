# app/utils/langchain_rag.py
import os
import time
import threading
from typing import Dict, List

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from app.config import GROQ_API_KEY, MODEL_GPT, BASE_DIR
except ModuleNotFoundError:
    import sys
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent)
    from config import GROQ_API_KEY, MODEL_GPT, BASE_DIR

# ── Chemins ───────────────────────────────────────────────────────────────────
DOCS_STAGE_FOLDER = os.path.join(BASE_DIR, "docs_stage")
CHROMA_DB_PATH    = os.path.join(BASE_DIR, "chroma_db")
MODEL_PATH        = os.path.join(BASE_DIR, "models", "paraphrase-multilingual-MiniLM-L12-v2")

os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(DOCS_STAGE_FOLDER, exist_ok=True)

# ── Modèles ───────────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ── Rate limiter pour RAG ───────────────────────────────────────────────────
_rag_lock = threading.Lock()
_rag_last_call = 0
RAG_MIN_DELAY = 2.5

def _rag_groq_call(prompt, max_tokens=4000):
    """Wrapper avec rate limiting pour le RAG."""
    global _rag_last_call
    
    with _rag_lock:
        wait = RAG_MIN_DELAY - (time.time() - _rag_last_call)
        if wait > 0:
            time.sleep(wait)
        
        try:
            response = client.chat.completions.create(
                model=MODEL_GPT,
                messages=[
                    {"role": "system", "content": "Tu es un assistant spécialisé en audit et commissariat aux comptes au Maroc. Ta priorité absolue est de reproduire FIDÈLEMENT la structure des documents modèles fournis, en te contentant de remplacer les valeurs par celles de l'entreprise cible. Tu ne modifies jamais la structure, les titres, ni l'ordre des sections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens
            )
            _rag_last_call = time.time()
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Erreur RAG Groq: {e}")
            _rag_last_call = time.time()
            return ""

# ── Mapping mots-clés → type de document ──────────────────────────────────────
# [SEULEMENT les 8 types que tu as dans docs_stage]
DOC_TYPE_MAP = {
    "fiche signaletique":       "fiche_signaletique",
    "fiche signalitique":       "fiche_signaletique",
    "signalitique":             "fiche_signaletique",
    "signaletique":             "fiche_signaletique",
    "signalétique":             "fiche_signaletique",
    "fiche":                    "fiche_signaletique",
    "acceptation de la mission":"acceptation_mission",
    "acceptation":              "acceptation_mission",
    "maintien de la mission":   "maintien_mission",
    "maintien":                 "maintien_mission",
    "inventaire physique":      "questionnaire_inventaire",
    "verification specifique":  "questionnaire_verification",
    "vérification spécifique":  "questionnaire_verification",
    "poste cloture":            "questionnaire_evenement",
    "poste clôture":            "questionnaire_evenement",
    "evenement":                "questionnaire_evenement",
    "événement":                "questionnaire_evenement",
    "prise de connaissance":    "questionnaire_pri",
    "fin de mission":           "questionnaire_fin",
}


def _detect_doc_type(filename: str) -> str:
    name_lower = filename.lower()
    sorted_keywords = sorted(DOC_TYPE_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for keyword, doc_type in sorted_keywords:
        if keyword in name_lower:
            return doc_type
    return "autre"


def _get_vectorstore(doc_type: str) -> Chroma:
    persist_dir = os.path.join(CHROMA_DB_PATH, doc_type)
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(
        collection_name=doc_type,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )


def index_all_docs() -> Dict:
    if not os.path.exists(DOCS_STAGE_FOLDER):
        print(f"⚠️ Dossier introuvable : {DOCS_STAGE_FOLDER}")
        return {}

    files = [f for f in os.listdir(DOCS_STAGE_FOLDER) if f.lower().endswith(".pdf")]
    if not files:
        print("⚠️ Aucun PDF dans docs_stage/")
        return {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )

    indexed = {}

    for filename in files:
        filepath = os.path.join(DOCS_STAGE_FOLDER, filename)
        doc_type = _detect_doc_type(filename)
        print(f"\n📄 '{filename}' → type : '{doc_type}'")

        try:
            loader    = PyPDFLoader(filepath)
            documents = loader.load()
        except Exception as e:
            print(f"  ❌ Erreur chargement : {e}")
            continue

        if not documents:
            print(f"  ⚠️ PDF vide ou illisible")
            continue

        total_text = "".join([doc.page_content for doc in documents])
        if len(total_text.strip()) < 50:
            print(f"  ⚠️ Texte trop court ({len(total_text)} caractères)")
            continue

        for doc in documents:
            doc.metadata["source"]   = filename
            doc.metadata["doc_type"] = doc_type

        chunks = splitter.split_documents(documents)
        print(f"  ✂️  {len(chunks)} chunks créés")

        try:
            vectorstore = _get_vectorstore(doc_type)
            try:
                existing = vectorstore._collection.count()
                if existing > 0:
                    results = vectorstore._collection.get(where={"source": filename})
                    if results and results.get("ids") and len(results["ids"]) > 0:
                        print(f"  ⚠️ Déjà indexé — ignoré")
                        continue
            except Exception:
                pass

            vectorstore.add_documents(chunks)
            indexed[filename] = {"doc_type": doc_type, "chunks": len(chunks)}
            print(f"  ✅ Indexé avec succès")

        except Exception as e:
            print(f"  ❌ Erreur indexation : {e}")
            continue

    print(f"\n📚 Résumé : {len(indexed)} fichier(s) indexé(s)")
    for fname, info in indexed.items():
        print(f"  • {fname} → {info['doc_type']} ({info['chunks']} chunks)")

    return indexed


def list_indexed_docs() -> Dict:
    result = {}
    # SEULEMENT les 8 types valides
    all_types = [
        "fiche_signaletique",
        "acceptation_mission",
        "maintien_mission",
        "questionnaire_pri",
        "questionnaire_inventaire",
        "questionnaire_verification",
        "questionnaire_evenement",
        "questionnaire_fin",
        "autre"
    ]
    for doc_type in all_types:
        try:
            vs    = _get_vectorstore(doc_type)
            count = vs._collection.count()
            if count > 0:
                items   = vs._collection.get()
                sources = list(set(m.get("source", "?") for m in items["metadatas"]))
                result[doc_type] = {"chunks": count, "sources": sources}
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPTS — SEULEMENT les 8 documents
# ─────────────────────────────────────────────────────────────────────────────
PROMPTS = {

    "fiche_signaletique": """Tu es un commissaire aux comptes au Maroc.

Voici le document MODÈLE exact (fiche signalétique réelle) issu de ta base documentaire :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure que ce modèle (mêmes titres, mêmes rubriques, même ordre, même mise en page), mais en remplaçant toutes les valeurs par celles de l'entreprise suivante :

{entreprise_info}

Règles STRICTES :
- Ne supprime AUCUNE section ni rubrique présente dans le modèle.
- Ne crée AUCUNE section absente du modèle.
- Si une valeur n'est pas disponible, écris "N/A".
- Conserve les en-têtes de colonnes, séparateurs, numéros de rubriques.
- Retourne uniquement le contenu rempli, sans explication ni commentaire.""",

    "acceptation_mission": """Tu es un commissaire aux comptes au Maroc.

Voici le document MODÈLE exact (acceptation de mission réelle) :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure (même en-tête, mêmes clauses, même ordre, mêmes formules), en remplaçant uniquement les informations spécifiques par celles de l'entreprise :

{entreprise_info}

Règles STRICTES :
- Garde la numérotation des articles/clauses telle quelle.
- Ne supprime AUCUN article ni clause.
- Si une valeur est inconnue, écris "[ à compléter ]".
- Retourne uniquement la lettre, sans explication.""",

    "maintien_mission": """Tu es un commissaire aux comptes au Maroc.

Voici le document MODÈLE exact (maintien de mission réelle) :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure (même en-tête, mêmes clauses, même ordre, mêmes formules), en remplaçant uniquement les informations spécifiques par celles de l'entreprise :

{entreprise_info}

Règles STRICTES :
- Garde la numérotation des articles/clauses telle quelle.
- Ne supprime AUCUN article ni clause.
- Si une valeur est inconnue, écris "[ à compléter ]".
- Retourne uniquement la lettre, sans explication.""",

    "questionnaire_pri": """Tu es un auditeur senior au Maroc.

Voici le document MODÈLE exact (questionnaire prise de connaissance réel) :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure, en adaptant l'en-tête à l'entreprise :

{entreprise_info}

Règles STRICTES :
- Conserve TOUTES les questions dans le même ordre.
- Retourne uniquement le questionnaire, sans explication.""",

    "questionnaire_inventaire": """Tu es un auditeur senior au Maroc.

Voici le document MODÈLE exact (questionnaire inventaire physique réel) :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure, en adaptant l'en-tête à l'entreprise :

{entreprise_info}

Règles STRICTES :
- Conserve TOUTES les questions dans le même ordre.
- Retourne uniquement le questionnaire, sans explication.""",

    "questionnaire_verification": """Tu es un auditeur senior au Maroc.

Voici le document MODÈLE exact (questionnaire vérification spécifique réel) :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure, en adaptant l'en-tête à l'entreprise :

{entreprise_info}

Règles STRICTES :
- Conserve TOUTES les questions dans le même ordre.
- Retourne uniquement le questionnaire, sans explication.""",

    "questionnaire_evenement": """Tu es un auditeur senior au Maroc.

Voici le document MODÈLE exact (questionnaire événements post-clôture réel) :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure, en adaptant l'en-tête à l'entreprise :

{entreprise_info}

Règles STRICTES :
- Conserve TOUTES les questions dans le même ordre.
- Retourne uniquement le questionnaire, sans explication.""",

    "questionnaire_fin": """Tu es un auditeur senior au Maroc.

Voici le document MODÈLE exact (questionnaire fin de mission réel) :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure, en adaptant l'en-tête à l'entreprise :

{entreprise_info}

Règles STRICTES :
- Conserve TOUTES les questions dans le même ordre.
- Retourne uniquement le questionnaire, sans explication.""",
}

DEFAULT_PROMPT = """Tu es un expert-comptable et commissaire aux comptes au Maroc.

Voici le document MODÈLE exact issu de ta base documentaire :
========== DÉBUT DU MODÈLE ==========
{context}
========== FIN DU MODÈLE ==========

Ta mission : reproduire EXACTEMENT la même structure que ce modèle, en remplaçant toutes les valeurs par celles de l'entreprise suivante :

{entreprise_info}

Règles STRICTES :
- Ne supprime AUCUNE section du modèle.
- Si une valeur est inconnue, écris "N/A".
- Retourne uniquement le contenu rempli, sans explication."""


def _build_entreprise_info(data: Dict) -> str:
    return (
        f"Raison sociale  : {data.get('raison_sociale',  'N/A')}\n"
        f"Forme juridique : {data.get('forme_juridique', 'N/A')}\n"
        f"ICE             : {data.get('ice',             'N/A')}\n"
        f"RC              : {data.get('rc',              'N/A')}\n"
        f"IF              : {data.get('if_val',          'N/A')}\n"
        f"Capital social  : {data.get('capital',         'N/A')} MAD\n"
        f"Activité        : {data.get('activite',        'N/A')}\n"
        f"Adresse         : {data.get('adresse',         'N/A')}\n"
        f"Exercice        : {data.get('exercice',        'N/A')}\n"
        f"Dirigeants      : {data.get('dirigeants_str',  'N/A')}\n"
        f"Banques         : {data.get('banques_str',     'N/A')}\n"
        f"Fiscalité       : {data.get('fiscalite',       'N/A')}\n"
        f"Secteur         : {data.get('secteur',         'N/A')}"
    )


def _retrieve_full_context(doc_type: str, entreprise_info: str) -> str:
    vectorstore = _get_vectorstore(doc_type)
    count = vectorstore._collection.count()

    if count == 0:
        print(f"⚠️ Aucun document modèle indexé pour '{doc_type}'")
        return ""

    k    = min(20, count)
    docs = vectorstore.similarity_search(entreprise_info, k=k)

    docs_sorted = sorted(
        docs,
        key=lambda d: (
            d.metadata.get("source", ""),
            d.metadata.get("page", 0)
        )
    )

    context = "\n\n".join([d.page_content for d in docs_sorted])
    print(f"✅ RAG '{doc_type}' : {len(docs_sorted)} chunks récupérés")
    return context


def generate_with_rag(doc_type: str, data: Dict) -> str:
    entreprise_info = _build_entreprise_info(data)
    context         = _retrieve_full_context(doc_type, entreprise_info)

    if not context:
        print(f"⚠️ Aucun modèle disponible pour '{doc_type}'")
        context = "Aucun document modèle disponible. Génère un document professionnel standard marocain."

    template = PROMPTS.get(doc_type, DEFAULT_PROMPT)
    prompt_text = template.format(
        context=context,
        entreprise_info=entreprise_info
    )

    # ── UTILISE LE RATE LIMITER ──
    result = _rag_groq_call(prompt_text, max_tokens=4000)
    
    if not result:
        return ""
    
    print(f"✅ Généré : {len(result)} caractères")
    return result.strip()