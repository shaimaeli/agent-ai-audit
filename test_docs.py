from typing import Dict, List, Tuple
from groq import Groq
import json
import time
import threading

from app.config import GROQ_API_KEY, MODEL_GPT

client = Groq(api_key=GROQ_API_KEY)

_ai_lock = threading.Lock()
_ai_last_call = 0
MIN_DELAY = 2.5

def _groq_call(prompt: str, max_tokens: int = 2500) -> str:
    """Appel Groq avec rate limiting global."""
    global _ai_last_call
    
    with _ai_lock:
        wait = MIN_DELAY - (time.time() - _ai_last_call)
        if wait > 0:
            time.sleep(wait)
        
        try:
            response = client.chat.completions.create(
                model=MODEL_GPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            _ai_last_call = time.time()
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f" Erreur Groq: {e}")
            _ai_last_call = time.time()
            return ""

def analyze_all(data: Dict) -> Tuple[List[str], List[str], str]:
    """
    UN SEUL appel Groq qui retourne :
    - risques
    - commentaires  
    - résumé
    """
    prompt = f"""Tu es un expert en audit et commissariat aux comptes au Maroc.

Analyse les données suivantes et retourne UNIQUEMENT un JSON avec cette structure exacte :

{{
  "risques": [
    "Baisse anormale de la trésorerie de 40% par rapport à N-1.",
    "Incohérence entre le capital déclaré et le bilan."
  ],
  "commentaires": [
    "Variation de 35% des charges externes : à justifier.",
    "Ratio de liquidité générale de 0.8 : risque de trésorerie."
  ],
  "resume": "La société XYZ SARL, spécialisée dans le commerce de gros, présente une hausse du CA de 12% en 2025. Cependant, la trésorerie a baissé de 15%."
}}

DONNÉES ENTREPRISE :
{json.dumps(data, ensure_ascii=False, indent=2)}

RÈGLES :
- Risques : 3 à 5 alertes sur variations, incohérences, comptes dormants, TVA, ratios
- Commentaires : 3 à 5 points de contrôle pour la feuille de maîtrise
- Résumé : 3-4 phrases synthétiques
- Retourne UNIQUEMENT le JSON, sans texte avant/après"""

    raw = _groq_call(prompt, max_tokens=2500)
    
    if not raw:
        return [], [], ""

    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = raw[start:end]
            result = json.loads(json_str)
        else:
            result = json.loads(raw)
        
        risques = [r.strip().lstrip("- ").strip() for r in result.get("risques", []) if r.strip()]
        commentaires = [c.strip().lstrip("- ").strip() for c in result.get("commentaires", []) if c.strip()]
        resume = result.get("resume", "").strip()
        
        return risques, commentaires, resume
        
    except Exception as e:
        print(f"⚠️ Erreur parsing JSON: {e}")
        return [], [], ""

_risques_cache = {}
_commentaires_cache = {}
_resume_cache = {}

def analyze_financial_risks(data: Dict) -> List[str]:
    cache_key = json.dumps(sorted(data.items()), ensure_ascii=False)
    if cache_key in _risques_cache:
        return _risques_cache[cache_key]
    risques, _, _ = analyze_all(data)
    _risques_cache[cache_key] = risques
    return risques

def generate_audit_commentaires(data: Dict) -> List[str]:
    cache_key = json.dumps(sorted(data.items()), ensure_ascii=False)
    if cache_key in _commentaires_cache:
        return _commentaires_cache[cache_key]
    _, commentaires, _ = analyze_all(data)
    _commentaires_cache[cache_key] = commentaires
    return commentaires

def generate_summary(data: Dict) -> str:
    cache_key = json.dumps(sorted(data.items()), ensure_ascii=False)
    if cache_key in _resume_cache:
        return _resume_cache[cache_key]
    _, _, resume = analyze_all(data)
    _resume_cache[cache_key] = resume
    return resume