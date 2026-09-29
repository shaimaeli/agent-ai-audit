import os

# Clés API
GROQ_API_KEY = os.environ.get("API_KEY")  
MODEL_GPT = "llama-3.3-70b-versatile"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
DOCS_STAGE_FOLDER = os.path.join(BASE_DIR, "docs_stage")
TEMPLATES_WORD_FOLDER = os.path.join(BASE_DIR, "templates_word")
TEMPLATES_HTML_FOLDER = os.path.join(BASE_DIR, "app", "templates")
CACHE_FOLDER = os.path.join(BASE_DIR, "cache")  # ← AJOUTE CECI

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, DOCS_STAGE_FOLDER, TEMPLATES_WORD_FOLDER, TEMPLATES_HTML_FOLDER, CACHE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

LLM_CACHE_FILE = os.path.join(CACHE_FOLDER, "llm_cache.json")

SECRET_KEY = "ta_clé_secrète_ici_change_la"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 Mo max
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "xlsx", "xls", "txt"}