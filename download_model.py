from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "paraphrase-multilingual-MiniLM-L12-v2")

print(f"Téléchargement de {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print(f"Sauvegarde dans {SAVE_PATH}...")
model.save(SAVE_PATH)
print("Modèle téléchargé et sauvegardé !")