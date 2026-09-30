from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "paraphrase-multilingual-MiniLM-L12-v2")

model = SentenceTransformer(MODEL_NAME)
model.save(SAVE_PATH)