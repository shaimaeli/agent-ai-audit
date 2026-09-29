import os
import uuid
from werkzeug.utils import secure_filename

# Import LOCAL uniquement — pas de circular dependency
from app.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS


def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_unique_filename(filename):
    """Génère un nom de fichier unique."""
    ext = filename.rsplit(".", 1)[1].lower()
    unique_id = str(uuid.uuid4())[:8]
    return f"{unique_id}_{secure_filename(filename)}"


def save_uploaded_file(file):
    """Sauvegarde un fichier uploadé et retourne son chemin."""
    if file and allowed_file(file.filename):
        filename = generate_unique_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return filepath
    return None


def get_file_extension(filename):
    """Retourne l'extension d'un fichier."""
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""