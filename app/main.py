import os
from flask import Flask, render_template, request, send_from_directory, jsonify, session
from app.config import UPLOAD_FOLDER, OUTPUT_FOLDER, SECRET_KEY, MAX_CONTENT_LENGTH
from app.utils.db import init_db, save_entreprise, save_document, get_entreprise_by_ice
from app.utils.helpers import save_uploaded_file, allowed_file
from app.utils.data_extractor import extract_data_from_files
from app.utils.langchain_rag import index_all_docs, list_indexed_docs
from app.utils.document_generator import (
    generate_fiche_signaletique,
    generate_acceptation_mission,
    generate_maintien_mission,
    generate_questionnaire_pri,
    generate_questionnaire_inventaire,
    generate_questionnaire_verification,
    generate_questionnaire_evenement,
    generate_questionnaire_fin,
)
from app.utils.pdf_converter import pdf_to_word, pdf_to_excel
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
init_db()

@app.route("/")
def index():
    """Page d'accueil."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Reçoit les fichiers uploadés et extrait les données entreprise.
    Retourne les données structurées au frontend.
    """
    if "files" not in request.files:
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Aucun fichier valide sélectionné"}), 400

    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filepath = save_uploaded_file(file)
            if filepath:
                file_paths.append(filepath)

    if not file_paths:
        return jsonify({"error": "Aucun fichier valide"}), 400

    try:
        data = extract_data_from_files(file_paths)

        print("TOUTES LES CLÉS EXTRAITES :")
        for k, v in data.items():
            if v and str(v).strip():
                print(f"   '{k}' = '{str(v)[:50]}'")

        if not data:
            return jsonify({"error": "Aucune donnée extraite"}), 400

        print("CLÉS APRÈS EXTRACTION :", list(data.keys()))

        entreprise_id = save_entreprise(data)
        data["entreprise_id"] = entreprise_id

        data["dirigeants_str"] = ", ".join(data.get("dirigeants", [])) if isinstance(data.get("dirigeants"), list) else data.get("dirigeants", "")
        data["banques_str"]    = ", ".join(data.get("banques", []))    if isinstance(data.get("banques"), list)    else data.get("banques", "")

        session["bilan_ia"]     = data.get("bilan_ia")
        session["bilan"]        = data.get("bilan", [])
        session["bilan_passif"] = data.get("bilan_passif", [])
        session["total_actif"]  = data.get("total_actif", 0)

        return jsonify({"success": True, "data": data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erreur lors du traitement : {str(e)}"}), 500


VALID_TYPES = {
    "fiche_signaletique",
    "acceptation_mission",
    "maintien_mission",
    "questionnaire_pri",
    "questionnaire_inventaire",
    "questionnaire_verification",
    "questionnaire_evenement",
    "questionnaire_fin",
}

VALID_FORMATS = {"pdf", "word", "excel"}

PDF_GENERATORS = {
    "fiche_signaletique":         generate_fiche_signaletique,
    "acceptation_mission":        generate_acceptation_mission,
    "maintien_mission":           generate_maintien_mission,
    "questionnaire_pri":          generate_questionnaire_pri,
    "questionnaire_inventaire":   generate_questionnaire_inventaire,
    "questionnaire_verification": generate_questionnaire_verification,
    "questionnaire_evenement":    generate_questionnaire_evenement,
    "questionnaire_fin":          generate_questionnaire_fin,
}


@app.route("/generate/<doc_type>", methods=["POST"])
def generate_document(doc_type):
    """Route legacy — PDF par défaut (pour compatibilité)."""
    return generate_document_with_format(doc_type, "pdf")


@app.route("/generate/<doc_type>/<format>", methods=["POST"])
def generate_document_with_format(doc_type, format):
    """
    Génère un document dans le format choisi par l'utilisateur.
    
    Process :
        1. Génère TOUJOURS le PDF rempli d'abord (via document_generator.py)
        2. Si Word/Excel demandé → convertit le PDF généré
    
    Formats disponibles :
        - pdf   : Document PDF fidèle au template (archivage officiel)
        - word  : Conversion du PDF en .docx éditable
        - excel : Conversion du PDF en .xlsx (données structurées)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Aucune donnée fournie"}), 400

    print(f"\n📄 Demande génération : {doc_type} en format '{format}'")
    print("CLÉS REÇUES :", list(data.keys()))

    if doc_type not in VALID_TYPES:
        return jsonify({
            "error": f"Type de document inconnu : '{doc_type}'",
            "valid_types": sorted(VALID_TYPES)
        }), 400

    if format not in VALID_FORMATS:
        return jsonify({
            "error": f"Format invalide : '{format}'",
            "valid_formats": sorted(VALID_FORMATS),
            "message": "Choisissez : pdf, word, ou excel"
        }), 400

    if "bilan_ia" not in data or not data.get("bilan_ia"):
        if session.get("bilan_ia"):
            data["bilan_ia"]     = session["bilan_ia"]
            data["bilan"]        = session.get("bilan", [])
            data["bilan_passif"] = session.get("bilan_passif", [])
            data["total_actif"]  = session.get("total_actif", 0)

    try:
       
        generator = PDF_GENERATORS.get(doc_type)
        if not generator:
            return jsonify({"error": f"Générateur PDF non trouvé pour '{doc_type}'"}), 500
        
        pdf_path = generator(data)
        
        if pdf_path is None:
            return jsonify({
                "error": f"Échec génération PDF pour '{doc_type}'",
                "detail": "Vérifiez que les templates sont dans docs_stage/"
            }), 500
        
        print(f"PDF de base généré : {os.path.basename(pdf_path)}")

       
        if format == "pdf":
            output_path = pdf_path
        
        elif format == "word":
            output_path = pdf_to_word(doc_type, data)
        
        elif format == "excel":
            output_path = pdf_to_excel(doc_type, data)

        if "entreprise_id" in data:
            save_document(data["entreprise_id"], f"{doc_type}_{format}", output_path)

        ext = os.path.splitext(output_path)[1].lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }

        return jsonify({
            "success":   True,
            "path":      output_path,
            "filename":  os.path.basename(output_path),
            "format":    format,
            "mime_type": mime_types.get(ext, "application/octet-stream"),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erreur génération : {str(e)}"}), 500


@app.route("/download/<filename>")
def download_file(filename):
    """Télécharge un fichier généré (PDF, Word ou Excel)."""
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
        as_attachment=True,
        mimetype=mime_types.get(ext)
    )


@app.route("/entreprise/<ice>")
def get_entreprise(ice):
    """Récupère une entreprise par son ICE."""
    entreprise = get_entreprise_by_ice(ice)
    if entreprise:
        return jsonify({"success": True, "data": entreprise})
    return jsonify({"error": "Entreprise non trouvée"}), 404


@app.route("/admin/index", methods=["POST"])
def admin_index():
    """Indexe tous les documents dans ChromaDB."""
    try:
        result  = index_all_docs()
        indexed = list_indexed_docs()
        return jsonify({
            "success":       True,
            "files_indexed": result,
            "total_indexed": indexed
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/indexed")
def admin_indexed():
    """Liste les documents indexés."""
    try:
        indexed = list_indexed_docs()
        return jsonify({"success": True, "data": indexed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/formats")
def list_formats():
    """Retourne les formats de génération disponibles."""
    return jsonify({
        "success": True,
        "formats": [
            {"id": "pdf",   "name": "PDF",   "ext": ".pdf",  "editable": False, "description": "Document PDF fidèle au template (archivage officiel)"},
            {"id": "word",  "name": "Word",  "ext": ".docx", "editable": True,  "description": "Document Word éditable converti depuis le PDF"},
            {"id": "excel", "name": "Excel", "ext": ".xlsx", "editable": True,  "description": "Tableur Excel avec les données extraites du PDF"},
        ]
    })


@app.route("/types")
def list_document_types():
    """Retourne les types de documents disponibles."""
    return jsonify({
        "success": True,
        "types": {
            "fiche_signaletique":         {"name": "Fiche signalétique",         "category": "standard"},
            "acceptation_mission":        {"name": "Acceptation de mission",     "category": "standard"},
            "maintien_mission":           {"name": "Maintien de mission",        "category": "standard"},
            "questionnaire_pri":          {"name": "Guide PRI",                  "category": "questionnaire"},
            "questionnaire_inventaire":   {"name": "Questionnaire inventaire",   "category": "questionnaire"},
            "questionnaire_verification": {"name": "Vérifications spécifiques",  "category": "questionnaire"},
            "questionnaire_evenement":    {"name": "Événements post-clôture",    "category": "questionnaire"},
            "questionnaire_fin":          {"name": "Fin de mission",             "category": "questionnaire"},
        }
    })