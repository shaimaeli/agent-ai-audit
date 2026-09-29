"""
Test isolé de langchain_rag.py
Placez ce fichier dans app/utils/ ou à la racine
"""
import os
import sys

# Détecter si on est dans app/utils/ ou à la racine
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "utils":
    # On est dans app/utils/
    app_dir = os.path.dirname(script_dir)
    base_dir = os.path.dirname(app_dir)
else:
    # On est à la racine
    base_dir = script_dir
    app_dir = os.path.join(base_dir, "app")

sys.path.insert(0, app_dir)

print("=" * 60)
print("TEST ISOLÉ DE LANGCHAIN_RAG.PY")
print("=" * 60)
print(f"Script dans : {script_dir}")
print(f"BASE_DIR détecté : {base_dir}")

# Test 1 : Vérifier les chemins
print("\n📁 TEST 1 : Chemins")

DOCS_STAGE_FOLDER = os.path.join(base_dir, "docs_stage")
CHROMA_DB_PATH = os.path.join(base_dir, "chroma_db")
MODEL_PATH = os.path.join(base_dir, "models", "paraphrase-multilingual-MiniLM-L12-v2")

print(f"\ndocs_stage = {DOCS_STAGE_FOLDER}")
print(f"docs_stage existe ? {os.path.exists(DOCS_STAGE_FOLDER)}")
if os.path.exists(DOCS_STAGE_FOLDER):
    files = os.listdir(DOCS_STAGE_FOLDER)
    pdfs = [f for f in files if f.lower().endswith(".pdf")]
    print(f"Fichiers dans docs_stage : {len(files)}")
    print(f"PDFs trouvés : {len(pdfs)}")
    for pdf in pdfs:
        print(f"  - {pdf}")

print(f"\nchroma_db = {CHROMA_DB_PATH}")
print(f"chroma_db existe ? {os.path.exists(CHROMA_DB_PATH)}")

print(f"\nmodels = {MODEL_PATH}")
print(f"models existe ? {os.path.exists(MODEL_PATH)}")

# Test 2 : Vérifier le modèle d\'embeddings
print("\n" + "=" * 60)
print("🤖 TEST 2 : Modèle d\'embeddings")
print("=" * 60)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ HuggingFaceEmbeddings importé")

    if os.path.exists(MODEL_PATH):
        print(f"\nChargement du modèle local : {MODEL_PATH}")
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Modèle chargé avec succès")

        # Test d\'embedding
        test_text = "Ceci est un test"
        result = embeddings.embed_query(test_text)
        print(f"✅ Test d\'embedding réussi (dimension : {len(result)})")
    else:
        print(f"❌ Modèle non trouvé : {MODEL_PATH}")
        print("   Exécutez d\'abord : python download_model.py")

except Exception as e:
    print(f"❌ Erreur embeddings : {e}")
    import traceback
    traceback.print_exc()

# Test 3 : Vérifier ChromaDB
print("\n" + "=" * 60)
print("💾 TEST 3 : ChromaDB")
print("=" * 60)

try:
    from langchain_chroma import Chroma
    print("✅ Chroma importé")

    # Test création d\'une collection
    test_persist = os.path.join(CHROMA_DB_PATH, "test")
    os.makedirs(test_persist, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        test_vs = Chroma(
            collection_name="test",
            embedding_function=embeddings,
            persist_directory=test_persist
        )
        print("✅ ChromaDB initialisé")

        # Test ajout de documents
        from langchain_core.documents import Document
        test_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.pdf"}),
            Document(page_content="Test document 2", metadata={"source": "test2.pdf"})
        ]
        test_vs.add_documents(test_docs)
        print("✅ Documents ajoutés")

        # Test recherche
        results = test_vs.similarity_search("test", k=1)
        print(f"✅ Recherche réussie ({len(results)} résultats)")

        # Nettoyage
        import shutil
        shutil.rmtree(test_persist)
        print("✅ Test nettoyé")
    else:
        print("⚠️ Test ChromaDB sauté (modèle non disponible)")

except Exception as e:
    print(f"❌ Erreur ChromaDB : {e}")
    import traceback
    traceback.print_exc()

# Test 4 : Test complet de l\'indexation
print("\n" + "=" * 60)
print("📚 TEST 4 : Indexation complète")
print("=" * 60)

try:
    # Importer depuis app/utils/
    sys.path.insert(0, os.path.join(app_dir, "utils"))
    from langchain_rag import index_all_docs, list_indexed_docs
    print("✅ langchain_rag importé")

    if os.path.exists(DOCS_STAGE_FOLDER) and os.path.exists(MODEL_PATH):
        print("\nLancement de l\'indexation...")
        result = index_all_docs()
        print(f"\n✅ Indexation terminée : {len(result)} fichiers indexés")

        print("\nListe des documents indexés :")
        indexed = list_indexed_docs()
        for doc_type, info in indexed.items():
            print(f"  • {doc_type} : {info['chunks']} chunks, {len(info['sources'])} sources")
    else:
        print("⚠️ Test indexation sauté (docs_stage ou modèle non disponible)")

except Exception as e:
    print(f"❌ Erreur indexation : {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST TERMINÉ")
print("=" * 60)