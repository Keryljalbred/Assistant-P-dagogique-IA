import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pickle

# Chargement des chunks produits par l'étape d'ingestion
CHUNKS_PATH = "docs/chunks.pkl"
INDEX_PATH = "index/faiss_index"

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(f"Fichier {CHUNKS_PATH} introuvable. Lance d'abord ingestion_loader.py.")

# Chargement des chunks
with open(CHUNKS_PATH, "rb") as f:
    documents = pickle.load(f)

print(f"Nombre de chunks à indexer : {len(documents)}")

# Embedding avec Sentence-BERT (MiniLM)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Création de l'index FAISS
db = FAISS.from_documents(documents, embedding_model)

# Sauvegarde de l'index
db.save_local(INDEX_PATH)
print(f"Index FAISS sauvegardé dans : {INDEX_PATH}")
