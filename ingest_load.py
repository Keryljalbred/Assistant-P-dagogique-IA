from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os

# 1. Charger tous les fichiers depuis /docs
loader = DirectoryLoader(
    path="docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()
print(f"{len(documents)} documents chargés")

# 2. Splitter les documents en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

print(f"{len(docs)} chunks générés.")

# 3. Sauvegarder les chunks
os.makedirs("docs", exist_ok=True)
with open("docs/chunks.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅ chunks.pkl sauvegardé dans /docs")
