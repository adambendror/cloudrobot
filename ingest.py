import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Paths
RAG_FOLDER = "rag"
CHROMA_DIR = "chroma_db"

# Read all .txt files in rag/
texts = []
for filename in os.listdir(RAG_FOLDER):
    if filename.endswith(".txt"):
        with open(os.path.join(RAG_FOLDER, filename), "r", encoding="utf-8") as f:
            texts.append(f.read())

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create or update Chroma DB
vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

vectordb.persist()
print(f"Chroma DB updated with {len(texts)} chunks.")
