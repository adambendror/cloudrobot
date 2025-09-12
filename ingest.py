import os
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_openai import OpenAIEmbeddings  # Updated import for OpenAIEmbeddings

# Paths
RAG_FOLDER = "rag"
CHROMA_DIR = "chroma_db"

# Read all .txt files in rag/
texts = []
for filename in os.listdir(RAG_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(RAG_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"Reading file: {filepath}")
            print(f"Content: {content[:100]}...")  # Print the first 100 characters
            texts.append(content)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create or update Chroma DB
vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name="weather_data"  # Explicitly name the collection
)

print(f"Chroma DB updated with {len(texts)} chunks.")
