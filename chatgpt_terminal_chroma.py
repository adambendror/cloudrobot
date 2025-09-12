#!/usr/bin/env python3
import os
import logging
from openai import OpenAI
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_openai import OpenAIEmbeddings  # Updated import for OpenAIEmbeddings

# Initialize logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Make sure your API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not set.")
    exit(1)

client = OpenAI(api_key=api_key)

# Initialize Chroma DB
CHROMA_DIR = "chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name="weather_data")  # Ensure collection name matches

# Inspect ChromaDB contents
try:
    all_docs = vectordb.similarity_search("", k=10)  # Retrieve up to 10 documents
    print("\n=== ChromaDB Contents ===")
    if all_docs:
        for i, doc in enumerate(all_docs):
            print(f"Document {i + 1}: {doc.page_content}\n")
    else:
        print("ChromaDB is empty. No documents found.\n")
except Exception as e:
    print(f"Error inspecting ChromaDB: {e}")

print("=== Cloud Terminal ===")
print("Type 'exit' or 'quit' to end the session.\n")

conversation_history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Query Chroma DB for relevant context
    try:
        results = vectordb.similarity_search(user_input, k=3)  # Retrieve top 3 relevant chunks
        context = "\n".join([result.page_content for result in results])
    except Exception as e:
        context = ""

    # Add limited context to the conversation history
    if context:
        limited_context = context.split("\n")[0]  # Use only the first chunk or line
        conversation_history.append({"role": "system", "content": f"Relevant context: {limited_context}"})

    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo"
            messages=conversation_history,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        print(f"Cloud: {answer}\n")

        conversation_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        print(f"Error: {e}")
