#!/usr/bin/env python3
import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# Make sure your API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not set.")
    exit(1)

client = OpenAI(api_key=api_key)

# Initialize Chroma client and load the database
chroma_client = chromadb.Client(Settings(
    persist_directory="/chroma_db",  # Path to your Chroma database
    chroma_db_impl="duckdb+parquet"
))
collection = chroma_client.get_collection("weather_data")  # Replace with your collection name

print("=== Cloud Terminal ===")
print("Type 'exit' or 'quit' to end the session.\n")

# Add a system prompt to define the assistant's behavior
conversation_history = [
    {"role": "system", "content": "You are Cloud, a helpful and friendly assistant for weather-related queries."}
]

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Query the Chroma database for relevant context
    try:
        results = collection.query(
            query_texts=[user_input],
            n_results=3  # Number of relevant results to retrieve
        )
        context = "\n".join(results["documents"])  # Combine the retrieved documents
    except Exception as e:
        context = ""
        print(f"Error querying Chroma database: {e}")

    # Add the retrieved context to the conversation history
    if context:
        conversation_history.append({"role": "system", "content": f"Relevant context: {context}"})

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
