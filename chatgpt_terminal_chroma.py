#!/usr/bin/env python3
import os
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Make sure your API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not set.")
    exit(1)

client = OpenAI(api_key=api_key)

# Initialize Chroma DB
CHROMA_DIR = "chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

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
        print(f"Error querying Chroma DB: {e}")
        context = ""

    # Add context to the conversation history
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
