#!/usr/bin/env python3
import os
from openai import OpenAI

# Make sure your API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not set.")
    exit(1)

client = OpenAI(api_key=api_key)

print("=== ChatGPT Terminal ===")
print("Type 'exit' or 'quit' to end the session.\n")

conversation_history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo"
            messages=conversation_history,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        print(f"ChatGPT: {answer}\n")

        conversation_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        print(f"Error: {e}")
