#!/usr/bin/env python3
import os
import openai

# Make sure you have set your API key in the environment:
# export OPENAI_API_KEY="sk-xxxxxx"

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OPENAI_API_KEY not set. Please run:")
    print("export OPENAI_API_KEY='your_key_here'")
    exit(1)

openai.api_key = openai_api_key

print("=== ChatGPT Terminal ===")
print("Type 'exit' or 'quit' to end the session.\n")

# Optional: remember conversation for context
conversation_history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Append user message
    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo"
            messages=conversation_history,
            temperature=0.7
        )
        answer = response.choices[0].message['content'].strip()
        print(f"ChatGPT: {answer}\n")

        # Append assistant message to history
        conversation_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        print(f"Error: {e}")
