from app.genai_helper import generate_chat_response

response = generate_chat_response(
    "What are symptoms of diabetes?"
)

print(response)