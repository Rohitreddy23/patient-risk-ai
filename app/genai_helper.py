import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def generate_chat_response(prompt):

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=60
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def generate_medical_report(data, risk):

    prompt = f"""
    Patient Details:
    Pregnancies: {data['Pregnancies']}
    Glucose: {data['Glucose']}
    BloodPressure: {data['BloodPressure']}
    BMI: {data['BMI']}
    Age: {data['Age']}

    Predicted Risk Level: {risk}

    Explain:
    1. Patient health condition
    2. Possible risks
    3. Lifestyle suggestions
    4. Preventive measures

    Keep response simple and professional.
    """

    return generate_chat_response(prompt)