from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

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

    response = model.generate_content(prompt)

    return response.text