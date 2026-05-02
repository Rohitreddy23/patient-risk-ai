from flask import Flask, request, jsonify
import pickle
import numpy as np

print("🚀 Starting Flask App...")

app = Flask(__name__)

model = pickle.load(open("../models/model.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))

@app.route("/")
def home():
    return "API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    age = float(data['age'])
    cholesterol = float(data['cholesterol'])
    glucose = float(data['glucose'])

    features = np.array([[age, cholesterol, glucose]])
    features = scaler.transform(features)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    result = "High Risk" if pred == 1 else "Low Risk"

    return jsonify({
        "prediction": result,
        "probability": float(prob)
    })

if __name__ == "__main__":
    print("🔥 Running server...")
    app.run(debug=True)