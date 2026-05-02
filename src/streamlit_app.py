import streamlit as st
import pickle
import numpy as np

# -------------------------------
# LOAD MODEL
# -------------------------------
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# -------------------------------
# INPUT FIELDS
# -------------------------------
age = st.number_input("Age", min_value=0, max_value=120, value=30)
cholesterol = st.number_input("Cholesterol Level", value=200)
glucose = st.number_input("Glucose Level", value=100)

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("Predict Risk"):

    features = np.array([[age, cholesterol, glucose]])
    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk ({probability:.2f})")
    else:
        st.success(f"✅ Low Risk ({probability:.2f})")