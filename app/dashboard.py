import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MediRisk Portal", layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("app.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    age INTEGER,
    bp INTEGER,
    glucose INTEGER,
    cholesterol INTEGER,
    probability REAL,
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/realtime_patient_data.csv")

df = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    features = ['Age', 'Systolic_BP', 'Glucose_Lvl', 'Cholesterol_Lvl']
    
    X = df[features]
    y = (df['Risk_Score'] > 50).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model(df)

# ---------------- LOGIN SYSTEM ----------------
if "user" not in st.session_state:
    st.session_state.user = None

st.sidebar.title("🧑‍⚕️ MediRisk Portal")

menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])

if st.session_state.user is None:
    if menu == "Login":
        user = st.sidebar.text_input("Username")
        if st.sidebar.button("Login"):
            st.session_state.user = user
            st.success("Logged in successfully")

    elif menu == "Signup":
        new_user = st.sidebar.text_input("Create Username")
        if st.sidebar.button("Signup"):
            st.session_state.user = new_user
            st.success("Account created!")

else:
    st.sidebar.success(f"Logged in as {st.session_state.user}")

    if st.sidebar.button("Logout"):
        st.session_state.user = None

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio("Navigate", 
    ["Patient Diagnostic", "Analytics", "Model Transparency", "History"]
)

# ============================================================
# 🏥 PATIENT DIAGNOSTIC
# ============================================================
if page == "Patient Diagnostic":

    st.title("👨‍⚕️ Patient Risk Assessment")

    col1, col2 = st.columns([1,2])

    with col1:
        age = st.slider("Age", 18, 90, 45)
        bp = st.slider("Systolic BP", 90, 200, 120)
        glucose = st.slider("Glucose", 70, 300, 110)
        cholesterol = st.slider("Cholesterol", 150, 300, 200)

    # Prediction
    input_df = pd.DataFrame([[age, bp, glucose, cholesterol]],
                            columns=['Age','Systolic_BP','Glucose_Lvl','Cholesterol_Lvl'])

    input_scaled = scaler.transform(input_df)
    risk_prob = model.predict_proba(input_scaled)[0][1] * 100

    # Fix 0% issue
    risk_prob = max(5, risk_prob)

    with col2:
        st.metric("Risk Score", f"{risk_prob:.2f}%")

        if risk_prob < 30:
            st.success("Low Risk ✅")
        elif risk_prob < 60:
            st.warning("Moderate Risk ⚠️")
        else:
            st.error("High Risk 🚨")

        # Graph
        fig, ax = plt.subplots()
        age_avg = df.groupby('Age')['Risk_Score'].mean()
        ax.plot(age_avg.index, age_avg.values)
        ax.scatter(age, risk_prob, color='red', s=200)
        ax.set_title("Your Health vs Population")
        st.pyplot(fig)

    # ---------------- Explanation ----------------
    st.subheader("🧠 Easy Explanation")

    if risk_prob < 30:
        st.success("You are in a healthy range. Keep it up!")
    elif risk_prob < 60:
        st.warning("Your risk is slightly above normal. Lifestyle changes can help.")
    else:
        st.error("Your risk is high. Immediate attention recommended.")

    # ---------------- Suggestions ----------------
    st.subheader("💡 Suggestions")

    suggestions = []

    if glucose > 140:
        suggestions.append("Reduce sugar intake (avoid sweets, soda)")

    if bp > 140:
        suggestions.append("Reduce salt intake & manage stress")

    if cholesterol > 240:
        suggestions.append("Avoid fried foods & increase exercise")

    if age > 60:
        suggestions.append("Regular health checkups recommended")

    if suggestions:
        for s in suggestions:
            st.info("👉 " + s)
    else:
        st.success("👍 Maintain your healthy lifestyle!")

    # ---------------- Save History ----------------
    if st.button("Save to History"):
        cursor.execute("""
        INSERT INTO predictions (username, age, bp, glucose, cholesterol, probability, result)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (st.session_state.user, age, bp, glucose, cholesterol, risk_prob,
              "High Risk" if risk_prob > 60 else "Low Risk"))
        conn.commit()
        st.success("Saved successfully!")

# ============================================================
# 📊 ANALYTICS
# ============================================================
elif page == "Analytics":

    st.title("📊 Health Analytics Dashboard")

    col1, col2 = st.columns(2)

    col1.metric("Avg Risk", round(df['Risk_Score'].mean(),2))
    col2.metric("Avg BP", round(df['Systolic_BP'].mean(),2))

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")

    corr = df[['Age','Systolic_BP','Glucose_Lvl','Cholesterol_Lvl','Risk_Score']].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.info("Darker color = stronger relationship with risk")

    # Scatter
    st.subheader("📈 Glucose vs Risk")
    st.scatter_chart(df, x="Glucose_Lvl", y="Risk_Score")

# ============================================================
# 🧠 MODEL TRANSPARENCY
# ============================================================
elif page == "Model Transparency":

    st.title("🧠 Model Transparency")

    importances = model.feature_importances_
    features = ['Age','BP','Glucose','Cholesterol']

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df, x="Feature", y="Importance")

    st.info("Higher value = stronger impact on risk")

# ============================================================
# 📜 HISTORY
# ============================================================
elif page == "History":

    st.title("📜 Patient History")

    data = pd.read_sql_query(
        f"SELECT * FROM predictions WHERE username='{st.session_state.user}'",
        conn
    )

    if data.empty:
        st.warning("No history found")
    else:
        st.dataframe(data)