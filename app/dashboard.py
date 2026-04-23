import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MediRisk Portal", layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("app.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    age INT,
    glucose INT,
    cholesterol INT,
    bp INT,
    probability REAL,
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/realtime_patient_data.csv")

    if "Risk_Level" not in df.columns:
        df["Risk_Level"] = df["Risk_Score"].apply(lambda x: 1 if x > 50 else 0)

    return df

df = load_data()

# ---------------- MODEL ----------------
@st.cache_resource
def train_model(df):
    X = df[['Age','Systolic_BP','Glucose_Lvl','Cholesterol_Lvl']]
    y = df['Risk_Level']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model(df)

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- AUTH ----------------
def login(u,p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
    return c.fetchone()

def signup(u,p):
    try:
        c.execute("INSERT INTO users VALUES (?,?)",(u,p))
        conn.commit()
        return True
    except:
        return False

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧑‍⚕️ MediRisk Portal")

menu = st.sidebar.selectbox("Menu", ["Login","Signup"])

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user}")
    page = st.sidebar.radio("Navigate",
        ["Patient Diagnostic","Analytics","Model Transparency","History"])

    if st.sidebar.button("Logout"):
        st.session_state.user = None

# ---------------- LOGIN ----------------
if not st.session_state.user:

    if menu == "Login":
        st.title("🔐 Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(u,p):
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    else:
        st.title("🆕 Signup")
        u = st.text_input("Create Username")
        p = st.text_input("Create Password")

        if st.button("Signup"):
            if signup(u,p):
                st.success("Account created! Go to login")
            else:
                st.error("User already exists")

# ---------------- MAIN APP ----------------
else:

    # ================= PATIENT PAGE =================
    if page == "Patient Diagnostic":

        st.title("🧑‍⚕️ Patient Risk Assessment")

        col1, col2 = st.columns([1,2])

        with col1:
            age = st.slider("Age",18,100,45)
            bp = st.slider("Systolic BP",90,200,120)
            glucose = st.slider("Glucose",70,300,110)
            chol = st.slider("Cholesterol",150,300,200)

        # FIXED (DataFrame → no warning)
        input_df = pd.DataFrame([[age,bp,glucose,chol]],
            columns=['Age','Systolic_BP','Glucose_Lvl','Cholesterol_Lvl'])

        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]*100
        result = "High Risk" if prob>50 else "Low Risk"

        with col2:
            st.metric("Risk Score", f"{prob:.2f}%")

            if prob > 50:
                st.error("High Risk Detected 🚨")
            else:
                st.success("Low Risk ✅")

            fig, ax = plt.subplots()
            avg = df.groupby("Age")["Risk_Score"].mean()
            ax.plot(avg.index, avg.values)
            ax.scatter(age, prob, color="red", s=200)
            ax.set_title("Your Health vs Population")
            st.pyplot(fig)

        # ---------- PATIENT EXPLANATION ----------
        st.divider()
        st.subheader("🧠 Easy Explanation")

        if prob < 30:
            st.success("You're in a healthy range. Keep it up!")
        elif prob < 60:
            st.warning("Moderate risk. Improving glucose or BP will help.")
        else:
            st.error("High risk. Please consult a doctor.")

        # ---------- SUGGESTIONS ----------
        st.subheader("💡 Suggestions")

        if glucose > 140:
            st.write("• Reduce sugar intake")
        if bp > 140:
            st.write("• Reduce salt & manage stress")
        if chol > 240:
            st.write("• Avoid oily foods")

        # ---------- SAVE ----------
        if st.button("Save to History"):
            c.execute("""INSERT INTO predictions
            (username,age,glucose,cholesterol,bp,probability,result)
            VALUES (?,?,?,?,?,?,?)""",
            (st.session_state.user,age,glucose,chol,bp,prob,result))
            conn.commit()
            st.success("Saved!")

    # ================= ANALYTICS =================
    elif page == "Analytics":

        st.title("📊 Health Analytics Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Avg Risk Score", f"{df['Risk_Score'].mean():.1f}")
            st.metric("Avg Glucose", f"{df['Glucose_Lvl'].mean():.1f}")

        with col2:
            st.metric("Avg BP", f"{df['Systolic_BP'].mean():.1f}")
            st.metric("Avg Cholesterol", f"{df['Cholesterol_Lvl'].mean():.1f}")

        st.divider()

        st.subheader("🔥 Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(
            df[['Age','Systolic_BP','Glucose_Lvl','Cholesterol_Lvl','Risk_Score']].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)

        st.info("Darker color = stronger impact on risk")

        st.divider()

        st.subheader("📈 Glucose vs Risk")
        st.scatter_chart(df.sample(500), x="Glucose_Lvl", y="Risk_Score")

        st.subheader("📊 Risk vs Age")
        age_risk = df.groupby("Age")["Risk_Score"].mean()
        st.line_chart(age_risk)

    # ================= MODEL =================
    elif page == "Model Transparency":

        st.title("🧠 Model Explanation")

        importance = model.feature_importances_
        features = ['Age','BP','Glucose','Cholesterol']

        fig, ax = plt.subplots()
        ax.bar(features, importance)
        st.pyplot(fig)

        st.info("Higher bar = more impact on health risk")

    # ================= HISTORY =================
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