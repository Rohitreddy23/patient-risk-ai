import shap

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voice_helper import speech_to_text, text_to_speech
# from rag_helper import process_pdf, ask_pdf_question
from genai_helper import generate_medical_report
from dotenv import load_dotenv
load_dotenv()

from genai_helper import generate_medical_report, generate_chat_response

from src.utils import get_risk_level, get_suggestions
from database.db import *


from pdf_helper import create_pdf_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

# ======================
# DB
# ======================
create_tables()
conn = sqlite3.connect("app.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT, role TEXT)""")
c.execute("""CREATE TABLE IF NOT EXISTS history(username TEXT, age INT, bp INT, glucose INT, cholesterol INT, risk TEXT, score REAL)""")
conn.commit()

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def signup(u,p,r):
    try:
        c.execute("INSERT INTO users VALUES (?,?,?)",(u,hash_pw(p),r))
        conn.commit()
        return True
    except:
        return False

def login(u,p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",(u,hash_pw(p)))
    return c.fetchone()

def save_hist(u,a,b,g,ch,r,s):
    c.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)",(u,a,b,g,ch,r,s))
    conn.commit()

def get_hist(u):
    c.execute("SELECT * FROM history WHERE username=?",(u,))
    return c.fetchall()

def get_all():
    c.execute("SELECT * FROM history")
    return c.fetchall()

# ======================
# DATA
# ======================
@st.cache_data
def load_data():
    base=os.path.dirname(os.path.abspath(__file__))
    path=os.path.join(base,"..","data","realtime_patient_data.csv")
    return pd.read_csv(path)

# ======================
# ML (cached)
# ======================
@st.cache_resource
def train_models(data):

    X=data[["Age","Systolic_BP","Glucose_Lvl","Cholesterol_Lvl"]]
    y=(data["Risk_Score"]>50).astype(int)

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)

    models={
        "Logistic":LogisticRegression(max_iter=1000),
        "Decision Tree":DecisionTreeClassifier(),
        "Random Forest":RandomForestClassifier(),
        "Gradient Boosting":GradientBoostingClassifier()
    }

    res=[]
    outputs={}

    for name,m in models.items():
        m.fit(Xtr,ytr)
        yp=m.predict(Xte)
        ypb=m.predict_proba(Xte)[:,1]

        res.append({
            "Model":name,
            "Accuracy":accuracy_score(yte,yp),
            "Precision":precision_score(yte,yp,zero_division=0),
            "Recall":recall_score(yte,yp,zero_division=0),
            "F1":f1_score(yte,yp,zero_division=0),
            "AUC":roc_auc_score(yte,ypb)
        })

        outputs[name]=(m,Xte,yte,yp,ypb)

    df=pd.DataFrame(res)
    best_name=df.sort_values("Accuracy",ascending=False)["Model"].iloc[0]
    best_model=outputs[best_name][0]

    return df,outputs,best_model,best_name

# ======================
# RISK
# ======================
def risk_level(score):
    if score<40:return "Low"
    elif score<70:return "Medium"
    else:return "High"

def suggestions(level):
    if level=="High":
        return ["Consult doctor","Reduce sugar","Daily monitoring","Avoid oily food"]
    elif level=="Medium":
        return ["Exercise","Balanced diet","Reduce salt"]
    else:
        return ["Maintain lifestyle","Stay active"]

# ======================
# UI
# ======================
st.set_page_config(layout="wide")
st.sidebar.title("🧠 Patient Risk System")

if "user" not in st.session_state:
    st.session_state.user=None
    st.session_state.role=None

# LOGIN
if st.session_state.user is None:

    st.sidebar.subheader("Login / Signup")
    u=st.sidebar.text_input("Username")
    p=st.sidebar.text_input("Password",type="password")
    r=st.sidebar.selectbox("Role",["Patient","Admin"])

    if st.sidebar.button("Signup"):
        if signup(u,p,r): st.success("Account created")
        else: st.warning("User exists")

    if st.sidebar.button("Login"):
        user=login(u,p)
        if user:
            st.session_state.user=user[0]
            st.session_state.role=user[2]
            st.rerun()
        else:
            st.error("Invalid login")

# ======================
# MAIN
# ======================
if st.session_state.user:

    st.sidebar.success(f"{st.session_state.user} ({st.session_state.role})")

    menu=["Prediction","AI Chatbot","Voice Assistant","Model Eval","EDA","History"] if st.session_state.role=="Admin" else ["Prediction","AI Chatbot","Voice Assistant","History"]

    page=st.sidebar.radio("Menu",menu)

    if st.sidebar.button("Logout"):
        st.session_state.user=None
        st.session_state.role=None
        st.rerun()

    data=load_data()
    df,outputs,best_model,best_name=train_models(data)

    # ======================
    # PREDICTION
    # ======================
    if page=="Prediction":
        st.title("Patient Risk Prediction")
        
        age = st.slider("Age", 18, 100, 40)
        bp = st.slider("BP", 80, 200, 120)
        gl = st.slider("Glucose", 50, 300, 100)
        ch = st.slider("Cholesterol", 100, 400, 200)
        
        input_df = pd.DataFrame(
            [[age, bp, gl, ch]],
            columns=["Age", "Systolic_BP", "Glucose_Lvl", "Cholesterol_Lvl"]
            )
        
        if st.button("Predict Risk"):
            prob = best_model.predict_proba(input_df)[0][1] * 100
            level = risk_level(prob)
            
            save_hist(
                st.session_state.user,
                age,
                bp,
                gl,
                ch,
                level,
                prob
            )
            
            patient_data = {
                "Pregnancies": 0,
                "Glucose": gl,
                "BloodPressure": bp,
                "BMI": 0,
                "Age": age
                }
            
            with st.spinner("Generating AI medical report..."):
                report = generate_medical_report(patient_data, level)
                
                st.metric("Risk Score", f"{prob:.2f}%")
                st.success(f"Best Model: {best_name}")
                
                st.subheader("AI Medical Report")
                st.info(report)
                
                pdf_file = "patient_report.pdf"
                create_pdf_report(
                    pdf_file,
                    patient_data, 
                    prob,
                    level,
                    report,
                    suggestions(level)
                )
                
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "Download AI Report",
                        f,
                        file_name="AI_Healthcare_Report.pdf",
                        mime="application/pdf"
                    )
                
                st.warning(
                    "This AI-generated report is for educational purposes only and not a medical diagnosis."
                    )
                
                if level == "High":
                    st.error(level)
                elif level == "Medium":
                    st.warning(level)
                else:
                    st.success(level)
                    
                st.subheader("Suggestions")
                    
                for s in suggestions(level):
                    st.write("✔️", s)
                    
                
        # ======================
        # FEATURE IMPORTANCE
        # ======================

            st.subheader("Feature Importance")
        
        # ======================
        # SHAP EXPLAINABILITY
        # ======================

            
            try:
                features = [
                    "Age",
                    "Systolic_BP",
                    "Glucose_Lvl",
                    "Cholesterol_Lvl"
                ]
                
                if hasattr(best_model, "feature_importances_"):
                    importance = best_model.feature_importances_
                    
                elif hasattr(best_model, "coef_"):
                    importance = abs(best_model.coef_[0])
                    
                else:
                    importance = [0, 0, 0, 0]
                    
                shap_df = pd.DataFrame({
                    "Feature": features,
                    "Impact": importance
                })
                
                st.subheader("AI Explainability (SHAP)")
                
                st.dataframe(shap_df)
                
                fig, ax = plt.subplots(figsize=(8,4))
                
                ax.barh(
                    shap_df["Feature"],
                    shap_df["Impact"]
                )
                
                ax.set_xlabel("Impact")
                ax.set_title("Feature Importance")
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Explainability Error: {e}")
    # ======================
    # MODEL EVAL
    # ======================
    elif page=="Model Eval":

        st.title("Model Comparison")
        st.dataframe(df)
        st.bar_chart(df.set_index("Model"))

        # ROC
        st.subheader("ROC Curve")
        fig,ax=plt.subplots()

        for name,(m,Xt,yt,yp,ypb) in outputs.items():
            fpr,tpr,_=roc_curve(yt,ypb)
            ax.plot(fpr,tpr,label=name)

        ax.plot([0,1],[0,1],'--')
        ax.legend()
        st.pyplot(fig)

        # Confusion
        st.subheader("Confusion Matrix")
        m,Xt,yt,yp,ypb=outputs[best_name]

        cm=confusion_matrix(yt,yp)
        fig2,ax2=plt.subplots()
        sns.heatmap(cm,annot=True,fmt='d',ax=ax2)
        st.pyplot(fig2)

        st.text(classification_report(yt,yp))

    # ======================
    # EDA
    # ======================
    elif page=="EDA":

        st.title("EDA")
        fig,ax=plt.subplots()
        sns.heatmap(data.corr(numeric_only=True),annot=True,ax=ax)
        st.pyplot(fig)
        
    # ======================
    # AI CHATBOT
    # ======================
    elif page == "AI Chatbot":
        
        st.title("AI Medical Assistant")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        user_question = st.chat_input("Ask your health question")
            
        if user_question:
            
            # Store user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
                
            # Generate AI response
            with st.chat_message("assistant"):
                
                with st.spinner("Thinking..."):
                    conversation = ""
                    
                    for msg in st.session_state.messages:
                        conversation += f"{msg['role']}: {msg['content']}\n"
                    
                    prompt = f"""
                    You are a professional medical AI assistant.
                    Continue the conversation naturally.
                    Conversation History:
                    {conversation}
                    Give educational medical guidance only.
                    """
                    
                    ai_response = generate_chat_response(prompt)
                    
                    st.markdown(ai_response)

            # Save AI response
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response
                })

            st.warning(
                "This AI response is for educational purposes only and not medical advice."
                )

    # ======================
    # VOICE ASSISTANT
    # ======================

    elif page == "Voice Assistant":
        st.title("AI Voice Medical Assistant")
        
        if st.button("🎤 Speak"):
            with st.spinner("Listening..."):
                text = speech_to_text()

            if text:
                st.success(f"You said: {text}")

                prompt = f"""
                You are a helpful medical AI assistant.

                Answer this health question professionally:

                {text}
                """

                ai_response = generate_chat_response(prompt)

                st.info(ai_response)

                audio_file = text_to_speech(ai_response)

                audio_bytes = open(audio_file, "rb").read()

                st.audio(audio_bytes, format="audio/mp3")

            else:
                st.error("Could not recognize speech")
            
    # ======================
    # HISTORY
    # ======================
    #elif page == "Medical Report Analyzer":
     #   st.title("AI Medical Report Analyzer")  
      #  uploaded_file = st.file_uploader(
       #     "Upload Medical PDF",
        #    type=["pdf"]
         #   )
        #
        #if uploaded_file:
         #   
          #  with st.spinner("Processing PDF..."):
           #     vector_store = process_pdf(uploaded_file)
            #    
            #question = st.text_input(
             #   "Ask question about the report"
            #)
            #
            #if question:
             #   with st.spinner("Analyzing report..."):
              #      
               #     answer = ask_pdf_question(
                #        vector_store,
                 #       question
                  #  )
                    
                #st.subheader("AI Analysis")
                #st.info(answer)
                
                #st.warning(
                 #   "AI analysis is for educational purposes only."
                  #  )
    
    
    elif page=="History":

        rows=get_all() if st.session_state.role=="Admin" else get_hist(st.session_state.user)

        if rows:
            dfh=pd.DataFrame(rows,columns=["user","age","bp","glucose","chol","risk","score"])
            st.dataframe(dfh)

            st.line_chart(dfh["score"])

            st.download_button("Download CSV",dfh.to_csv(index=False),"report.csv")
        else:
            st.info("No data")
            
    
    