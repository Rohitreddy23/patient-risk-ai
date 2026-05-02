import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

# ======================
# DB
# ======================
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

    menu=["Prediction","Model Eval","EDA","History"] if st.session_state.role=="Admin" else ["Prediction","History"]
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

        age=st.slider("Age",18,100,40)
        bp=st.slider("BP",80,200,120)
        gl=st.slider("Glucose",50,300,100)
        ch=st.slider("Cholesterol",100,400,200)

        input_df=pd.DataFrame([[age,bp,gl,ch]],
            columns=["Age","Systolic_BP","Glucose_Lvl","Cholesterol_Lvl"])

        prob=best_model.predict_proba(input_df)[0][1]*100
        level=risk_level(prob)

        st.metric("Risk Score",f"{prob:.2f}%")
        st.success(f"Best Model: {best_name}")

        if level=="High": st.error(level)
        elif level=="Medium": st.warning(level)
        else: st.success(level)

        st.subheader("Suggestions")
        for s in suggestions(level):
            st.write("✔️",s)

        # ✅ FIXED FEATURE IMPORTANCE
        st.subheader("Feature Importance")

        features=["Age","Systolic_BP","Glucose_Lvl","Cholesterol_Lvl"]
        fig,ax=plt.subplots()

        if hasattr(best_model,"feature_importances_"):
            imp=best_model.feature_importances_

        elif hasattr(best_model,"coef_"):
            imp=abs(best_model.coef_[0])

        else:
            imp=None
            st.info("Not available")

        if imp is not None:
            imp=imp/np.sum(imp)
            ax.barh(features,imp)
            st.pyplot(fig)

        if st.button("Save"):
            save_hist(st.session_state.user,age,bp,gl,ch,level,prob)
            st.toast("Saved")

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
    # HISTORY
    # ======================
    elif page=="History":

        rows=get_all() if st.session_state.role=="Admin" else get_hist(st.session_state.user)

        if rows:
            dfh=pd.DataFrame(rows,columns=["user","age","bp","glucose","chol","risk","score"])
            st.dataframe(dfh)

            st.line_chart(dfh["score"])

            st.download_button("Download CSV",dfh.to_csv(index=False),"report.csv")
        else:
            st.info("No data")