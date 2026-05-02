import pandas as pd
import pickle

print("🚀 Script started")

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv("../data/realtime_patient_data.csv")

print("✅ Data loaded")
print(data.head())

# -------------------------------
# CLEAN
# -------------------------------
data = data.dropna()

# -------------------------------
# TARGET
# -------------------------------
data['risk'] = (data['Risk_Score'] >= 50).astype(int)

# -------------------------------
# FEATURES
# -------------------------------
X = data[['Age', 'Cholesterol_Lvl', 'Glucose_Lvl']]
y = data['risk']

print("✅ Features ready")

# -------------------------------
# SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# SCALING
# -------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# MODELS
# -------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=4),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# -------------------------------
# EVALUATION
# -------------------------------
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, pred)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "auc": auc
    }

    print(f"\n📊 {name}")
    print("Accuracy:", round(acc, 3))
    print("AUC:", round(auc, 3))
    print("Confusion Matrix:\n", cm)

# -------------------------------
# SELECT BEST MODEL
# -------------------------------
best_model_name = max(results, key=lambda x: results[x]["auc"])
best_model = results[best_model_name]["model"]

print(f"\n🏆 Best Model: {best_model_name}")

# -------------------------------
# SAVE BEST MODEL
# -------------------------------
pickle.dump(best_model, open("../models/model.pkl", "wb"))
pickle.dump(scaler, open("../models/scaler.pkl", "wb"))

print("✅ Best model saved successfully!")

