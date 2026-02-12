import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

from PIL import Image


# ===============================
# APP CONFIG
# ===============================

st.set_page_config(page_title="Customer Churn App", layout="wide")

# Safe logo loading
try:
    logo = Image.open("logo.jpg")
    st.image(logo, width=150)
except:
    pass

st.title("📊 Customer Churn Prediction App")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Churn CSV",
    type=["csv"]
)

# ===============================
# LOAD DATA
# ===============================

if uploaded_file is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Clean TotalCharges if exists
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    ).fillna(0)

st.subheader("Dataset Preview")
st.dataframe(df.head())


# ===============================
# FEATURE / TARGET SPLIT
# ===============================

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

if "Churn" not in df.columns:
    st.error("Dataset must contain 'Churn' column.")
    st.stop()

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Encode target if needed
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)


# ===============================
# PREPROCESSING PIPELINE
# ===============================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", 
         Pipeline(steps=[
             ("encoder", 
              LabelEncoder())  # handled below differently
         ]), categorical_features)
    ],
    remainder="drop"
)

# Since LabelEncoder cannot be directly used in ColumnTransformer,
# we encode categoricals manually before pipeline:

X_encoded = X.copy()
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

scaler = StandardScaler()
X_encoded[numeric_features] = scaler.fit_transform(
    X_encoded[numeric_features]
)

# ===============================
# TRAIN / TEST SPLIT
# ===============================

test_size = st.sidebar.slider(
    "Select Test Size",
    0.1, 0.5, 0.3, 0.05
)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# ===============================
# MODEL SELECTION
# ===============================

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "Random Forest"]
)

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=5, random_state=42)

else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)


# ===============================
# TRAIN MODEL
# ===============================

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ===============================
# MODEL EVALUATION
# ===============================

st.subheader("📈 Model Evaluation")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
col5.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")

fig_cm, ax_cm = plt.subplots()
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ROC Curve
st.subheader("ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_proba)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=model_choice)
ax_roc.plot([0, 1], [0, 1], "k--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)


# ===============================
# USER PREDICTION SECTION
# ===============================

st.subheader("🔮 Predict New Customer")

user_input = {}

for col in X.columns:
    if col in categorical_features:
        user_input[col] = st.selectbox(
            col,
            df[col].unique()
        )
    else:
        user_input[col] = st.number_input(
            col,
            value=float(df[col].mean())
        )

input_df = pd.DataFrame([user_input])

# Encode input
for col in categorical_features:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Scale numeric
input_df[numeric_features] = scaler.transform(
    input_df[numeric_features]
)

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"Customer Likely to Churn (Probability: {probability:.2f})")
else:
    st.success(f"Customer Likely to Stay (Probability: {probability:.2f})")
