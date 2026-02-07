# Customer Churn Prediction App
## 🚀 Project Overview

Customer churn refers to customers who stop using a company’s service. Identifying churn early allows businesses to take proactive steps to improve customer retention.

This project builds and compares multiple supervised machine learning models to predict customer churn and provides an interactive dashboard where users can:

Explore churn patterns

Train and evaluate models

Predict churn probability for individual customers

---

## 🧠 Machine Learning Models Used

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Each model is evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

---

## 📂 Dataset Description

The dataset includes customer-level telecom data such as:

Customer Info: Gender, SeniorCitizen, Partner, Dependents

Services: InternetService, OnlineSecurity, TechSupport, Streaming services

Account Details: Contract type, Payment method, Tenure, Monthly & Total charges

Target Variable: Churn (Yes / No)

---

## 🔍 Exploratory Data Analysis (EDA)

Churn distribution and class imbalance analysis

Summary statistics and missing value handling

Correlation analysis of numerical features

Churn analysis by categorical variables

Box plots to compare churn vs non-churn customers

---

## ⚙️ Data Preprocessing

Converted non-numeric fields to numeric values

Handled missing values

Label-encoded categorical variables

Standardized numerical features using StandardScaler

Prepared clean datasets for model training

---

## 📈 Model Evaluation & Insights

Compared multiple classification models side-by-side

Identified key churn drivers such as:

Contract type

Tenure

Monthly and Total Charges

Visualized:

Feature importance

Confusion matrices

ROC curves

---

## 🖥️ Streamlit Web Application

The Streamlit app allows users to:

Upload their own churn dataset (.csv)

Perform EDA interactively

Train and compare ML models

Input new customer details

Predict churn probability in real time

---

## 🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn, Plotly

Streamlit

---

## 👤 Author
**Sharifatu Musah**
Data Analyst
