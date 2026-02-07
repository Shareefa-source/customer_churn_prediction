# Customer Churn Analysis & Prediction – Telecom Industry Case Study
## 🚀 Project Overview

Customer churn refers to customers who stop using a company’s service. Identifying churn early allows businesses to take proactive steps to improve customer retention.

This project builds and compares multiple supervised machine learning models to predict customer churn and provides an interactive dashboard where users can:

Explore churn patterns

Train and evaluate models

Predict churn probability for individual customers

---

## 📌 Business Problem

Telecom companies lose significant revenue when customers churn. Retaining existing customers is more cost-effective than acquiring new ones, making churn prediction a critical business problem.

🎯 Business Objective

Identify customers likely to churn

Understand why customers churn

Enable data-driven retention strategies


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

## 📈 Model Evaluation &  Key Insights

Compared multiple classification models side-by-side

Identified key churn drivers such as:

Contract type

Tenure

Monthly and Total Charges

Visualized:

Feature importance

Confusion matrices

ROC curves

## Key Insights from Analysis

- Customers on month-to-month contracts are significantly more likely to churn than those on long-term contracts.

- Higher monthly charges strongly correlate with increased churn risk.

- Customers with longer tenure show substantially lower churn rates.

- Value-added services such as Online Security and Tech Support are associated with improved customer retention.


---


## 📊 Model Performance Summary

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80     | 0.79      | 0.75   | 0.77     | 0.84    |
| Decision Tree       | 0.78     | 0.76      | 0.73   | 0.74     | 0.81    |
| Random Forest       | 0.82     | 0.81      | 0.78   | 0.79     | 0.87    |



---


## Model Selection Rationale
Random Forest provided the best overall balance between recall and ROC-AUC, making it suitable for identifying at-risk customers while minimizing false negatives.


---


## 📌 Recommendations
- Incentivize long-term contracts for high-risk customers
- Offer discounts to customers with high monthly charges
- Promote value-added services to improve retention


--- 


## 🖥️ Streamlit Web Application

The Streamlit app allows users to:

Upload their own churn dataset (.csv)

Perform EDA interactively

Train and compare ML models

Input new customer details

Predict churn probability in real time

Live App: http://localhost:8502/

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
**Skills**: SQL | Python | Power BI | Machine Learning  





---

