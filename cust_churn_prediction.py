import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import rmse
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, roc_auc_score, recall_score
from sklearn.metrics import roc_curve
import plotly.express as px
import plotly.figure_factory as ff
import joblib

logo = Image.open('logo.jpg')
st.image(logo)

st.title('Customer Churn Prediction')
uploaded_file = st.sidebar.file_uploader("Upload Customer Churn CSV", type=["csv"])

# Initialize dataframe
df = None

# If file is uploaded, read it
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

def page1():
    st.title("Welcome to the Customer Churn Prediction App")

    st.markdown("""
        ### Group 3 Members
        - 1.Stephen Kwesi Darko - 22253086 - Group Leader
        - 2.Michael Boakye Sarpong - 11410789
        - 3.Clifford Srekumah Gli - 22252473
        - 4.Joseph Kingsley Nana Safo - 22255408
        - 5.Emefa Akaba – 22260476
        ### Overview
        This application is designed to help you **analyze and predict customer churn** based on a telecom customer dataset.

        Churn refers to customers who **stop using a service**. Identifying these customers early allows companies to take proactive steps and reduce churn rates.
        
        ### Project Objective
        The objective of this project is to create and implement an interactive web application that utilizes supervised machine learning 
        algorithms to predict customer churn. This application is meant to assist businesses in telecom and other related industries to 
        pinpoint customers who are on the verge of discontinuing services so that necessary actions can be taken to retain them.
        
        
        ### Dataset Description
        The dataset used in this app typically includes the following features:
        - **CustomerID**: Unique identifier for each customer
        - **Demographic Info**: Gender, SeniorCitizen, Partner, Dependents
        - **Service Usage**: InternetService, OnlineSecurity, TechSupport, etc.
        - **Account Info**: Contract Type, Payment Method, MonthlyCharges, Tenure
        - **Churn**: Whether the customer left the company (Yes/No)

        ### App Features
        This app provides the following functionalities:
        - **Exploratory Data Analysis (EDA)**: Visualize distributions, correlations, and trends
        - **Data Preprocessing**: Handle missing values, encode categorical features, and scale numerical values
        - **Model Training**: Train Logistic Regression, Decision Tree, and Random Forest models
        - **Model Evaluation**: Compare model performance using metrics like accuracy, precision, and ROC-AUC
        - **Prediction**: Make predictions on new or existing customer data

        ### How to Use the App
        1. **Upload** your telecom customer dataset (`.csv`) from the sidebar
        2. Navigate through the pages using the **sidebar menu**:
           - Exploratory Data Analysis
           - Data Processing
           - Model Training
           - Classification Results
        3. Use insights to understand churn drivers and optimize retention strategies

        ---
        """)



def page2():
    if df is None:
        st.warning("Please upload a CSV file to proceed.")
        return
    st.write("Data Preview", df.head(7))
    st.title('Exploratory Data Analysis')
    # Summary stats
    st.write("Summary Statistics")
    st.write(df.describe().T)

    # Data types
    st.subheader("Column Data Types")
    st.write(df.dtypes)

    # Handle non-numeric total charges, total charges was recognized as an object and not integer
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    st.subheader('Missing Values')
    st.write(df.isnull().sum())

    # Replace missing values in Total Charges with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    st.subheader('Missing Values after replacing NAN with 0')
    st.write(df.isnull().sum())

    # Extract numeric and categorical columns
    num_cols = df.select_dtypes(include='number').columns
    categorical_cols = ['Contract', 'InternetService', 'PaymentMethod', 'gender']

    # Define options for visualizations
    viz_options = [
        "Churn Count (Bar Chart)",
        "Histograms of Numeric Features",
        "Correlation Matrix",
        "Churn by Categorical Features",
        "Box Plots by Churn"
    ]

    st.subheader("Select a Visualization")
    selected_viz = st.selectbox("Choose a visualization to display:", viz_options)

    # Visualization logic
    # Create plots based on selection
    if selected_viz == "Churn Count (Bar Chart)":
        st.subheader("Churn vs No Churn")
        st.bar_chart(df['Churn'].value_counts())
        st.caption("Shows class imbalance between churned and retained customers.")

    elif selected_viz == "Histograms of Numeric Features":
        st.subheader("Histogram of Numerical Features")
        for col in num_cols:
            st.write(f"Histogram of {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

    elif selected_viz == "Correlation Matrix":
        st.subheader("Correlation Matrix")
        st.write("Correlation for numerical variables")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.caption("High correlation between TotalCharges and MonthlyCharges indicates pricing structure dependency.")


    elif selected_viz == "Churn by Categorical Features":
        for col in categorical_cols:
            st.subheader(f'Churn by {col}')
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, hue='Churn', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

    elif selected_viz == "Box Plots by Churn":
        for col in num_cols:
            st.subheader(f'{col} by Churn')
            fig, ax = plt.subplots()
            sns.boxplot(x='Churn', y=col, data=df, ax=ax)
            st.pyplot(fig)

# === Page 3: Data Preprocessing ===
def page3():
    st.title('Data Preprocessing')

    # Previewing original data
    st.write("Original Data", df.head(7))

    # Make a copy of the data
    df_scaled = df.copy()
    custID = df['customerID']
    df_scaled = df_scaled.drop(columns='customerID', axis=1)

    # Convert and fill missing values
    df_scaled['TotalCharges'] = pd.to_numeric(df_scaled['TotalCharges'], errors='coerce').fillna(0)

    # Scale numerical features
    num_cols = df_scaled.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

    # Encode categorical features
    cat_cols = df_scaled.select_dtypes(include='object').columns
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_scaled[col] = le.fit_transform(df_scaled[col])
        label_encoders[col] = le

    # Now df_scaled contains both standardized numerics and encoded categoricals
    st.subheader('Processed Data')

    # Combine back with customer ID for reference
    df_scaled = pd.concat([custID, df_scaled], axis=1)
    st.write(df_scaled.head(7))

    st.success("Data has been successfully processed and scaled.")
    st.markdown("Processed data includes encoded categorical variables and scaled numeric columns, ensuring models treat all features uniformly.")

    # Store processed data in session state
    st.session_state['df_scaled'] = df_scaled


def page4():
    st.title('Model Training')

    if 'df_scaled' not in st.session_state:
        st.warning("Please preprocess the data first in 'Data Processing' page.")
        return

    df_scaled = st.session_state['df_scaled']

    # Split data
    X = df_scaled.drop(columns=['customerID', 'Churn'])
    y = df_scaled['Churn']

    # Model selection options
    st.subheader("Train/Test Split Settings")
    split_ratio = st.slider("Select test size (proportion)", min_value=0.1, max_value=0.5, value=0.3, step=0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42, stratify=y
    )

    #Preview train and test data
    train_data = st.checkbox('Show X_train and y_train')
    test_data = st.checkbox('Show X_test and y_test')

    if train_data:
        st.write('Preview X Train data')
        st.write(X_train)
        st.write(f"Train size: {X_train.shape[0]} samples")
        st.write('Preview Y Train data')
        st.write(y_train)
        st.write(f"Test size: {X_test.shape[0]} samples")

    if test_data:
        st.write('Preview X Test data')
        st.write(X_test)
        st.write('Preview Y Test data')
        st.write(y_test)

    Class_options = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Compare All Models"
    ]
    model_select = st.selectbox('Select Model', Class_options)
    model_metrics = []  # to collect metrics for all models
    if model_select == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display performance metrics
        st.subheader("Model Evaluation")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
        col4.metric("F1-score", f"{f1_score(y_test, y_pred):.2f}")
        st.caption("Accuracy is the overall correctness")
        st.caption("Precision focuses on positive prediction quality")
        st.caption("Recall measures how many actual churns were identified.")
        st.caption(
            "**F1 Score**: Harmonic mean of precision and recall — balances false positives and false negatives.")

        # Feature Importance (coefficients)
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0]
        }).sort_values(by='Coefficient', key=abs, ascending=False)

        #Create dataframe for coefficient
        st.subheader("Feature Influence (Logistic Regression Coefficients)")
        st.dataframe(coefficients)

        #Create a barplot for coefficients
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coefficients)
        ax.set_title('Feature Influence on Churn (Higher = More Impact)')
        st.pyplot(fig)
        st.markdown("""
        ### Key Insights:
        *1. Most Influential Factors (Negative Coefficients):*
        - **Tenure** has the strongest **negative** impact on churn. The longer a customer has stayed, the less likely they are to leave.
        - **PhoneService** and **Contract** types also help reduce churn, suggesting more committed or long-term users.

        *2. Factors Increasing Churn (Positive Coefficients):*
        - **MonthlyCharges** and **TotalCharges** have the most **positive** impact. Higher bills are strongly linked to higher churn risk.
        - **PaperlessBilling** customers may be more tech-savvy and likely to switch providers, contributing to higher churn.

        *3. Moderate Influences:*
        - Features such as **OnlineSecurity**, **TechSupport**, and **InternetService** negatively correlate with churn, indicating customers who use these services tend to stay longer.

        *4. Minimal Impact Features:*
        - **Gender**, **StreamingTV**, **StreamingMovies**, and **Partner** have coefficients close to zero, showing very little impact on churn behavior.

        """)

        #Create Confusion plot
        st.subheader("Confusion Matrix:")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title('Logistic Regression Confusion Matrix')
        st.pyplot(fig_cm)
        st.caption("The matrix shows predicted vs actual values. True positives and true negatives represent correct predictions.")

        #Create ROC CURVE
        st.subheader('Roc Curve')
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label='Logistic Regression')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve - Logistic Regression')
        ax_roc.legend()
        st.pyplot(fig_roc)
        st.caption("The ROC Curve illustrates model ability to distinguish between classes. The higher the curve, the better the model.")



    #Decision tree
    elif model_select == "Decision Tree":
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt_model.fit(X_train, y_train)
        y_pred2 = dt_model.predict(X_test)
        y_proba = dt_model.predict_proba(X_test)[:, 1]

        # Evaluation
        st.subheader("Model Evaluation")
        col1i, col2i, col3i, col4i, col5i = st.columns(5)

        col1i.metric("Accuracy", f"{accuracy_score(y_test, y_pred2):.2f}")
        col2i.metric("Precision", f"{precision_score(y_test, y_pred2):.2f}")
        col3i.metric("Recall", f"{recall_score(y_test, y_pred2):.2f}")
        col4i.metric("F1 Score", f"{f1_score(y_test, y_pred2):.2f}")
        col5i.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.2f}")
        st.caption("Accuracy is the overall correctness")
        st.caption("Precision focuses on positive prediction quality")
        st.caption("Recall measures how many actual churns were identified.")
        st.caption(
            "**F1 Score**: Harmonic mean of precision and recall — balances false positives and false negatives.")

        # Feature Importance
        st.subheader("Feature Importance (Decision Tree)")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': dt_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        ax.set_title('Decision Tree Feature Importance')
        st.pyplot(fig)
        st.markdown("""
         ### Key Observations:

        #### 1. Top Influential Feature
        - **Contract** is by far the most significant feature (over 0.5 importance).
          - Customers on flexible or month-to-month contracts are much more likely to churn.
          - Locking customers into longer contracts could help reduce churn.

        #### 2. Other Important Features
        - **OnlineSecurity**, **tenure**, **InternetService**, and **MonthlyCharges** have moderate impact.
          - Longer tenure and security services are associated with retention.
          - High bills could be a sign of dissatisfaction leading to churn.

        #### 3. Low-Impact Features
        - Features like **gender**, **PhoneService**, **StreamingTV**, **OnlineBackup**, and **TechSupport** show very low or negligible importance.
          - These do not significantly influence churn decisions in the model.
        """)

        #Create confusion matrix plot
        st.write("Confusion Matrix:")
        fig2, ax2 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred2), annot=True, fmt='d', cmap='Blues')
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)
        st.caption(
            "The matrix shows predicted vs actual values. True positives and true negatives represent correct predictions.")

        # ROC Curve
        fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba)
        fig_roc_dt, ax_roc_dt = plt.subplots()
        ax_roc_dt.plot(fpr_dt, tpr_dt, label='Decision Tree')
        ax_roc_dt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax_roc_dt.set_xlabel('False Positive Rate')
        ax_roc_dt.set_ylabel('True Positive Rate')
        ax_roc_dt.set_title('ROC Curve - Decision Tree')
        ax_roc_dt.legend()
        st.pyplot(fig_roc_dt)

    #Random Forest
    elif model_select == "Random Forest":
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred3 = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]


        #Create metrics
        col1i, col2i, col3i, col4i, col5i = st.columns(5)

        col1i.metric("Accuracy", f"{accuracy_score(y_test, y_pred3):.2f}")
        col2i.metric("Precision", f"{precision_score(y_test, y_pred3):.2f}")
        col3i.metric("Recall", f"{recall_score(y_test, y_pred3):.2f}")
        col4i.metric("F1 Score", f"{f1_score(y_test, y_pred3):.2f}")
        col5i.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba_rf):.2f}")
        st.caption("Accuracy is the overall correctness")
        st.caption("Precision focuses on positive prediction quality")
        st.caption("Recall measures how many actual churns were identified.")
        st.caption(
            "**F1 Score**: Harmonic mean of precision and recall — balances false positives and false negatives.")

        #Display features in order of importance
        st.subheader("Feature Importance (Random Forest)")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        #Create bar plot for features
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        ax.set_title('Random Forest Feature Importance')
        st.pyplot(fig)
        st.markdown("""
        ### Key Observations:

        #### 1. *Top Features*
        - **TotalCharges** and *MonthlyCharges* are the most important predictors.
          - Customers with high or fluctuating bills are more likely to churn.
        - **Tenure** also ranks highly.
          - Longer-tenured customers are generally more loyal.
        - **Contract** is still a key feature, though not as dominant as in the Decision Tree model.

        #### 2. *Mid-Level Features*
        - **PaymentMethod**, **OnlineSecurity**, and **TechSupport** show moderate importance.
          - Secure services and support options can positively influence retention.
        - **Gender**, **OnlineBackup**, and **PaperlessBilling** have some predictive power, more than in previous models.

        #### 3. *Lower Importance Features*
        - Features like **StreamingTV**, **StreamingMovies**, **PhoneService**, and **SeniorCitizen** show very little impact.
          - These may not be strong differentiators of churn behavior in this context."""
        )

        #Create confusion matrix
        st.write("Confusion Matrix:")
        fig2, ax2 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred3), annot=True, fmt='d', cmap='Blues')
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)
        st.caption(
            "The matrix shows predicted vs actual values. True positives and true negatives represent correct predictions.")

        # ROC Curve
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        fig_roc_rf, ax_roc_rf = plt.subplots()
        ax_roc_rf.plot(fpr_rf, tpr_rf, label='Random Forest')
        ax_roc_rf.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax_roc_rf.set_xlabel('False Positive Rate')
        ax_roc_rf.set_ylabel('True Positive Rate')
        ax_roc_rf.set_title('ROC Curve - Random Forest')
        ax_roc_rf.legend()
        st.pyplot(fig_roc_rf)


        #Create database for all models and metrics
    elif model_select == "Compare All Models":

        model_metrics = []

        # Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        model_metrics.append({
            'Model': 'Logistic Regression',
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        })

        # Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)
        y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

        model_metrics.append({
            'Model': 'Decision Tree',
            'Accuracy': accuracy_score(y_test, y_pred_dt),
            'Precision': precision_score(y_test, y_pred_dt),
            'Recall': recall_score(y_test, y_pred_dt),
            'F1 Score': f1_score(y_test, y_pred_dt),
            'ROC AUC': roc_auc_score(y_test, y_proba_dt)
        })

        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

        model_metrics.append({
            'Model': 'Random Forest',
            'Accuracy': accuracy_score(y_test, y_pred_rf),
            'Precision': precision_score(y_test, y_pred_rf),
            'Recall': recall_score(y_test, y_pred_rf),
            'F1 Score': f1_score(y_test, y_pred_rf),
            'ROC AUC': roc_auc_score(y_test, y_proba_rf)
        })

        # Display comparison table
        st.subheader("Comparison of All Models")
        metrics_df = pd.DataFrame(model_metrics)
        st.dataframe(metrics_df)


def page5():
    st.title('User Page')
    if 'df_scaled' not in st.session_state:
        st.warning("Please preprocess the data first in 'Data Processing' page.")
        return

    # Load processed data
    df_scaled = st.session_state['df_scaled']
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    #Drop customerID and Church columns
    X_columns = df_scaled.drop(columns=['customerID', 'Churn']).columns

    # Select classifier by creating selectbox
    st.subheader("Select Classifier")
    model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Random Forest"])

    # Input section for new customer data
    st.subheader("Enter Customer Information")
    user_input = {}
    for col in X_columns:
        if col == 'SeniorCitizen':
            user_input[col] = st.number_input(f"{col} (0 = Not Senior, 1 = Senior)", min_value=0, max_value=1, step=1)
        elif df[col].dtype == 'object':
            options = df[col].unique().tolist()
            user_input[col] = st.selectbox(f"{col}", options)
        else:
            user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    # Convert input to dataframe
    input_df = pd.DataFrame([user_input])

    # Make a copy of input_df
    input_processed = input_df.copy()

    # Encode categorical columns using the same LabelEncoder fitted on df
    for col in input_processed.select_dtypes(include='object').columns:
        le = LabelEncoder()
        le.fit(df[col])  # Fit on original training data
        # Ensure the input value exists in the fitted classes
        input_processed[col] = input_processed[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    # Identify numeric columns in the original training data
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Scale numeric columns
    scaler = StandardScaler()
    scaler.fit(df[num_cols])  # Fit on training data
    input_processed[num_cols] = scaler.transform(input_processed[num_cols])

    st.subheader("Processed Input Data")
    st.write(input_processed)

    # Split for fitting model (you could reuse trained models if saved earlier)
    X = df_scaled.drop(columns=['customerID', 'Churn'])
    y = df_scaled['Churn']

    # Train and Predict
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    #Fit and predict
    model.fit(X, y)
    prediction = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]

    st.subheader("Prediction Result")
    st.write("Prediction:", " Churn" if prediction else " No Churn")
    st.write(f"Probability of Churn: {prob}")


pages = {
    'Introductory Page': page1,
    'Exploratory Data Analysis': page2,
    'Data Processing': page3,
    'Modeling': page4,
    'User Page': page5
}

##creating the sidebar with selection box
select_page = st.sidebar.selectbox('Select Page', list(pages.keys()))

# Display pages when clicked
pages[select_page]()
