import streamlit as st
import joblib
import pandas as pd
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 📌 1️⃣ Load the trained loan approval model
model = joblib.load("/Users/jiawei/Downloads/MSM_532/Heloc/best_lg.pkl")
knn_model = joblib.load("/Users/jiawei/Downloads/MSM_532/Heloc/knn_model.pkl")

# 📌 2️⃣ Load the training data for LIME explanation and KNN profiling
X_train = pd.read_csv("/Users/jiawei/Downloads/MSM_532/Heloc/X_train.csv")
y_train = pd.read_csv("/Users/jiawei/Downloads/MSM_532/Heloc/y_train.csv")
feature_names = X_train.columns.tolist()

# 📌 3️⃣ Create LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["Approved", "Rejected"],
    mode="classification"
)

# 📌 4️⃣ Standardize data for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 📌 5️⃣ Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
X_train['Cluster'] = kmeans.labels_

# 📌 6️⃣ Set page layout for a clean UI
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# 📌 7️⃣ Title Section
st.title("Loan Approval Prediction")
st.subheader("Enter the details below to check loan eligibility")

# 📌 8️⃣ User Inputs
NoCreditHistory = st.radio("Do you have a credit history?", ["Yes", "No"])

disable_inputs = NoCreditHistory == "No"

def disable_if_no_credit(value):
    return 0 if disable_inputs else value

AverageMInFile = st.number_input("Average account age (months)", min_value=0, max_value=500, value=disable_if_no_credit(50), disabled=disable_inputs)
ExternalRiskEstimate = st.slider("External Credit Risk Estimate (0-100)", min_value=0, max_value=100, value=disable_if_no_credit(60), disabled=disable_inputs)
NumTotalTrades = st.number_input("Total number of trades (credit history)", min_value=0, max_value=800, value=disable_if_no_credit(10), disabled=disable_inputs)
MSinceMostRecentDelq = st.number_input("Months since most recent delinquency", min_value=0, max_value=500, value=disable_if_no_credit(12), disabled=disable_inputs)
Total_Debt_Burden = st.slider("Total Debt Burden (0-100)", min_value=0, max_value=100, value=disable_if_no_credit(30), disabled=disable_inputs)
NumInqLast6M = st.number_input("Number of inquiries in last 6 months", min_value=0, max_value=1000, value=disable_if_no_credit(3), disabled=disable_inputs)

NoCreditHistory = 1 if NoCreditHistory == "No" else 0

user_data = pd.DataFrame({
    "NoCreditHistory": [NoCreditHistory],
    "AverageMInFile": [AverageMInFile],
    "ExternalRiskEstimate": [ExternalRiskEstimate],
    "NumTotalTrades": [NumTotalTrades],
    "MSinceMostRecentDelq": [MSinceMostRecentDelq],
    "Total_Debt_Burden": [Total_Debt_Burden],
    "NumInqLast6M": [NumInqLast6M]
})

if st.button("Predict Loan Approval"):
    st.session_state.prediction = model.predict(user_data)[0]
    st.session_state.probability = model.predict_proba(user_data)[0][1]
    st.session_state.exp = explainer.explain_instance(user_data.iloc[0].values, model.predict_proba, num_features=5)
    user_scaled = scaler.transform(user_data)
    st.session_state.cluster = kmeans.predict(user_scaled)[0]

if "prediction" in st.session_state:
    prediction = st.session_state.prediction
    probability = st.session_state.probability
    cluster = st.session_state.cluster
    risk_levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    risk_category = risk_levels.get(cluster, "Unknown")
    
    st.markdown(f"<h2 style='text-align: center; color: {'red' if prediction == 1 else 'green'};'>{'❌ Loan Rejected' if prediction == 1 else '✅ Loan Approved'}</h2>", unsafe_allow_html=True)
    st.write(f"**Default Probability:** {probability:.2%}")
    
    st.subheader("Decision Explanation")
    st.write("Key features that influenced this decision:")
    explanation = st.session_state.exp.as_list()
    for feature, importance in explanation[:5]:
        st.write(f"- {feature}: {importance:.2f}")
    
    st.subheader("What-If Analysis")
    variable_to_adjust = st.selectbox("Choose a variable to modify:", list(user_data.columns))
    new_value = st.slider(f"Adjust {variable_to_adjust}:", min_value=int(X_train[variable_to_adjust].min()), max_value=int(X_train[variable_to_adjust].max()), value=int(user_data[variable_to_adjust][0]))
    modified_data = user_data.copy()
    modified_data[variable_to_adjust] = new_value
    modified_prediction = model.predict(modified_data)[0]
    modified_probability = model.predict_proba(modified_data)[0][1]
    st.write(f"**Modified Prediction:** {'❌ Loan Rejected' if modified_prediction == 1 else '✅ Loan Approved'}")
    st.write(f"**Original Probability:** {probability:.2%}")
    st.write(f"**Modified Probability:** {modified_probability:.2%}")
    
    st.subheader("Customer Profiling & Benchmarking")
    similar_customers = X_train[X_train['Cluster'] == cluster]
    approval_rate = y_train.loc[similar_customers.index].mean()[0] * 100
    st.write(f"Your profile is similar to **{approval_rate:.2f}%** of approved applicants in your category.")
    st.write(f"You belong to the **{risk_category}** risk category.")

 # 📌 Model Performance & Transparency
    st.subheader("Model Performance & Transparency")
    X_train_model = X_train.drop(columns=["Cluster"], errors="ignore")
    y_pred = model.predict(X_train_model)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    st.write("### Model Evaluation Metrics")
    st.write(f"- **Accuracy:** {accuracy:.2f}")
    st.write(f"- **Precision:** {precision:.2f}")
    st.write(f"- **Recall:** {recall:.2f}")
    st.write(f"- **F1 Score:** {f1:.2f}")

    # 📌 Feature Importance Visualization
    st.write("### Feature Importance")
    if hasattr(model, 'coef_'):
        feature_importance = pd.Series(model.coef_[0], index=X_train_model.columns).sort_values(key=abs, ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feature_importance, y=feature_importance.index)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance in Loan Approval Model")
        st.pyplot(plt)
    else:
        st.write("Feature importance visualization is not available for this model.")

