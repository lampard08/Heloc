import streamlit as st
import joblib
import pandas as pd
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# 📌 4️⃣ Apply KMeans clustering directly on raw data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
X_train['Cluster'] = kmeans.labels_

# 📌 5️⃣ Set page layout for a clean UI
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# 📌 6️⃣ Title Section
st.title("Loan Approval Prediction")
st.subheader("Enter the details below to check loan eligibility")

# 📌 7️⃣ User Inputs
NoCreditHistory = st.radio("Do you have a credit history?", ["Yes", "No"])

disable_inputs = NoCreditHistory == "No"

def disable_if_no_credit(value):
    return 0 if disable_inputs else value

AverageMInFile = st.number_input("Average account age (months) (Max: 800)", min_value=0, max_value=800, value=disable_if_no_credit(50), disabled=disable_inputs)
ExternalRiskEstimate = st.slider("External Credit Risk Estimate (0-100) (Max: 100)", min_value=0, max_value=100, value=disable_if_no_credit(60), disabled=disable_inputs)
NumTotalTrades = st.number_input("Total number of trades (credit history) (Max: 1000)", min_value=0, max_value=1000, value=disable_if_no_credit(10), disabled=disable_inputs)
MSinceMostRecentDelq = st.number_input("Months since most recent delinquency (Max: 800)", min_value=0, max_value=800, value=disable_if_no_credit(12), disabled=disable_inputs)
Total_Debt_Burden = st.slider("Total Debt Burden (0-100) (Max: 100)", min_value=0, max_value=100, value=disable_if_no_credit(30), disabled=disable_inputs)
NumInqLast6M = st.number_input("Number of inquiries in last 6 months (Max: 500)", min_value=0, max_value=500, value=disable_if_no_credit(3), disabled=disable_inputs)

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
    st.session_state.cluster = kmeans.predict(user_data)[0]

    if st.session_state.prediction == 1:
        explanation = st.session_state.exp.as_list()
        filtered_explanations = [(feature, importance) for feature, importance in explanation if importance > 0]

        if filtered_explanations:
            st.subheader("How to Improve Your Loan Approval Chances")
            st.write("Your loan application was rejected due to the following reasons. Here’s how you can improve:")
            for feature, importance in filtered_explanations:
                # if "CreditHistory" in feature:
                #     st.write("- Try to maintain a longer credit history and avoid late payments.")
                if "ExternalRisk" in feature:
                    st.write("- Your credit risk score is low. You can improve it by paying bills on time and reducing outstanding debt.")
                if "Total_Debt_Burden" in feature:
                    st.write("- You may have too much debt compared to your income. Paying off some existing loans can increase your approval chances.")
                if "Inq" in feature:
                    st.write("- Too many recent credit applications can be a red flag. Try to limit new credit inquiries for a few months.")
                if "Delq" in feature:
                    st.write("- Having recent late payments or delinquencies can impact your score. Ensuring all payments are on time will help.")
            st.write("Making these changes can significantly improve your chances of getting your loan approved in the future.")
        else:
            st.write("We couldn't find specific areas needing improvement. However, maintaining good financial habits will always help your loan approval chances.")

    if NoCreditHistory == 1:
        st.subheader("Reminder")
        st.write("You have no credit history. A guarantor/co-signer or International Credit Report or Proof of Income is required.")

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
