import streamlit as st
import joblib
import pandas as pd
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# 📌 1️⃣ Load the trained loan approval model
model = joblib.load("/Users/jiawei/Downloads/MSM_532/Streamlit/best_lg.pkl")

# 📌 2️⃣ Load the training data for LIME explanation
X_train = pd.read_csv("/Users/jiawei/Downloads/MSM_532/Streamlit/X_train.csv")
feature_names = X_train.columns.tolist()

# 📌 3️⃣ Create LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["Approved", "Rejected"],
    mode="classification"
)

# 📌 4️⃣ Set page layout for a clean UI
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# 📌 5️⃣ Apply simple styling
st.markdown("""
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "San Francisco", Arial, sans-serif; }
        .title { text-align: center; font-size: 30px; font-weight: 600; color: #222; margin-bottom: 5px; }
        .subtitle { text-align: center; font-size: 16px; color: #555; margin-bottom: 20px; }
        .result-box { padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; }
        .positive { background-color: #E8F5E9; color: #2E7D32; }
        .negative { background-color: #FFEBEE; color: #C62828; }
        .divider { border-top: 1px solid #E0E0E0; margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

# 📌 6️⃣ Title Section
st.markdown("<h1 class='title'>Loan Approval Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter the details below to check loan eligibility</p>", unsafe_allow_html=True)

# 📌 7️⃣ User Inputs
NoCreditHistory = st.radio("Do you have a credit history?", ["Yes", "No"])
AverageMInFile = st.number_input("Average account age (months)", min_value=0, max_value=500, value=50)
ExternalRiskEstimate = st.slider("External Credit Risk Estimate (0-100)", min_value=0, max_value=100, value=60)
NumTotalTrades = st.number_input("Total number of trades (credit history)", min_value=0, value=10)
MSinceMostRecentDelq = st.number_input("Months since most recent delinquency", min_value=0, max_value=500, value=12)
Total_Debt_Burden = st.slider("Total Debt Burden (0-100)", min_value=0, max_value=100, value=30)
NumInqLast6M = st.number_input("Number of inquiries in last 6 months", min_value=0, max_value=50, value=3)

# 📌 8️⃣ Convert user input
NoCreditHistory = 1 if NoCreditHistory == "No" else 0  # Convert Yes/No to binary 0/1

# 📌 9️⃣ Prepare Data for Prediction
user_data = pd.DataFrame({
    "NoCreditHistory": [NoCreditHistory],
    "AverageMInFile": [AverageMInFile],
    "ExternalRiskEstimate": [ExternalRiskEstimate],
    "NumTotalTrades": [NumTotalTrades],
    "MSinceMostRecentDelq": [MSinceMostRecentDelq],
    "Total_Debt_Burden": [Total_Debt_Burden],
    "NumInqLast6M": [NumInqLast6M]
})

# 📌 1️⃣0️⃣ Divider
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# 📌 1️⃣1️⃣ Predict Button
predict_button = st.button("Predict Loan Approval")

# Initialize session state for prediction if不存在
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None

# 📌 1️⃣2️⃣ Prediction Output 
if predict_button:
    st.session_state.prediction = model.predict(user_data)[0]
    st.session_state.probability = model.predict_proba(user_data)[0][1]

# 如果预测结果存在，则显示结果以及后续对比分析
if st.session_state.prediction is not None:
    prediction = st.session_state.prediction
    probability = st.session_state.probability

    # Display results
    risk_label = "Loan Rejected" if prediction == 1 else "Loan Approved"
    result_class = "negative" if prediction == 1 else "positive"
    st.markdown(f"<div class='result-box {result_class}'><b>{risk_label}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 16px;'>Default Probability: <b>{probability:.2%}</b></p>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # 📌 1️⃣3️⃣ What-If Analysis
    st.markdown("<h3 style='text-align: center; font-size:22px; font-weight:bold;'>What-If Analysis</h3>", unsafe_allow_html=True)

    variable_to_adjust = st.selectbox(
        "Choose a variable to modify:",
        list(user_data.columns)
    )

    if variable_to_adjust == "NoCreditHistory":
        min_val, max_val = 0, 1  # Binary (0 or 1)
    else:
        # Use X_train min/max values to ensure a broader range
        min_val = int(X_train[variable_to_adjust].min())
        max_val = int(X_train[variable_to_adjust].max())

    # Ensure valid slider values
    if min_val == max_val:
        max_val = min_val + 10  # Ensures the range is always valid

    new_value = st.slider(
        f"Adjust {variable_to_adjust}:",
        min_value=int(min_val),
        max_value=int(max_val),
        value=int(user_data[variable_to_adjust][0])
    )

    modified_data = user_data.copy()
    modified_data[variable_to_adjust] = new_value

    # 📌 Predict with Modified Input
    modified_prediction = model.predict(modified_data)[0]
    modified_probability = model.predict_proba(modified_data)[0][1]

    # 📌 Display Comparison Results
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; font-size:20px; font-weight:bold;'>Prediction Comparison</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔹 Original Prediction**")
        st.markdown(f"<p style='text-align: center; font-size:16px;'><b>{'Loan Rejected' if prediction == 1 else 'Loan Approved'}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size:16px;'>Default Probability: <b>{probability:.2%}</b></p>", unsafe_allow_html=True)

    with col2:
        st.markdown("**🔹 Modified Prediction**")
        st.markdown(f"<p style='text-align: center; font-size:16px;'><b>{'Loan Rejected' if modified_prediction == 1 else 'Loan Approved'}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size:16px;'>Default Probability: <b>{modified_probability:.2%}</b></p>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if modified_prediction != prediction:
        st.markdown("<p style='text-align: center; font-size:18px; font-weight:bold; color:#2E7D32;'>✅ Adjusting this factor changed the loan decision!</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center; font-size:18px; font-weight:bold; color:#D32F2F;'>⚠️ The adjustment did not change the decision.</p>", unsafe_allow_html=True)