import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- App Config ---
st.set_page_config(page_title="Bank Churn Predictor", layout="centered")
st.title("💼 Bank Customer Churn Prediction")

# --- Sidebar Info ---
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This app predicts the risk of customer churn based on bank records using a trained ML model.  
- Enter the customer's details  
- Click **Predict Churn Risk**  
- Get an instant risk score  
""")

# --- Input Form ---
st.markdown("### 🧾 Enter Customer Details Below")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.number_input("Customer ID", min_value=100000, value=100000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.slider("Tenure (Years with bank)", 0, 10, 3)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col2:
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=70000.0)

    submitted = st.form_submit_button("🚀 Predict Churn Risk")

# --- Prediction Logic ---
if submitted:
    # Convert categorical values
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0
    gender_1 = 1 if gender == "Male" else 0
    geography_1 = 1 if geography == "Germany" else 0
    geography_2 = 1 if geography == "Spain" else 0

    # Build input vector in correct order
    input_data = np.array([[
        customer_id,             # id
        credit_score,            # CreditScore
        age,                     # Age
        tenure,                  # Tenure
        balance,                 # Balance
        num_of_products,         # NumOfProducts
        has_cr_card,             # HasCrCard
        is_active_member,        # IsActiveMember
        estimated_salary,        # EstimatedSalary
        geography_1,             # Geography_1 (Germany)
        geography_2,             # Geography_2 (Spain)
        gender_1                 # Gender_1 (Male)
    ]])

    # Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]  # Probability of churn

    # --- Output ---
    st.markdown("---")
    st.markdown("### 🔍 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High risk of churn!**\n\n💣 **Risk Score: {proba*100:.2f}%**")
    else:
        st.success(f"✅ **Low risk of churn.**\n\n🟢 **Risk Score: {proba*100:.2f}%**")
