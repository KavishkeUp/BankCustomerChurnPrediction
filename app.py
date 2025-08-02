import streamlit as st
import numpy as np
import joblib

# Background Image 
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)),
                    url("https://www.lytics.com/wp-content/uploads/2022/07/article-Customer-Churn-Prediction_-How-to-do-It-and-Reduce-Customer-Churn.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# App Config
st.set_page_config(page_title="Bank Churn Predictor", layout="centered")
st.title("👥📉 Bank Customer Churn Prediction")

# Sidebar Info
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This app predicts the risk of customer churn based on bank records using a trained ML model.  
- Enter the customer's details  
- Click **Predict Churn Risk**  
- Get an instant risk score  
""")

# Input Form
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

if submitted:
    # Convert categorical inputs
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0
    gender_1 = 1 if gender == "Male" else 0
    geography_1 = 1 if geography == "Germany" else 0
    geography_2 = 1 if geography == "Spain" else 0

    # Build input vector WITHOUT customer_id (exclude it from model)
    input_data = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_of_products,
        has_cr_card,
        is_active_member,
        estimated_salary,
        geography_1,
        geography_2,
        gender_1
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    # Show result along with Customer ID
    st.markdown("---")
    st.markdown(f"### 🔍 Prediction Result for Customer ID: {customer_id}")

    if prediction == 1:
        st.error(f"⚠️ **High risk of churn!**\n\n💣 **Risk Score: {proba*100:.2f}%**")
    else:
        st.success(f"✅ **Low risk of churn.**\n\n🟢 **Risk Score: {proba*100:.2f}%**")
