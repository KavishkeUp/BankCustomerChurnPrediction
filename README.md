# 💼 Bank Customer Churn Prediction App

This project is a **machine learning web app** built using **Streamlit** that predicts whether a bank customer is likely to churn (leave the bank) based on their profile data.

It was developed as part of the [Kaggle competition: Bank Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge/overview).

---

## 🚀 Features

- 🔍 Predict customer churn risk using a trained Random Forest model
- 📊 Input customer features such as age, balance, geography, etc.
- ✅ Live probability-based prediction output
- 🎨 Clean and responsive user interface with background styling
- 💾 Pre-trained model and scaler included (`churn_model.pkl`, `scaler.pkl`)

---

## 🧠 Model Information

- **Model Used:** Random Forest Classifier
- **Features Used:**
  - Credit Score
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Estimated Salary
  - Geography (encoded)
  - Gender (encoded)
- **Scaler:** StandardScaler from scikit-learn

---

## 📁 Project Structure

bank-churn-app/
│
├── app.py # Streamlit application
├── scaler.pkl # Saved StandardScaler used during model training
├── churn_model.pkl # Trained Random Forest model
├── requirements.txt # Required Python packages
└── README.md # Project documentation (this file)


---

## ▶️ How to Run the App

### 🔧 1. Clone the repository

```bash
git clone https://github.com/your-username/bank-churn-app.git
cd bank-churn-app

📦 2. Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

📥 3. Install dependencies

pip install -r requirements.txt

🏁 4. Run the Streamlit app

streamlit run app.py

🌐 Deployment

You can deploy this app to:
Streamlit Cloud
Render / Heroku / Railway / AWS / Azure, etc.
Make sure to include churn_model.pkl, scaler.pkl, and requirements.txt in your repo.

📸 App Screenshot

<img width="1366" height="649" alt="image" src="https://github.com/user-attachments/assets/2aff2296-cd2b-47c8-a3eb-e3df8c1d40da" />

📚 Acknowledgements
Kaggle Dataset: Bank Customer Churn Prediction Challenge
scikit-learn, pandas, NumPy, Streamlit
Background image credit: Lytics

