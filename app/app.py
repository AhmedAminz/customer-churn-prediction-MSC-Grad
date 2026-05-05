import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")
st.title("Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict churn probability.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 35)
    tenure = st.slider("Tenure (months)", 1, 60, 30)
    usage_frequency = st.slider("Usage Frequency", 1, 30, 15)
    support_calls = st.slider("Support Calls", 0, 10, 3)
    payment_delay = st.slider("Payment Delay (days)", 0, 30, 10)

with col2:
    total_spend = st.slider("Total Spend ($)", 100, 1000, 500)
    last_interaction = st.slider("Last Interaction (days ago)", 1, 30, 15)
    gender = st.selectbox("Gender", ["Female", "Male"])
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Annual", "Monthly", "Quarterly"])

if st.button("Predict", use_container_width=True):
    input_df = pd.DataFrame([{
        'Age': age,
        'Tenure': tenure,
        'Usage Frequency': usage_frequency,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Total Spend': total_spend,
        'Last Interaction': last_interaction,
        'Gender': gender,
        'Subscription Type': subscription,
        'Contract Length': contract
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"High Churn Risk — {probability:.1%} probability")
    else:
        st.success(f"Low Churn Risk — {probability:.1%} probability")