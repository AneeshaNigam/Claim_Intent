import streamlit as st
import pandas as pd
import joblib

# Load trained model pipeline
model = joblib.load("model.pkl")

st.title("🚗 Insurance Claim Prediction")

st.markdown("Predict whether a customer is likely to claim based on their profile.")

# Input form
with st.form("claim_form"):
    subscription_length = st.number_input("Subscription Length (years)", min_value=0.0, step=0.1)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0.0, step=0.1)
    customer_age = st.number_input("Customer Age", min_value=0, step=1)
    region_code = st.selectbox("Region Code", ["C2", "C8", "C10", "C13"])
    fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
    ncap_rating = st.slider("NCAP Rating", 0, 5, 3)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "subscription_length": subscription_length,
        "vehicle_age": vehicle_age,
        "customer_age": customer_age,
        "region_code": region_code,
        "fuel_type": fuel_type,
        "ncap_rating": ncap_rating
    }])

    prediction = model.predict(input_data)[0]
    result = "✅ Claim Likely" if prediction == 1 else "❌ No Claim Expected"
    st.subheader("Prediction:")
    st.success(result)
