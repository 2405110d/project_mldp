import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("model.pkl")

st.set_page_config(page_title="SmartDeposit", layout="wide")

st.title("SmartDeposit")
st.subheader("Bank Term Deposit Subscription Predictor")

st.markdown("""
Predict whether a customer will subscribe to a term deposit.
Use this tool to target high-probability customers and improve campaign ROI.
""")

age = st.slider("Age", 18, 95, 35)
balance = st.number_input("Account Balance", value=1000)
duration = st.number_input("Call Duration (seconds)", value=200)
campaign = st.slider("Number of Contacts During Campaign", 1, 50, 1)
pdays = st.number_input("Days Since Last Contact", value=999)
previous = st.slider("Previous Contacts", 0, 20, 0)

job = st.selectbox("Job", [
    "admin.", "technician", "services", "management",
    "retired", "blue-collar", "unemployed"
])

marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
contact = st.selectbox("Contact Type", ["cellular", "telephone"])
month = st.selectbox("Last Contact Month", [
    "jan","feb","mar","apr","may","jun",
    "jul","aug","sep","oct","nov","dec"
])

if st.button("Predict Subscription Probability"):

    df_input = pd.DataFrame({
        "age": [age],
        "balance": [balance],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "month": [month]
    })

    df_input = pd.get_dummies(df_input)

    model_columns = model.feature_names_in_
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.success(f"High probability of subscription ({probability:.2%})")
    else:
        st.warning(f"Low probability of subscription ({probability:.2%})")

    st.progress(float(probability))
    
