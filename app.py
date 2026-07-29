import streamlit as st
import pickle
import numpy as np


# Load model and scaler
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details")


Age = st.number_input("Age", 1, 100)

RestingBP = st.number_input("Resting Blood Pressure")

Cholesterol = st.number_input("Cholesterol")

FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

MaxHR = st.number_input("Maximum Heart Rate")

Oldpeak = st.number_input("Oldpeak")


# Encoded categorical features

Sex_M = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])


ChestPainType_ATA = st.selectbox("Chest Pain ATA", [0, 1])
ChestPainType_NAP = st.selectbox("Chest Pain NAP", [0, 1])
ChestPainType_TA = st.selectbox("Chest Pain TA", [0, 1])


RestingECG_Normal = st.selectbox("Resting ECG Normal", [0, 1])
RestingECG_ST = st.selectbox("Resting ECG ST", [0, 1])


ExerciseAngina_Y = st.selectbox("Exercise Angina Yes", [0, 1])


ST_Slope_Flat = st.selectbox("ST Slope Flat", [0, 1])
ST_Slope_Up = st.selectbox("ST Slope Up", [0, 1])


if st.button("Predict"):

    input_data = np.array([[
        Age,
        RestingBP,
        Cholesterol,
        FastingBS,
        MaxHR,
        Oldpeak,
        Sex_M,
        ChestPainType_ATA,
        ChestPainType_NAP,
        ChestPainType_TA,
        RestingECG_Normal,
        RestingECG_ST,
        ExerciseAngina_Y,
        ST_Slope_Flat,
        ST_Slope_Up
    ]])


    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)


    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")