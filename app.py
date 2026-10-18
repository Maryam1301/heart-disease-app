"""Heart Disease Prediction - Streamlit app (Assignment 2, Task 2)
Student: Maryam Akram | Reg. No: FA24-BED-089
Learning project only - NOT a medical tool.
"""
import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# Same fixed category lists used in the notebook (Step 5.1), so the numeric codes match training.
CATEGORIES = {
    "Sex": ["Male", "Female"],
    "ChestPain": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    "ExerciseAngina": ["Yes", "No"],
    "STSlope": ["Upsloping", "Flat", "Downsloping"],
}
encoders = {name: LabelEncoder().fit(values) for name, values in CATEGORIES.items()}


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_svc_trained_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("❤️ Heart Disease Prediction System")
st.write("Choose the four patient attributes below. The trained Support Vector Classifier "
         "predicts whether heart disease is likely.")

sex = st.selectbox("Sex", CATEGORIES["Sex"])
chest_pain = st.selectbox("Chest pain type", CATEGORIES["ChestPain"])
exercise_angina = st.selectbox("Exercise-induced angina", CATEGORIES["ExerciseAngina"])
st_slope = st.selectbox("Slope of ST segment (exercise ECG)", CATEGORIES["STSlope"])

if st.button("Predict"):
    row = pd.DataFrame({"Sex": [sex], "ChestPain": [chest_pain],
                        "ExerciseAngina": [exercise_angina], "STSlope": [st_slope]})
    for col in row.columns:
        row[col] = encoders[col].transform(row[col])
    result = int(model.predict(row)[0])
    if result == 1:
        st.error("Prediction: HEART DISEASE LIKELY")
    else:
        st.success("Prediction: HEART DISEASE NOT LIKELY")

st.warning("Educational demo trained on 303 patients and only 4 attributes. "
           "It is NOT a medical diagnosis tool.")
st.caption("Maryam Akram (FA24-BED-089), AIC354 - Machine Learning Fundamentals.")
