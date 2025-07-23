# app.py – Streamlit Web App for Predicting Early-Stage Diabetes
#This app:
#    Loads a trained LightGBM model (saved as a pipeline)
#    Collects user inputs for age, gender, and 14 symptoms
#    Predicts whether the case is Positive (1) or Negative (0)


import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# Step 1: Load the Trained Model


# Path to the trained LightGBM model
MODEL_PATH = "rf_symptom_full.pkl"

# Define a function to load the model safely
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not Path(path).exists():
        st.error(f"Model file not found at: {path}")
        st.stop()
    return joblib.load(path)

# Load the model once when app starts
model = load_model(MODEL_PATH)
st.success("Model loaded successfully. Ready for prediction!")

# -----------------------------------------------------------------
# Step 2: Set up the Web Interface
# -----------------------------------------------------------------

# App title and instructions
st.title("Early‑Stage Diabetes Prediction App")
st.markdown("""
Enter the required information below.  
Click **Predict** to find out if the person is at risk of early-stage diabetes  
(based on the symptoms and personal info).
""")

# Collect user input for age
age = st.slider("Age (in years)", min_value=1, max_value=120, value=40)

# Select gender
gender = st.radio("Gender", ["Male", "Female"])

# List of symptom-related questions (Yes/No)
symptom_questions = {
    "Polyuria (excessive urination)": "Polyuria",
    "Polydipsia (excessive thirst)": "Polydipsia",
    "Sudden weight loss": "sudden weight loss",
    "Weakness": "weakness",
    "Polyphagia (excessive hunger)": "Polyphagia",
    "Genital thrush": "Genital thrush",
    "Visual blurring": "visual blurring",
    "Itching": "Itching",
    "Irritability": "Irritability",
    "Delayed healing": "delayed healing",
    "Partial paresis": "partial paresis",
    "Muscle stiffness": "muscle stiffness",
    "Alopecia": "Alopecia",
    "Obesity": "Obesity"
}

# Store the answers in a dictionary
user_symptom_inputs = {}

st.markdown("Please answer the following symptom questions:")
for question, column_name in symptom_questions.items():
    answer = st.radio(f"{question}", ["No", "Yes"], key=column_name)
    user_symptom_inputs[column_name] = answer


# Step 3: Create Input DataFrame for Model

def create_input_dataframe():
    input_data = {"Age": age, "Gender": gender}
    input_data.update(user_symptom_inputs)
    return pd.DataFrame([input_data])

# Step 4: Predict Button and Display Results


if st.button("Predict"):
    input_df = create_input_dataframe()

    # Make prediction using the trained pipeline
    prediction = model.predict(input_df)[0]              # 0 or 1
    prediction_proba = model.predict_proba(input_df)[0][1]  # probability of class 1 (Positive)

    # Show results
    result = "Positive (1)" if prediction == 1 else "Negative (0)"
    st.subheader(f"Prediction Result: {result}\n\nConsult a qualified physician before making health decisions.\n")
    st.write(f"Probability of being Positive: **{prediction_proba:.3f}**")

    # Show the raw data passed to the model
    st.markdown("Model Input Preview:")
    st.dataframe(input_df)

#To run the model open the terminal. Copy "streamlit run app.py" and paste it and enter.
