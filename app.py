import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import tempfile

# Load model and features
model = joblib.load("asthma_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title
st.title("🩺 Asthma Diagnosis App 🩺")

# Patient details
patient_name = st.text_input("Enter Patient Name:")
patient_age = st.number_input("Enter Patient Age:", min_value=0, max_value=120, step=1)
patient_gender = st.selectbox("Select Patient Gender:", ["Select", "Male", "Female", "Other"])

st.write("### Select Patient Symptoms")
user_input = []
positive_symptoms = []

# Create symptom inputs
for feature in feature_names:
    value = st.radio(f"{feature}:", ["No", "Yes"], key=feature)
    is_yes = 1 if value == "Yes" else 0
    user_input.append(is_yes)
    if is_yes:
        positive_symptoms.append(feature)

# Predict button
if st.button("Predict"):
    if not patient_name.strip() or patient_age == 0 or patient_gender == "Select":
        st.warning("Please enter patient name, age, and select gender.")
    else:
        input_df = pd.DataFrame([user_input], columns=feature_names)
        prediction = model.predict(input_df)[0]
        diagnosis = "The Patient is Asthma Positive" if prediction == 1 else "The Patient is Asthma Negative"

        st.subheader(f"Patient: {patient_name}, Age: {patient_age}, Gender: {patient_gender}")
        st.subheader(f"Report: {diagnosis}")

        if prediction == 1 and positive_symptoms:
            st.write("### Positive Symptoms:")
            for symp in positive_symptoms:
                st.markdown(f"- {symp}")

        # Generate PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica",style="BU", size=20)

        pdf.cell(200, 10, text="Asthma Diagnosis Report", ln=True, align="C")
        pdf.set_font("helvetica", size=12)

        pdf.cell(200, 10, text=f"Patient Name: {patient_name}", ln=True)
        pdf.cell(200, 10, text=f"Age: {int(patient_age)}", ln=True)
        pdf.cell(200, 10, text=f"Gender: {patient_gender}", ln=True)
        pdf.cell(200, 10, text=f"Diagnosis: {diagnosis}", ln=True)

        if diagnosis == "The Patient is Asthma Positive":
            pdf.cell(200, 10, text="Positive Symptoms:", ln=True)
            for symp in positive_symptoms:
                pdf.cell(200, 10, text=f"° {symp}", ln=True)

        # Save and offer download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, "rb") as f:
                st.download_button("Download Report", f, file_name=f"{patient_name}_asthma_report.pdf", mime="application/pdf")
