import gradio as gr
import json
import joblib

# Load encoders & model
with open("D:/Term7/Capstone-Project-I/data/processed/name_symptom.json", "r") as f:
    symptom_dict = json.load(f)

with open("D:/Term7/Capstone-Project-I/data/processed/name_disease.json", "r") as f:
    disease_dict = json.load(f)

model = joblib.load("D:/Term7/Capstone-Project-I/src/model/random_forest.pkl")

# Function to predict disease
def predict_disease(user_symptoms):
    encoded_symptoms = [symptom_dict[f"Symptom_{i+1}"].get(symptom, -1) for i, symptom in enumerate(user_symptoms)]
    input_data = encoded_symptoms + [-1] * (17 - len(user_symptoms))  # Fill missing with -1

    # Predict
    prediction = model.predict([input_data])[0]
    predicted_disease = {str(v): k for k, v in disease_dict.items()}[str(prediction)]

    return predicted_disease

# Gradio Interface
gr.Interface(
    fn=predict_disease,
    inputs=gr.CheckboxGroup(list(symptom_dict["Symptom_1"].keys()), label="Select Symptoms"),
    outputs="text",
    title="Disease Prediction",
    description="Select your symptoms to get a disease prediction"
).launch()
