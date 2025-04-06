import gradio as gr
import json
import joblib
import numpy as np

# Load data from JSON files
with open("D:/Term7/Capstone-Project-I/data/processed/symptom_mapping.json", "r") as f:
    features_data = json.load(f)

with open("D:/Term7/Capstone-Project-I/data/processed/disease_mapping.json", "r") as f:
    diseases_data = json.load(f)

feature_names = [feature["name"] for feature in features_data]
disease_names = list(diseases_data.keys())

# Load your trained model
model = joblib.load("D:/Term7/Capstone-Project-I/model/trained_model.joblib") #Replace with your model path

def predict(selected_features):
    selected_feature_ids = [features_data[feature_names.index(feature)]["id"] for feature in selected_features]

    # Create input vector for the model
    input_vector = np.zeros(len(features_data))
    for feature_id in selected_feature_ids:
        input_vector[feature_id] = 1

    # Reshape the input vector for prediction
    input_vector = input_vector.reshape(1, -1)

    # Get the model's prediction probabilities
    prediction_probabilities = model.predict_proba(input_vector)[0]

    # Create a dictionary of disease names and probabilities
    prediction = {disease: probability for disease, probability in zip(disease_names, prediction_probabilities)}

    # Format output as disease: probability (percentage)
    formatted_prediction = {disease: f"{probability * 100:.2f}%" for disease, probability in prediction.items()}

    # Sort output by probability
    sorted_prediction = dict(sorted(formatted_prediction.items(), key=lambda item: float(item[1].strip('%')), reverse=True))

    return sorted_prediction

iface = gr.Interface(
    fn=predict,
    inputs=gr.CheckboxGroup(choices=feature_names, label="Select your symptoms"),
    outputs=gr.Label(label="Prediction Results"),
    title="Medical Prediction Model",
    description="Select your symptoms to get a prediction of possible diseases.",
)

iface.launch()