import gradio as gr
import json
import joblib

# Load symptom encodings
try:
    with open("D:/Term7/Capstone-Project-I/data/processed/name_symptom.json", "r") as f:
        symptom_encodings = json.load(f)
except FileNotFoundError:
    print("Error: symptom_encodings.json not found.")
    symptom_encodings = {}

# Load trained model
try:
    model = joblib.load("D:/Term7/Capstone-Project-I/src/models/random_forest.pkl")
    print("Model loaded successfully:", model)  # Debugging print
except FileNotFoundError:
    print("Error: random_forest.pkl model file not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load disease mappings
try:
    with open("D:/Term7/Capstone-Project-I/data/processed/name_disease.json", "r") as f:
        disease_mappings = json.load(f)
except FileNotFoundError:
    print("Warning: disease_name.json not found. Disease names will not be displayed.")
    disease_mappings = {}

# Invert the disease_mappings dictionary
disease_index_to_name = {v: k for k, v in disease_mappings.items()}
def process_symptoms(selected_symptoms):
    # Ensure all 17 symptoms are encoded, with 'none' (75) for missing symptoms
    encoded_symptoms = [symptom_encodings.get(symptom, 75) for symptom in selected_symptoms]  # Default to 'none' (75) for missing symptoms

    # If fewer than 17 features are provided, pad the rest with the 'none' encoding
    while len(encoded_symptoms) < 17:
        encoded_symptoms.append(75)  # Add 'none' encoding for missing symptoms

    # Check if the number of features is correct (17)
    if len(encoded_symptoms) != 17:
        return "Error: The number of symptoms should be 17 features.", [], "Prediction not possible."

    if model:
        try:
            prediction = model.predict([encoded_symptoms])
            print("Model prediction:", prediction)  # Debugging print

            predicted_disease_index = prediction[0]

            # Use inverted disease index-to-name mapping
            if predicted_disease_index in disease_index_to_name:
                predicted_disease_name = disease_index_to_name[predicted_disease_index]
                prediction_text = f"Predicted disease: {predicted_disease_name}"
            else:
                prediction_text = f"Predicted disease index: {predicted_disease_index} (No name found)"
        except Exception as e:
            prediction_text = f"Prediction error: {e}"
    else:
        prediction_text = "Model not loaded. Prediction unavailable."

    return f"Selected symptoms: {', '.join(selected_symptoms)}, Encoded values: {encoded_symptoms}", encoded_symptoms, prediction_text

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            checkbox_group = gr.CheckboxGroup(list(symptom_encodings.keys()), label="Select your symptoms")
            submit_button = gr.Button("Submit")

        with gr.Column():
            text_output = gr.Textbox(label="Output")
            machine_output = gr.State([])
            prediction_output = gr.Textbox(label="Prediction")

    submit_button.click(
        fn=process_symptoms,
        inputs=checkbox_group,
        outputs=[text_output, machine_output, prediction_output]
    )

demo.launch()
