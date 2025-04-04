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

try:
    with open("D:/Term7/Capstone-Project-I/data/processed/name_disease.json", "r") as f:
        disease_mappings = json.load(f)
except FileNotFoundError:
    print("Warning: disease_name.json not found. Disease names will not be displayed.")
    disease_mappings = {}

def process_symptoms(selected_symptoms):
    if not selected_symptoms:
        return "No symptoms selected.", [], "No prediction available."
    else:
        encoded_symptoms = [symptom_encodings.get(symptom, -1) for symptom in selected_symptoms]
        if -1 in encoded_symptoms:
            return "One or more symptoms not found.", [], "Prediction not possible."

        if model:
            try:
                prediction = model.predict([encoded_symptoms])
                print("Model prediction:", prediction)  # Debugging print
                print("Prediction type:", type(prediction)) # Debugging print

                predicted_disease_index = prediction[0]

                if disease_mappings and str(predicted_disease_index) in disease_mappings:
                    predicted_disease_name = disease_mappings[str(predicted_disease_index)]
                    prediction_text = f"Predicted disease: {predicted_disease_name}"
                else:
                    prediction_text = f"Predicted disease index: {predicted_disease_index}"
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