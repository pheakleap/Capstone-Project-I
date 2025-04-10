import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
MODEL_DIR = 'models/naive_bayes/' # Path relative to project root
MODEL_PATH = os.path.join(MODEL_DIR, 'naive_bayes_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'disease_label_encoder.joblib')
SYMPTOMS_PATH = os.path.join(MODEL_DIR, 'symptoms_list.joblib')
PRECAUTIONS_PATH = os.path.join(MODEL_DIR, 'precautions_processed.csv')
DESCRIPTION_PATH = os.path.join(MODEL_DIR, 'description_processed.csv')
SEVERITY_PATH = os.path.join(MODEL_DIR, 'severity_processed.csv')

# --- Load Artifacts ---
try:
    model = joblib.load(MODEL_PATH)
    disease_encoder = joblib.load(ENCODER_PATH)
    symptoms_list = joblib.load(SYMPTOMS_PATH)
    df_precautions = pd.read_csv(PRECAUTIONS_PATH).set_index('Disease')
    df_description = pd.read_csv(DESCRIPTION_PATH).set_index('Disease')
    df_severity = pd.read_csv(SEVERITY_PATH).set_index('Symptom')
    print("Model and data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading artifacts: {e}")
    print("Please ensure the Naive_Bayes_Training.ipynb notebook has been run successfully.")
    # Handle error appropriately, maybe exit or disable prediction
    model = None
    disease_encoder = None
    symptoms_list = []
    # Create empty dataframes to avoid errors later if files not found
    df_precautions = pd.DataFrame(columns=['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'])
    df_description = pd.DataFrame(columns=['Description'])
    df_severity = pd.DataFrame(columns=['weight'])


# --- Prediction Function ---
def predict_disease(selected_symptoms):
    if model is None or disease_encoder is None or not symptoms_list:
        return "Model not loaded. Please run the training notebook.", "", "", "", ""

    if not selected_symptoms:
         return "Please select at least one symptom.", "", "", "", ""

    # Create input vector (binary: 1 if symptom selected, 0 otherwise)
    input_vector = np.zeros(len(symptoms_list))
    for symptom in selected_symptoms:
        if symptom in symptoms_list:
            # Find the index of the symptom and set it to 1
            try:
                idx = symptoms_list.index(symptom)
                input_vector[idx] = 1
            except ValueError:
                # Should not happen if symptoms_list is correct
                print(f"Warning: Symptom '{symptom}' not found in the list.")
                pass # Ignore symptoms not in the original list

    # Reshape for the model (expects 2D array)
    input_vector = input_vector.reshape(1, -1)

    # Predict probabilities
    probabilities = model.predict_proba(input_vector)[0]

    # Get top 3 predictions
    top_n = 3
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_diseases = disease_encoder.inverse_transform(top_indices)
    top_probabilities = probabilities[top_indices]

    # --- Prepare Output ---
    # Top prediction details
    top_disease_name = top_diseases[0]
    top_disease_prob = top_probabilities[0]

    # Description
    try:
        description = df_description.loc[top_disease_name, 'Description']
    except KeyError:
        description = "No description available."

    # Precautions
    try:
        precautions_series = df_precautions.loc[top_disease_name]
        precautions = "\n".join([f"- {p}" for p in precautions_series.dropna().tolist()])
        if not precautions:
             precautions = "No precautions listed."
    except KeyError:
        precautions = "No precautions available."

    # Severity Score
    severity_score = 0
    unknown_symptoms = []
    for symptom in selected_symptoms:
        try:
            # Use .get() with default 0 for severity lookup
            severity_score += df_severity.loc[symptom, 'weight'] if symptom in df_severity.index else 0
            if symptom not in df_severity.index:
                 unknown_symptoms.append(symptom)
        except KeyError: # Should be handled by .get() now, but keep for safety
             unknown_symptoms.append(symptom)

    severity_text = f"Calculated Severity Score: {severity_score}"
    if unknown_symptoms:
        severity_text += f"\n(Severity unknown for: {', '.join(unknown_symptoms)})"


    # Other potential predictions
    other_predictions = ""
    if len(top_diseases) > 1:
        other_predictions = "Other possibilities:\n"
        for i in range(1, len(top_diseases)):
            other_predictions += f"- {top_diseases[i]} ({top_probabilities[i]:.2%})\n"


    return (
        f"{top_disease_name} ({top_disease_prob:.2%})",
        description,
        precautions,
        severity_text,
        other_predictions.strip()
    )

# --- Gradio Theme ---
# Basic theme with blue/white colors
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill="#F0F8FF", # AliceBlue
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="#E0E0E0",
    block_title_text_weight="600",
    block_label_text_weight="600",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

# --- Gradio Interface Definition ---
with gr.Blocks(theme=theme) as iface:
    gr.Markdown("# Disease Prediction System")
    gr.Markdown("Select the symptoms you are experiencing to get potential disease predictions, descriptions, precautions, and a severity score.")

    with gr.Row():
        with gr.Column(scale=1):
            symptom_input = gr.CheckboxGroup(
                choices=symptoms_list,
                label="Select Symptoms",
                info="Check all symptoms that apply."
            )
            predict_btn = gr.Button("Predict Disease", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## Prediction Results")
            predicted_disease_output = gr.Textbox(label="Most Likely Disease (Probability)", interactive=False)
            description_output = gr.Textbox(label="Description", lines=5, interactive=False)
            precautions_output = gr.Textbox(label="Precautions", lines=5, interactive=False)
            severity_output = gr.Textbox(label="Symptom Severity Score", interactive=False)
            other_preds_output = gr.Textbox(label="Other Possibilities", lines=3, interactive=False)


    predict_btn.click(
        fn=predict_disease,
        inputs=symptom_input,
        outputs=[
            predicted_disease_output,
            description_output,
            precautions_output,
            severity_output,
            other_preds_output
        ]
    )

# --- Launch the Interface ---
if __name__ == "__main__":
    if model is not None: # Only launch if model loaded
        iface.launch()
    else:
        print("\nApplication cannot launch because the model failed to load.")
        print("Please run the 'notebooks/NaiveBayes/Naive_Bayes_Training.ipynb' notebook first.")
