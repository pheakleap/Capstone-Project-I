import gradio as gr
import sys
import os

# Add the src directory to the Python path to allow importing predict_ensemble
# Assumes this script is in app/ and src/ is at the same level as app/
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Now import from the models module within src
try:
    from models.predict_ensemble import load_ensemble_model, predict_disease
except ImportError as e:
    print(f"Error importing prediction functions: {e}")
    print(f"Please ensure src/models/predict_ensemble.py exists and '{SRC_DIR}' is accessible.")
    # Define dummy functions if import fails, so Gradio can still launch (with errors)
    def load_ensemble_model(): return None, [], []
    def predict_disease(*args, **kwargs): return []

# --- Load Model and Data ---
# Load the model, feature names, and target names once when the app starts
print("Loading ensemble model for Gradio interface...")
loaded_model, feature_names, target_names = load_ensemble_model()

if loaded_model is None:
    print("CRITICAL ERROR: Failed to load the model. The Gradio app may not function correctly.")
    # Provide a default empty list if loading failed, Gradio needs choices.
    feature_names = ["Error: Model not loaded"]

# Sort feature names alphabetically for better user experience in CheckboxGroup
feature_names.sort()

# --- Gradio Interface Function ---

def get_prediction(selected_symptoms):
    """
    Wrapper function for Gradio interface. Takes selected symptoms,
    calls the prediction function, and formats the output.
    """
    if loaded_model is None or not feature_names or not target_names:
        return {"Error": "Model not loaded or invalid."}

    if not selected_symptoms:
        return {"Info": "Please select at least one symptom."}

    # Call the prediction function from predict_ensemble.py
    # Pass the pre-loaded model and names
    predictions = predict_disease(selected_symptoms, loaded_model, feature_names, target_names, top_n=5)

    if not predictions:
        return {"Info": "Prediction returned no results. Ensure selected symptoms are relevant or check model."}

    # Format results for gr.Label (dictionary format)
    output_dict = {disease: f"{prob:.3f}" for disease, prob in predictions}
    return output_dict

# --- Create Gradio Interface ---

# Use CheckboxGroup for multi-select. Height can be adjusted.
symptom_input = gr.CheckboxGroup(
    choices=feature_names,
    label="Select Symptoms You Are Experiencing",
    info="Scroll down to see all available symptoms. Check all that apply."
)

# Use Label for output, which handles dictionaries well for ranked results.
prediction_output = gr.Label(
    num_top_classes=5,
    label="Top 5 Predicted Diseases (with probability)"
)

# Define the interface
iface = gr.Interface(
    fn=get_prediction,
    inputs=symptom_input,
    outputs=prediction_output,
    title="Disease Prediction from Symptoms (Ensemble Model)",
    description="Select symptoms from the list below to predict the potential disease. Based on an improved ensemble model (RF, XGBoost, Logistic Regression).",
    allow_flagging="never",
    examples=[
        # Provide examples based on likely symptoms in feature_names
        # Example: ['itching', 'skin_rash', 'nodal_skin_eruptions'], # Fungal infection?
        # Example: ['continuous_sneezing', 'shivering', 'chills'], # Allergy?
        # Example: ['fever', 'cough', 'headache', 'congestion'] # Common Cold/Flu?
        # Note: Actual examples depend heavily on the specific feature_names loaded
        [['itching', 'skin_rash']],
        [['fever', 'cough', 'headache']]
    ] if feature_names and "Error" not in feature_names[0] else None # Only show examples if model loaded
)

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    # share=True creates a public link (optional)
    iface.launch(share=False)
