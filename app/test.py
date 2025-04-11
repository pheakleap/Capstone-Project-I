import gradio as gr
import numpy as np
import pandas as pd
import joblib

# Load models
model = joblib.load("D:/Term7/Capstone-Project-I/src/models/best_ensemble_model.joblib")
symptom_encoder = joblib.load("D:/Term7/Capstone-Project-I/src/models/symptom_encoder.joblib")
disease_encoder = joblib.load("D:/Term7/Capstone-Project-I/src/models/disease_encoder.joblib")

all_symptoms = symptom_encoder.classes_
all_diseases = disease_encoder.classes_

def predict_disease(*symptom_values):
    try:
        # Convert checkbox values to symptom names
        selected_symptoms = [all_symptoms[i] for i, checked in enumerate(symptom_values) if checked]

        # Create feature DataFrame with correct column names
        features = pd.DataFrame(np.zeros((1, len(all_symptoms))), columns=all_symptoms)

        # Set selected symptoms to 1
        for symptom in selected_symptoms:
            features[symptom] = 1

        # Get predictions
        predicted_index = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # Top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]

        return {
            "Primary Diagnosis": {
                "Disease": all_diseases[top3_indices[0]],
                "Confidence": f"{probabilities[top3_indices[0]] * 100:.1f}%"
            },
            "Secondary Options": [
                {
                    "Disease": all_diseases[top3_indices[1]],
                    "Confidence": f"{probabilities[top3_indices[1]] * 100:.1f}%"
                },
                {
                    "Disease": all_diseases[top3_indices[2]],
                    "Confidence": f"{probabilities[top3_indices[2]] * 100:.1f}%"
                }
            ],
            "Symptoms Considered": len(selected_symptoms),
            "Detailed Symptoms": [s.replace('_', ' ').title() for s in selected_symptoms]
        }

    except Exception as e:
        return {"error": str(e)}

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Medical Diagnosis Assistant") as app:
    gr.Markdown("## 🩺 Medical Diagnosis Assistant")

    with gr.Row():
        with gr.Column(scale=2):
            # Symptom checkboxes (3 columns)
            checkboxes = []
            with gr.Tabs():
                with gr.Tab("General Symptoms"):
                    with gr.Row():
                        for col in range(3):  # 3 columns
                            with gr.Column():
                                start_idx = col * (len(all_symptoms) // 3)
                                end_idx = (col + 1) * (len(all_symptoms) // 3)
                                for symptom in all_symptoms[start_idx:end_idx]:
                                    checkboxes.append(gr.Checkbox(label=symptom.replace('_', ' ').title()))
            submit_btn = gr.Button("Analyze Symptoms", variant="primary")

        with gr.Column(scale=1):
            with gr.Accordion("Results", open=True):
                diagnosis_output = gr.JSON()
            with gr.Accordion("How to Interpret", open=False):
                gr.Markdown("""
                    - **Primary Diagnosis**: Most likely condition
                    - **Secondary Options**: Alternative possibilities
                    - **Confidence**: Model's certainty
                    """)

    # Prediction logic
    submit_btn.click(
        fn=predict_disease,
        inputs=checkboxes,
        outputs=diagnosis_output
    )

# Launch
if __name__ == "__main__":
    app.launch() # Removed unneeded parameters.