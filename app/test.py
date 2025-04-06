import gradio as gr
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load all necessary artifacts
model = joblib.load("D:/Term7/Capstone-Project-I//src/models/best_ensemble_model.joblib")
symptom_encoder = joblib.load("D:/Term7/Capstone-Project-I//src/models/symptom_encoder.joblib")
disease_encoder = joblib.load("D:/Term7/Capstone-Project-I//src/models/disease_encoder.joblib")

# Get all symptom and disease names
all_symptoms = symptom_encoder.classes_
all_diseases = disease_encoder.classes_

def predict_disease(*selected_symptoms_indices):
    """Predict disease based on selected symptom indices"""
    try:
        # Convert Gradio checkbox indices to symptom names
        selected_symptoms = [all_symptoms[i] for i, checked in enumerate(selected_symptoms_indices) if checked]
        
        # Create feature vector (0 for absent, 1 for present)
        features = np.zeros(len(all_symptoms))
        for symptom in selected_symptoms:
            idx = np.where(symptom_encoder.classes_ == symptom)[0][0]
            features[idx] = 1
        
        # Reshape for prediction (scaling is handled internally by the pipelines)
        features_reshaped = features.reshape(1, -1)
        
        # Get predictions and probabilities
        predicted_index = model.predict(features_reshaped)[0]
        probabilities = model.predict_proba(features_reshaped)[0]
        
        # Get top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        
        # Format results
        results = {
            "Primary Diagnosis": {
                "Disease": all_diseases[top3_indices[0]],
                "Confidence": f"{probabilities[top3_indices[0]]*100:.1f}%"
            },
            "Secondary Options": [
                {
                    "Disease": all_diseases[top3_indices[1]],
                    "Confidence": f"{probabilities[top3_indices[1]]*100:.1f}%"
                },
                {
                    "Disease": all_diseases[top3_indices[2]],
                    "Confidence": f"{probabilities[top3_indices[2]]*100:.1f}%"
                }
            ],
            "Symptoms Considered": len(selected_symptoms),
            "Detailed Symptoms": [s.replace('_', ' ').title() for s in selected_symptoms]
        }
        
        return results
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Medical Diagnosis Assistant") as app:
    gr.Markdown("""
    # 🩺 Medical Diagnosis Assistant
    Select your symptoms to receive potential diagnoses.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Organize symptoms in tabs by category
            with gr.Tabs():
                # General Symptoms Tab
                with gr.Tab("General Symptoms"):
                    with gr.Row():
                        # Split symptoms into 3 columns
                        cols = 3
                        symptoms_per_col = (len(all_symptoms) + cols - 1) // cols
                        
                        for col_idx in range(cols):
                            with gr.Column():
                                start_idx = col_idx * symptoms_per_col
                                end_idx = min((col_idx + 1) * symptoms_per_col, len(all_symptoms))
                                
                                for symptom in all_symptoms[start_idx:end_idx]:
                                    gr.Checkbox(
                                        label=symptom.replace('_', ' ').title(),
                                        value=False
                                    )
            
            # Example cases
            gr.Examples(
                examples=[
                    [True if s in ["itching", "skin_rash", "nodal_skin_eruptions"] else False 
                     for s in all_symptoms],
                    [True if s in ["continuous_sneezing", "shivering", "watering_from_eyes"] else False 
                     for s in all_symptoms],
                    [True if s in ["high_fever", "headache", "vomiting", "fatigue"] else False 
                     for s in all_symptoms]
                ],
                label="Try Example Cases",
                inputs=[comp for comp in app.blocks.values() if isinstance(comp, gr.Checkbox)]
            )
            
            submit_btn = gr.Button("Analyze Symptoms", variant="primary")
        
        with gr.Column(scale=1):
            with gr.Accordion("Diagnosis Results", open=True):
                diagnosis_output = gr.JSON(label="Prediction Results")
                
            with gr.Accordion("How to Interpret Results", open=False):
                gr.Markdown("""
                - **Primary Diagnosis**: The most likely condition based on your symptoms
                - **Secondary Options**: Other possible conditions to consider
                - **Confidence Percentage**: The model's certainty in each prediction
                """)
    
    # Footer
    gr.Markdown("""
    <div style='text-align: center; margin-top: 20px; font-size: 0.9em; color: #666;'>
    <i>Important: This tool provides informational predictions only and is not a substitute 
    for professional medical advice, diagnosis, or treatment.</i>
    </div>
    """)
    
    # Connect the button
    submit_btn.click(
        fn=predict_disease,
        inputs=[comp for comp in app.blocks.values() if isinstance(comp, gr.Checkbox)],
        outputs=diagnosis_output
    )

# Run the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )