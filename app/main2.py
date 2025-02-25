import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from typing import List, Tuple, Optional

# Load the model and necessary data
rf_model = joblib.load('../models/random_forest/rf_model.joblib')
le = joblib.load('../data/processed/label_encoder.joblib')
processed_data = joblib.load('../data/processed/processed_data.joblib')
feature_names = processed_data['feature_names']

# Load description and precaution data
description_df = pd.read_csv('../data/raw/symptom_Description.csv')
precaution_df = pd.read_csv('../data/raw/symptom_precaution.csv')


symptom_categories = {
    "General": [
        "fatigue", "lethargy", "malaise", "weakness", "weight_loss", "weight_gain",
        "high_fever", "mild_fever", "sweating", "chills", "dehydration",
        "toxic_look_(typhos)", "obesity", "restlessness"
    ],
    
    "Pain & Discomfort": [
        "headache", "stomach_pain", "abdominal_pain", "back_pain", "chest_pain",
        "joint_pain", "muscle_pain", "neck_pain", "knee_pain", "hip_joint_pain",
        "belly_pain", "pain_behind_the_eyes", "pain_during_bowel_movements",
        "pain_in_anal_region", "cramps"
    ],
    
    "Skin & Nails": [
        "itching", "skin_rash", "nodal_skin_eruptions", "yellowish_skin",
        "dischromic _patches", "bruising", "red_spots_over_body", "skin_peeling",
        "blister", "red_sore_around_nose", "yellow_crust_ooze", "pus_filled_pimples",
        "blackheads", "scurring", "silver_like_dusting", "small_dents_in_nails",
        "inflammatory_nails", "brittle_nails"
    ],
    
    "Respiratory": [
        "cough", "breathlessness", "runny_nose", "congestion", "sinus_pressure",
        "phlegm", "throat_irritation", "patches_in_throat", "mucoid_sputum",
        "rusty_sputum", "blood_in_sputum", "continuous_sneezing"
    ],
    
    "Digestive & Urinary": [
        "nausea", "vomiting", "diarrhoea", "constipation", "stomach_bleeding",
        "distention_of_abdomen", "acidity", "ulcers_on_tongue", "loss_of_appetite",
        "excessive_hunger", "indigestion", "bloody_stool", "irregular_sugar_level",
        "polyuria", "burning_micturition", "spotting_ urination", "passage_of_gases",
        "foul_smell_of urine", "continuous_feel_of_urine", "bladder_discomfort",
        "yellow_urine", "dark_urine"
    ],
    
    "Neurological": [
        "dizziness", "unsteadiness", "lack_of_concentration", "altered_sensorium",
        "depression", "irritability", "slurred_speech", "visual_disturbances",
        "blurred_and_distorted_vision", "spinning_movements", "loss_of_balance",
        "loss_of_smell", "mood_swings", "anxiety", "coma"
    ],
    
    "Cardiovascular & Circulation": [
        "chest_pain", "palpitations", "fast_heart_rate", "swollen_blood_vessels",
        "prominent_veins_on_calf", "cold_hands_and_feets", "fluid_overload"
    ],
    
    "Musculoskeletal": [
        "muscle_weakness", "muscle_wasting", "movement_stiffness", "stiff_neck",
        "swelling_joints", "painful_walking", "swollen_legs", "swollen_extremeties",
        "weakness_in_limbs", "weakness_of_one_body_side"
    ],
    
    "Face & Eyes": [
        "puffy_face_and_eyes", "sunken_eyes", "redness_of_eyes",
        "watering_from_eyes", "yellowing_of_eyes"
    ],
    
    "Medical History": [
        "family_history", "history_of_alcohol_consumption", "extra_marital_contacts",
        "receiving_blood_transfusion", "receiving_unsterile_injections"
    ]
}

# Add remaining symptoms to "Others" category
all_categorized = set([symptom for symptoms in symptom_categories.values() for symptom in symptoms])
symptom_categories["Others"] = [s for s in feature_names if s not in all_categorized]
# Constants
MAX_SYMPTOMS = 10
MIN_SYMPTOMS = 1

def create_probability_chart(diseases: List[str], probabilities: List[float]) -> go.Figure:
    """Create an enhanced probability chart with better styling."""
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities * 100,
            y=diseases,
            orientation='h',
            marker_color=colors,
            text=[f'{p:.1f}%' for p in probabilities * 100],
            textposition='auto',
            hovertemplate='Disease: %{y}<br>Probability: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Disease Probability Distribution',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Probability (%)',
        yaxis_title='Disease',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

def validate_symptoms(symptoms: List[str]) -> Tuple[bool, str]:
    """Validate the number of selected symptoms."""
    if not symptoms:
        return False, "⚠️ Please select at least one symptom."
    if len(symptoms) > MAX_SYMPTOMS:
        return False, f"⚠️ Maximum {MAX_SYMPTOMS} symptoms allowed. Please reduce your selection."
    return True, ""

def combine_symptoms(*category_symptoms) -> List[str]:
    """Combine symptoms from all categories and remove duplicates."""
    all_symptoms = []
    for symptoms in category_symptoms:
        if symptoms:
            all_symptoms.extend(symptoms)
    return list(set(all_symptoms))  # Remove duplicates

def get_severity_score(symptoms: List[str]) -> int:
    """Calculate severity score based on number and type of symptoms."""
    base_score = len(symptoms) * 10
    critical_symptoms = ["breathlessness", "chest_pain", "high_fever"]
    critical_count = sum(1 for s in symptoms if s in critical_symptoms)
    return base_score + (critical_count * 15)

def predict_disease(*category_symptoms) -> Tuple[Optional[go.Figure], str]:
    """Make disease predictions with enhanced error handling and severity assessment."""
    try:
        # Combine and validate symptoms
        all_symptoms = combine_symptoms(*category_symptoms)
        is_valid, message = validate_symptoms(all_symptoms)
        if not is_valid:
            return None, message
        
        # Create feature vector
        feature_vector = np.zeros(len(feature_names))
        for symptom in all_symptoms:
            if symptom in feature_names:
                feature_vector[feature_names.index(symptom)] = 1
        
        # Get prediction probabilities
        probabilities = rf_model.predict_proba([feature_vector])[0]
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = le.inverse_transform(top_3_idx)
        top_3_probs = probabilities[top_3_idx]
        
        # Calculate severity score
        severity_score = get_severity_score(all_symptoms)
        severity_level = (
            "🔴 High" if severity_score > 70 else
            "🟡 Medium" if severity_score > 40 else
            "🟢 Low"
        )
        
        # Create probability chart
        prob_chart = create_probability_chart(top_3_diseases, top_3_probs)
        
        # Format detailed output with enhanced styling
        result = f"📊 Analysis based on {len(all_symptoms)} symptoms:\n"
        result += f"Severity Level: {severity_level}\n\n"
        
        for disease, prob in zip(top_3_diseases, top_3_probs):
            result += f"🏥 Disease: {disease}\n"
            result += f"Probability: {prob*100:.1f}%\n\n"
            
            description = description_df[description_df['Disease'] == disease]['Description'].iloc[0]
            result += f"📝 Description:\n{description}\n\n"
            
            precautions = precaution_df[precaution_df['Disease'] == disease].iloc[0][1:].tolist()
            result += "⚕️ Recommended Precautions:\n"
            for i, precaution in enumerate(precautions, 1):
                result += f"{i}. {precaution}\n"
            
            result += "\n" + "─"*50 + "\n\n"
        
        # Add warning message with better formatting
        result += "\n⚠️ IMPORTANT MEDICAL DISCLAIMER:\n"
        result += "• This is a prototype system for educational purposes only\n"
        result += "• Not a substitute for professional medical advice\n"
        result += "• Please consult a healthcare provider for proper diagnosis and treatment\n"
        
        return prob_chart, result
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}\nPlease try again or contact support."
        return None, error_msg

# Create Gradio interface with enhanced styling
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # 🏥 Disease Prediction System
        ### Select symptoms for AI-powered health analysis
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Symptom selection tabs
            with gr.Tabs():
                category_inputs = []
                for category, symptoms in symptom_categories.items():
                    with gr.Tab(f"📋 {category}"):
                        category_input = gr.CheckboxGroup(
                            choices=sorted(symptoms),
                            label=f"{category} Symptoms",
                            value=[],
                            info=f"Select applicable {category.lower()} symptoms"
                        )
                        category_inputs.append(category_input)
            
            # Add symptom counter
            symptom_counter = gr.Markdown(
                f"Selected: 0/{MAX_SYMPTOMS} symptoms",
                elem_id="symptom_counter"
            )
            
            predict_btn = gr.Button(
                "🔍 Analyze Symptoms",
                variant="primary",
                size="lg"
            )
            
            # Add clear button
            clear_btn = gr.Button(
                "🗑️ Clear All",
                variant="secondary",
                size="sm"
            )
            
        with gr.Column(scale=2):
            prob_plot = gr.Plot(label="Disease Probability Distribution")
            output = gr.Textbox(
                label="Detailed Analysis",
                lines=15,
                show_copy_button=True
            )
    
    # Add quick selection examples
    with gr.Accordion("📚 Common Symptom Combinations", open=False):
        example_grid = gr.Dataset(
            components=[gr.CheckboxGroup(visible=False)],
            samples=[
                [["continuous_sneezing", "runny_nose", "chills"]],
                [["itching", "skin_rash", "nodal_skin_eruptions"]],
                [["stomach_pain", "acidity", "ulcers_on_tongue"]]
            ],
            headers=["Cold Symptoms", "Skin Infection", "Digestive Issues"],
            type="index"
        )
    
    # Event handlers
    def update_counter(*selections):
        total = len(combine_symptoms(*selections))
        color = "red" if total > MAX_SYMPTOMS else "green"
        return f"Selected: {total}/{MAX_SYMPTOMS} symptoms"
    
    def clear_selections():
        return [[] for _ in category_inputs]
    
    # Update symptom counter when selections change
    for input_box in category_inputs:
        input_box.change(
            update_counter,
            inputs=category_inputs,
            outputs=symptom_counter
        )
    
    # Clear button functionality
    clear_btn.click(
        clear_selections,
        outputs=category_inputs
    )
    
    # Prediction button
    predict_btn.click(
        fn=predict_disease,
        inputs=category_inputs,
        outputs=[prob_plot, output]
    )
    
    # Add helpful information
    gr.Markdown(
        """
        ### 📋 How to Use:
        1. Select symptoms from each category tab
        2. Click 'Analyze Symptoms' to get predictions
        3. Review the probability chart and detailed analysis
        4. Use 'Clear All' to start over
        
        ### ℹ️ Features:
        - Multi-category symptom selection
        - Probability visualization
        - Severity assessment
        - Detailed disease information
        - Recommended precautions
        """
    )

# Launch the interface
if __name__ == "__main__":
    iface.launch()
