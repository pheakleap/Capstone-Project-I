import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Optional
import networkx as nx

# Load models and data
results = joblib.load('../models/ensemble_models.joblib')
voting_clf = results['voting_classifier']
feature_names = results['feature_names']
target_names = results['target_names']

# Load additional data
description_df = pd.read_csv('../data/raw/symptom_Description.csv')
precaution_df = pd.read_csv('../data/raw/symptom_precaution.csv')
symptom_analysis = joblib.load('../data/processed/symptom_analysis.joblib')
symptom_severity_df = pd.read_csv('../data/raw/Symptom-severity.csv')
# print("Available columns:", symptom_severity_df.columns.tolist())

# category
symptom_categories = {
    "General": [
        " fatigue", " lethargy", " malaise", " weakness_in_limbs", " weight_loss", " weight_gain",
        " high_fever", " mild_fever", " sweating", " chills", " dehydration", " shivering"
    ],
    "Pain & Discomfort": [
        " headache", " stomach_pain", " abdominal_pain", " back_pain", " chest_pain", 
        " joint_pain", " muscle_pain", " neck_pain", " knee_pain", " hip_joint_pain",
        " stiff_neck", " muscle_wasting", " cramps", " pain_behind_the_eyes"
    ],
    "Skin & External": [
        "itching", " skin_rash", " nodal_skin_eruptions", " yellowish_skin",
        " bruising", " red_spots_over_body", " dischromic _patches", " toxic_look_(typhos)",
        " yellow_crust_ooze", " pus_filled_pimples", " blackheads", " scurring",
        " skin_peeling", " blister", " red_sore_around_nose"
    ],
    "Respiratory": [
        " cough", " breathlessness", " runny_nose", " congestion", " sinus_pressure",
        " phlegm", " throat_irritation", " continuous_sneezing", " mucoid_sputum",
        " rusty_sputum", " blood_in_sputum"
    ],
    "Digestive": [
        " nausea", " vomiting", " diarrhoea", " constipation", " stomach_bleeding",
        " distention_of_abdomen", " acidity", " ulcers_on_tongue", " loss_of_appetite",
        " excessive_hunger", " indigestion", " passage_of_gases", " internal_itching",
        " bloody_stool"
    ],
    "Neurological": [
        " dizziness", " unsteadiness", " lack_of_concentration", " altered_sensorium",
        " depression", " irritability", " slurred_speech", " visual_disturbances",
        " anxiety", " loss_of_balance", " loss_of_smell", " mood_swings", " coma"
    ],
    "Cardiovascular": [
        " chest_pain", " palpitations", " fast_heart_rate", " swollen_blood_vessels",
        " prominent_veins_on_calf", " cold_hands_and_feets", " fluid_overload"
    ],
    "Urinary & Diabetes": [
        " burning_micturition", " spotting_ urination", " dark_urine", " yellow_urine",
        " polyuria", " bladder_discomfort", " continuous_feel_of_urine", 
        " irregular_sugar_level", " increased_appetite", " foul_smell_of urine"
    ],
    "Eyes & Face": [
        " puffy_face_and_eyes", " sunken_eyes", " redness_of_eyes",
        " watering_from_eyes", " yellowing_of_eyes", " blurred_and_distorted_vision"
    ],
    "Musculoskeletal": [
        " muscle_weakness", " movement_stiffness", " swelling_joints",
        " painful_walking", " swollen_legs", " swollen_extremeties",
        " swelling_of_stomach"
    ],
    "Medical History": [
        " family_history", " history_of_alcohol_consumption", " extra_marital_contacts",
        " receiving_blood_transfusion", " receiving_unsterile_injections"
    ],
    "Others": []  # Placeholder for remaining symptoms
}

# Add remaining symptoms to "Others" category
all_categorized = set([symptom for symptoms in symptom_categories.values() for symptom in symptoms])
symptom_categories["Others"] = [s for s in feature_names if s not in all_categorized]

# Validate that all symptoms exist in feature_names
for category, symptoms in symptom_categories.items():
    for symptom in symptoms:
        if symptom not in feature_names:
            print(f"Warning: '{symptom}' in category '{category}' is not in model features")




# Now modify the severity info creation based on actual columns
severity_info = dict(zip(
    symptom_severity_df['Symptom'].str.strip().str.lower().str.replace(' ', '_'),
    symptom_severity_df['weight']  
))

def create_prediction_chart(diseases: List[str], probabilities: List[float]) -> go.Figure:
    """Create an enhanced probability chart with fixed scale and better text visibility"""
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6']  # Green, Blue, Red, Yellow, Purple
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities * 100,
            y=diseases,
            orientation='h',
            marker_color=colors[:len(diseases)],
            text=[f'{p:.1f}%' for p in probabilities * 100],
            textposition='auto',
            textfont=dict(
                color='white',  # Make percentage text white
                size=14
            )
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Disease Probability Distribution',
            'font': {'color': 'white', 'size': 16}  
        },
        xaxis_title={
            'text': 'Probability (%)',
            'font': {'color': 'white', 'size': 14}  
        },
        yaxis_title={
            'text': 'Disease',
            'font': {'color': 'white', 'size': 14}
        },
        xaxis=dict(
            range=[0, 100],
            tickmode='linear',
            tick0=0,
            dtick=20,
            ticksuffix='%',
            tickfont={'color': 'white'},  
            gridcolor='rgba(255, 255, 255, 0.1)' 
        ),
        yaxis=dict(
            tickfont={'color': 'white'}  
        ),
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}  
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    
    return fig

def create_symptom_network(selected_symptoms: List[str]) -> go.Figure:
    """Create interactive network visualization of related symptoms"""
    # Use symptom correlation data to show relationships
    corr_matrix = symptom_analysis['correlation_matrix']
    
    # Create nodes and edges for the network
    nodes = set(selected_symptoms)
    edges = []
    
    # Add related symptoms (those with correlation > 0.3)
    for symptom in selected_symptoms:
        related = corr_matrix[symptom][corr_matrix[symptom] > 0.3].index.tolist()
        for rel in related:
            if rel in feature_names:  
                nodes.add(rel)
                correlation = corr_matrix[symptom][rel]
                edges.append((symptom, rel, abs(correlation)))

    # Create the network layout using a spring layout
    pos = nx.spring_layout(list(nodes))

    fig = go.Figure()
    
    # Add edges (connections between symptoms)
    for start, end, weight in edges:
        fig.add_trace(go.Scatter(
            x=[pos[start][0], pos[end][0]],
            y=[pos[start][1], pos[end][1]],
            mode='lines',
            line=dict(width=weight*5, color='rgba(128, 128, 128, 0.5)'),
            hoverinfo='none'
        ))
    
    # Add nodes (symptoms)
    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]
    
    # Highlight selected symptoms in a different color
    node_colors = ['red' if node in selected_symptoms else 'blue' for node in nodes]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=20, color=node_colors),
        text=list(nodes),
        textposition='bottom center',
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Symptom Relationship Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig

def calculate_severity_score(symptoms: List[str]) -> float:
    """Calculate severity score using the symptom severity dataset"""
    try:
        total_weight = 0
        for symptom in symptoms:
            # Clean symptom name to match severity dataset
            clean_symptom = symptom.strip().lower().replace(' ', '_')
            # Get weight, default to 1 if not found
            weight = severity_info.get(clean_symptom, 1)
            total_weight += weight
        
        # Calculate severity score
        max_possible_weight = 5 * len(symptoms)  
        severity_score = (total_weight / max_possible_weight) * 100
        
        return min(100, severity_score)
        
    except Exception as e:
        print(f"Error calculating severity: {str(e)}")
        return 50  

def predict_disease(*category_symptoms) -> Tuple[go.Figure, go.Figure, go.Figure, str]:
    try:
        # Combine all selected symptoms
        all_symptoms = []
        for symptoms in category_symptoms:
            if symptoms:
                all_symptoms.extend(symptoms)
        all_symptoms = list(set(all_symptoms))
        
        if not all_symptoms:
            return None, None, None, "Please select at least one symptom."
        
        # Create feature vector with proper feature names
        feature_vector = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        for symptom in all_symptoms:
            if symptom in feature_names:
                feature_vector[symptom] = 1
        
        # Get top 5 predictions instead of 3
        probabilities = voting_clf.predict_proba(feature_vector)[0]
        top_5_idx = np.argsort(probabilities)[-5:][::-1] 
        top_5_diseases = [target_names[i] for i in top_5_idx]  
        top_5_probs = probabilities[top_5_idx]  
        
        # Create visualizations
        pred_chart = create_prediction_chart(top_5_diseases, top_5_probs)  
        symptom_net = create_symptom_network(all_symptoms)
        
        # Calculate severity
        severity = calculate_severity_score(all_symptoms)
        
        # Create severity gauge
        severity_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=severity,
            title={'text': "Severity Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': severity
                }
            }
        ))
        
        # Format output text
        output = "📊 Analysis Results\n\n"
        output += f"Severity Level: {'🔴 High' if severity > 70 else '🟡 Medium' if severity > 30 else '🟢 Low'} ({severity:.1f}%)\n\n"
        
        # Show top 5 diseases in output
        for disease, prob in zip(top_5_diseases, top_5_probs):
            output += f"🏥 Disease: {disease}\n"
            output += f"Probability: {prob*100:.1f}%\n"
            
            desc = description_df[description_df['Disease'] == disease]['Description'].iloc[0]
            output += f"\nDescription:\n{desc}\n\n"
            
            precautions = precaution_df[precaution_df['Disease'] == disease].iloc[0][1:].tolist()
            output += "Recommended Precautions:\n"
            for i, precaution in enumerate(precautions, 1):
                output += f"{i}. {precaution}\n"
            
            output += "\n" + "─"*50 + "\n\n"
        
        return pred_chart, symptom_net, severity_gauge, output
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, None, f"An error occurred: {str(e)}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # 🏥 Advanced Disease Prediction System
        ### Select symptoms for AI-powered health analysis
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Symptom selection with categories
            with gr.Tabs():
                category_inputs = []
                for category, symptoms in symptom_categories.items():
                    with gr.Tab(f"📋 {category}"):
                        category_input = gr.CheckboxGroup(
                            choices=sorted(symptoms),
                            label=f"{category} Symptoms",
                            value=[]
                        )
                        category_inputs.append(category_input)
            
            # Search box for symptoms
            symptom_search = gr.Textbox(
                label="🔍 Search Symptoms",
                placeholder="Type to search symptoms..."
            )
            
            predict_btn = gr.Button("Analyze Symptoms", variant="primary")
            clear_btn = gr.Button("Clear All")
        
        with gr.Column(scale=2):
            # Output visualizations
            with gr.Tabs():
                with gr.Tab("Predictions"):
                    prob_plot = gr.Plot(label="Disease Probabilities")
                with gr.Tab("Symptom Network"):
                    network_plot = gr.Plot(label="Related Symptoms")
                with gr.Tab("Severity"):
                    severity_plot = gr.Plot(label="Severity Score")
            
            output = gr.Textbox(
                label="Detailed Analysis",
                lines=15,
                show_copy_button=True
            )
    
    # Example cases
    with gr.Accordion("📚 Common Cases", open=False):
        gr.Examples(
            examples=[
                [["fever", "cough", "fatigue"]],
                [["skin_rash", "itching"]],
                [["headache", "nausea", "dizziness"]]
            ],
            inputs=category_inputs,
            outputs=[prob_plot, network_plot, severity_plot, output],
            fn=predict_disease,
            label="Common Symptom Combinations"
        )
    
    # Event handlers
    predict_btn.click(
        fn=predict_disease,
        inputs=category_inputs,
        outputs=[
            prob_plot,      
            network_plot,   
            severity_plot, 
            output       
        ]
    )
    
    clear_btn.click(
        fn=lambda: [[] for _ in category_inputs],
        outputs=category_inputs
    )

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)
    # print("Available features in model:")
    # print(sorted(feature_names))