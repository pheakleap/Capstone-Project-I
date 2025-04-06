import gradio as gr
import pandas as pd

# Mock data - Replace with your actual data
SYMPTOMS = ["headache", "fatigue", "fever", "cough", "rash"]  # 150+ symptoms
CATEGORIES = {
    "Pain": ["headache", "chest_pain", "joint_pain"],
    "Respiratory": ["cough", "shortness_of_breath"],
    "General": ["fever", "fatigue"]
}

# Initialize state
selected_symptoms = set()
MAX_SYMPTOMS = 17

def filter_symptoms(search_term):
    return [s for s in SYMPTOMS if search_term.lower() in s.lower()]

def update_interface(search_term, *args):
    global selected_symptoms
    # Update selections
    for symptom, selected in zip(SYMPTOMS, args):
        if selected and len(selected_symptoms) < MAX_SYMPTOMS:
            selected_symptoms.add(symptom)
        elif not selected and symptom in selected_symptoms:
            selected_symptoms.discard(symptom)
    
    # Create selected tags
    tags = " ".join([f"**{s}** ✕" for s in selected_symptoms])
    
    # Create warning
    warning = ""
    if len(selected_symptoms) >= MAX_SYMPTOMS:
        warning = "⚠️ Maximum 17 symptoms allowed"
    
    # Filter search results
    search_results = filter_symptoms(search_term) if search_term else []
    
    return {
        selected_panel: f"**Selected ({len(selected_symptoms)}/17):**\n{tags}",
        warning_component: warning,
        search_dropdown: gr.Dropdown(choices=search_results)
    }

def predict():
    if not selected_symptoms:
        return "**Please select at least 1 symptom**"
    
    # Add your prediction logic here
    return "**Likely Condition:** Migraine\n**Confidence:** 82%"

# Build interface
with gr.Blocks(theme=gr.themes.Soft(), title="Medical Symptom Checker") as demo:
    # Header
    gr.Markdown("# 🩺 Symptom Checker")
    
    # Search row
    with gr.Row():
        search = gr.Textbox(label="Search symptoms", placeholder="Start typing...")
        search_dropdown = gr.Dropdown(label="Matching symptoms", interactive=True)
    
    # Category sections
    with gr.Tabs():
        for category, symptoms in CATEGORIES.items():
            with gr.Tab(category):
                with gr.Row():
                    for symptom in symptoms:
                        gr.Checkbox(symptom, label=symptom.replace("_", " ").title())
    
    # Selected panel
    selected_panel = gr.Markdown("**Selected (0/17):**")
    warning_component = gr.Markdown()
    
    # Prediction
    with gr.Row():
        gr.Button("Clear Selections", variant="secondary")
        predict_btn = gr.Button("Analyze Symptoms", variant="primary")
    
    result = gr.Markdown()

    # Event handling
    search.change(
        update_interface,
        [search] + [comp for comp in demo.blocks.values() if isinstance(comp, gr.Checkbox)],
        [selected_panel, warning_component, search_dropdown]
    )
    
    predict_btn.click(predict, None, result)

demo.launch()