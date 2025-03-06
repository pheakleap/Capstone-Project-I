import gradio as gr
import pandas as pd
import joblib

# Load your model here (replace with your actual model loading)
# model = joblib.load("your_model.pkl")

# Load the data from the CSV file
df = pd.read_csv("your_data.csv") # Replace 'your_data.csv' with the actual filename

# Extract unique symptoms
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)] # Assuming up to 17 symptoms

all_symptoms = set()
for col in symptom_columns:
    if col in df.columns: # Check if the column exists in the dataframe
        all_symptoms.update(df[col].unique())

all_symptoms = sorted(list(all_symptoms)) # Convert to list and sort

# Create exactly 17 dropdowns
inputs = [
    gr.Dropdown(choices=["none"] + all_symptoms, label=f"Symptom {i+1}")
    for i in range(17)
]

# Function to predict disease (replace with your actual prediction logic)
def predict_disease(*selected_symptoms):
    selected = [s.strip() for s in selected_symptoms if s and s != "none"]

    if len(selected) < 3:
        return "Please select at least 3 symptoms."

    # Ensure exactly 17 symptoms (fill remaining with 'none')
    while len(selected) < 17:
        selected.append("none")

    # Create DataFrame with correct feature names
    input_df = pd.DataFrame([selected], columns=symptom_columns)

    # Predict disease (replace with your actual prediction)
    # prediction = model.predict(input_df)
    # return prediction[0]
    return "Prediction Placeholder" # Replace with your actual model prediction.

# Gradio Interface
interface = gr.Interface(
    fn=predict_disease,
    inputs=inputs,
    outputs=gr.Textbox(label="Predicted Disease"),
    title="Disease Prediction from Symptoms",
    description="Select at least 3 symptoms. If you don't have all 17, leave the rest as 'none'."
)

interface.launch()