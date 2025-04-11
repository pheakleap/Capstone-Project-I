import gradio as gr
import joblib
import pandas as pd
import numpy as np

model = joblib.load("D:/Term7/Capstone-Project-I/src/models/best_ensemble_model.joblib")
feature_encoders = joblib.load("D:/Term7/Capstone-Project-I/src/models/symptom_encoder.joblib")
target_encoder = joblib.load("D:/Term7/Capstone-Project-I/src/models/disease_encoder.joblib")

# Define function to clean and preprocess the input data
def clean_and_encode(features):
    df = pd.DataFrame([features], columns=[f'feature_{i+1}' for i in range(17)])

    # Handle "none" as null values
    df.replace("none", np.nan, inplace=True)

    # Encode the features using the loaded feature encoders
    for i in range(17):  # Assuming 17 features
        feature_name = f'feature_{i+1}'
        if df[feature_name].isnull().any():
            continue  # Skip encoding if it's null (since it might be handled by model)

        # Use the appropriate LabelEncoder for each feature
        df[feature_name] = feature_encoders[feature_name].transform(df[feature_name])

    return df

# Define prediction function
def predict(features):
    # Preprocess the features
    processed_data = clean_and_encode(features)

    # Make prediction with the model (probabilities)
    prediction = model.predict_proba(processed_data)
    return prediction[0]

# Create Gradio Interface
def feature_input():
    # Define the possible values for each feature (display as names)
    feature_options = {
        f"feature_{i+1}": [f"Value {j}" for j in range(130)]  # Replace with actual values for each feature
        for i in range(17)
    }
    
    # Add "none" option for null values
    for key in feature_options:
        feature_options[key].append("none")

    # Create inputs dynamically for each feature
    inputs = [
        gr.Dropdown(label=f"Feature {i+1}", choices=feature_options[f"feature_{i+1}"], type="str") 
        for i in range(17)
    ]

    # Prediction output as probability
    output = gr.Label()

    return inputs, output

# Create a Gradio interface
inputs, output = feature_input()
iface = gr.Interface(fn=predict, inputs=inputs, outputs=output, live=True)

# Launch Gradio
iface.launch()
