import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("D:/Term7/Capstone-Project-I/src/models/random_forest.pkl")

# Load datasets for encoding
before_encode = pd.read_csv("D:/Term7/Capstone-Project-I/data/processed/before_encode.csv")
encode_data = pd.read_csv("D:/Term7/Capstone-Project-I/data/processed/encode_data.csv")

# Extract feature columns and target labels
X = before_encode.drop(columns=["Disease"])
y = before_encode["Disease"]

X_encode = encode_data.drop(columns=["Disease"])
y_encode = encode_data["Disease"]

# Create encoding dictionaries
encoding_map = {col: {val: idx for idx, val in enumerate(X[col].unique())} for col in X.columns}
disease_mapping = dict(enumerate(y.unique()))  # Map numbers back to disease names

# Function to preprocess user input
def preprocess_input(*user_inputs):
    """
    Convert user inputs into the encoded format required by the trained model.
    """
    input_dict = {col: [encoding_map[col].get(value, -1)] for col, value in zip(X.columns, user_inputs)}
    input_df = pd.DataFrame(input_dict)
    
    # Handle missing values or unknown inputs
    input_df = input_df.replace(-1, 0)  # Replace unknown values with default (e.g., 0)
    
    return input_df

# Prediction function
def predict_disease(*user_inputs):
    """
    Takes raw user input, encodes it, and predicts the disease using the trained model.
    """
    input_df = preprocess_input(*user_inputs)
    prediction = model.predict(input_df)[0]
    
    # Decode predicted disease number to actual disease name
    predicted_disease = disease_mapping.get(prediction, "Unknown Disease")
    
    return f"Predicted Disease: {predicted_disease}"

# Create Gradio interface
inputs = [gr.Dropdown(choices=list(X[col].unique()), label=col) for col in X.columns]
output = gr.Textbox(label="Prediction Result")

demo = gr.Interface(fn=predict_disease, inputs=inputs, outputs=output, title="Disease Prediction System")

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()
