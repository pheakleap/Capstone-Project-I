import gradio as gr
import json
import joblib

# Load symptom encodings (label encoded)
try:
    with open("D:/Term7/Capstone-Project-I/data/processed/name_symptom.json", "r") as f:
        symptom_encodings = json.load(f)
except FileNotFoundError:
    print("Error: symptom_encodings.json not found.")
    symptom_encodings = {}
print(symptom_encodings)