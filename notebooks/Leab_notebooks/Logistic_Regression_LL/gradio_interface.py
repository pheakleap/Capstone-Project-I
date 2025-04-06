import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Load the trained model from the file
model = joblib.load(r'D:\CADT University\CADT-Y3\CodeAllSub\Capstone\Capstone-Project-I\notebooks\Leab_notebooks\Encoding\logistic_regression_model.pkl')
disease_encoder = joblib.load(r'D:\CADT University\CADT-Y3\CodeAllSub\Capstone\Capstone-Project-I\notebooks\Leab_notebooks\Models\disease_encoder.pkl')  # Disease label encoder
symptom_encoder = joblib.load(r'D:\CADT University\CADT-Y3\CodeAllSub\Capstone\Capstone-Project-I\notebooks\Leab_notebooks\Models\symptom_encoder.pkl')  # Symptom encoder

import joblib
import pandas as pd
import numpy as np

# Load the pre-trained Logistic Regression model and encoders
model = joblib.load('logistic_regression_model.pkl')  # Path to your model
disease_encoder = joblib.load('disease_encoder.pkl')  # Disease encoder (if needed)
symptom_encoder = joblib.load('symptom_encoder.pkl')  # Symptom encoder (if needed)

# Define the symptom names that the model expects
symptom_names = [
    'fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level',
    'blurred_and_distorted_vision', 'obesity', 'excessive_hunger', 'increased_appetite', 'polyuria',
    'nausea', 'vomiting', 'dizziness', 'headache', 'fever', 'cough', 'shortness_of_breath',
    'chest_pain', 'back_pain', 'joint_pain', 'stomach_pain', 'skin_rash', 'itchiness', 'swelling', 'chills',
    'sore_throat', 'runny_nose', 'loss_of_appetite', 'sleep_disorder', 'bloody_urine', 'muscle_weakness',
    'pale_skin', 'rapid_heart_rate', 'yellow_skin', 'diarrhea', 'constipation'
]

# Define the disease mapping from encoded labels to disease names
disease_mapping = {
    0: 'Chicken pox',
    1: 'Common Cold',
    2: 'Dengue',
    3: 'Diabetes',
    4: 'Hepatitis B',
    5: 'Hepatitis D',
    6: 'Hepatitis E',
    7: 'Hyperthyroidism',
    8: 'Hypoglycemia',
    9: 'Hypothyroidism',
    10: 'Migraine',
    11: 'Pneumonia',
    12: 'Tuberculosis',
    13: 'Typhoid',
    14: 'Hepatitis A'
}

# Manually input symptoms (replace with real data)
# Example input: Symptoms present are fatigue, weight_loss, and excessive_hunger
input_symptoms = ['fatigue', 'weight_loss', 'excessive_hunger']

# Create a binary list where 1 indicates the presence of a symptom and 0 indicates its absence
symptom_values = [1 if symptom in input_symptoms else 0 for symptom in symptom_names]

# Convert the symptoms to a DataFrame
input_data = pd.DataFrame([symptom_values], columns=symptom_names)

# Ensure the input data is in the correct format (numeric, filled with 0 if necessary)
input_data_encoded = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Make predictions using the Logistic Regression model
predicted_class = model.predict(input_data_encoded)

# Map the predicted class to the disease name
predicted_disease = disease_mapping.get(predicted_class[0], "Unknown disease")

# Output the result
print(f"The predicted disease is: {predicted_disease}")
