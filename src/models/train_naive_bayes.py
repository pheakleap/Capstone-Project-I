import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model():
    # Define file paths (relative to project root)
    data_path = 'data/raw/'
    model_save_path = 'models/naive_bayes/'

    # Create model save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    try:
        df_train = pd.read_csv(os.path.join(data_path, 'dataset.csv'))
    except Exception as e:
        print(f"Error loading dataset.csv: {e}")
        return # Exit if main dataset can't be loaded

    df_precautions = pd.read_csv(os.path.join(data_path, 'symptom_precaution.csv'))
    df_severity = pd.read_csv(os.path.join(data_path, 'Symptom-severity.csv'))
    df_description = pd.read_csv(os.path.join(data_path, 'symptom_Description.csv'))
    print("Datasets loaded.")

    # Preprocess Training Data
    print("Preprocessing training data...")
    # Identify symptom columns (assuming they are named Symptom_1, Symptom_2, etc.)
    symptom_cols = [col for col in df_train.columns if col.startswith('Symptom_')]

    # Fill NaN values with a placeholder (e.g., empty string) before processing
    df_train[symptom_cols] = df_train[symptom_cols].fillna('')

    # --- Binarize Symptoms ---
    # Combine symptoms for each row into a list, removing placeholders and duplicates
    # Also strip whitespace from symptom names
    symptoms_per_row = df_train[symptom_cols].apply(lambda row: list(set(str(s).strip() for s in row if s and pd.notna(s) and str(s).strip())), axis=1)


    # Use MultiLabelBinarizer to create binary features for each unique symptom
    mlb = MultiLabelBinarizer()
    X_binarized = mlb.fit_transform(symptoms_per_row)
    unique_symptoms = list(mlb.classes_) # Get the list of unique symptoms (features)

    # Create a new DataFrame with binarized features
    X = pd.DataFrame(X_binarized, columns=unique_symptoms)

    # Target variable
    y = df_train['Disease']
    print(f"Number of unique symptoms found: {len(unique_symptoms)}")

    # Encode the target variable 'Disease'
    le_disease = LabelEncoder()
    y_encoded = le_disease.fit_transform(y)
    disease_classes = le_disease.classes_
    print(f"Number of disease classes: {len(disease_classes)}")
    print("Preprocessing complete.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Train Naive Bayes Model
    print("Training Naive Bayes model...")
    nb_model = BernoulliNB()
    nb_model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate Model (optional, for confirmation)
    y_pred_nb = nb_model.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"\nNaive Bayes Model Accuracy on Test Split: {accuracy_nb:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_nb, target_names=disease_classes, zero_division=0)) # Added zero_division

    # Save Artifacts
    print(f"Saving model and artifacts to: {model_save_path}")
    joblib.dump(nb_model, os.path.join(model_save_path, 'naive_bayes_model.joblib'))
    joblib.dump(le_disease, os.path.join(model_save_path, 'disease_label_encoder.joblib'))
    # Save the *unique* symptoms list derived from binarization
    joblib.dump(unique_symptoms, os.path.join(model_save_path, 'symptoms_list.joblib'))

    # Prepare and save auxiliary data
    df_precautions.set_index('Disease', inplace=True)
    df_description.set_index('Disease', inplace=True)
    # Clean severity symptom names (remove leading/trailing spaces) and set index
    df_severity['Symptom'] = df_severity['Symptom'].str.strip()
    df_severity.set_index('Symptom', inplace=True)
    df_precautions.to_csv(os.path.join(model_save_path, 'precautions_processed.csv'))
    df_description.to_csv(os.path.join(model_save_path, 'description_processed.csv'))
    df_severity.to_csv(os.path.join(model_save_path, 'severity_processed.csv'))
    print("Artifacts saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
