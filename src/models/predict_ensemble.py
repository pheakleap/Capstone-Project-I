import joblib
import pandas as pd
import numpy as np
import os

# Define the expected path to the model file relative to this script's location
# Use os.path.abspath to ensure robustness across environments
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up two levels from src/models/ to the project root, then into models/
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_models_improved.joblib')
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_models.joblib') # Define fallback path

def load_ensemble_model(model_path=DEFAULT_MODEL_PATH):
    """
    Loads the trained ensemble model and associated data from a joblib file.

    Args:
        model_path (str): The path to the .joblib file containing the model results.

    Returns:
        tuple: A tuple containing (model, feature_names, target_names)
               Returns (None, None, None) if loading fails.
    """
    try:
        # Check if the primary model path exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at '{model_path}'")
            # Try the defined fallback path
            if os.path.exists(FALLBACK_MODEL_PATH):
                print(f"Attempting to load fallback model: '{FALLBACK_MODEL_PATH}'")
                model_path = FALLBACK_MODEL_PATH
            else:
                 print(f"Error: Fallback model file also not found at '{FALLBACK_MODEL_PATH}'")
                 return None, None, None

        # Load the joblib file
        results = joblib.load(model_path)
        model = results.get('voting_classifier')
        feature_names = results.get('feature_names')
        target_names = results.get('target_names')

        # Validate loaded components
        if model is None or feature_names is None or target_names is None:
            print("Error: Model file is missing required keys ('voting_classifier', 'feature_names', 'target_names').")
            return None, None, None

        print(f"Model loaded successfully from '{model_path}'")
        return model, feature_names, target_names

    except FileNotFoundError:
        # This might occur if the path becomes invalid between check and load (rare)
        print(f"Error: FileNotFoundError encountered unexpectedly for path: '{model_path}'")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None, None

def prepare_input_features(symptoms_list, feature_names):
    """
    Converts a list of symptoms into a binary feature vector based on the model's feature names.

    Args:
        symptoms_list (list): A list of symptom strings provided by the user.
        feature_names (list): The list of all possible symptom features the model was trained on.

    Returns:
        np.ndarray: A 1D numpy array (shape [1, n_features]) representing the input features.
                    Returns None if feature_names is invalid or no symptoms match.
    """
    if not feature_names:
        print("Error: Feature names list is empty or invalid.")
        return None

    # Create a dictionary for quick lookup of feature indices
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    # Initialize the feature vector with zeros
    input_vector = np.zeros(len(feature_names))

    # Set the corresponding feature to 1 for each symptom in the input list
    symptoms_found_count = 0
    unknown_symptoms = []
    for symptom in symptoms_list:
        # Basic cleaning: lower case and replace common separators with space
        cleaned_symptom = symptom.lower().replace('_', ' ').replace('-', ' ').strip()
        # Attempt exact match first
        if cleaned_symptom in feature_index:
            input_vector[feature_index[cleaned_symptom]] = 1
            symptoms_found_count += 1
        else:
            # Try matching with spaces removed as well (e.g., "skin rash" vs "skinrash")
            symptom_no_space = cleaned_symptom.replace(' ', '')
            found_no_space = False
            # Iterate through feature names to find a potential match without spaces
            for f_name in feature_names:
                 if f_name.replace(' ', '') == symptom_no_space:
                      input_vector[feature_index[f_name]] = 1
                      symptoms_found_count += 1
                      found_no_space = True
                      break # Stop checking once a match is found for this symptom
            if not found_no_space:
                 unknown_symptoms.append(symptom) # Keep track of original symptom name

    # Report any symptoms that couldn't be matched
    if unknown_symptoms:
        print(f"Warning: The following input symptoms were not found in the model's feature list: {unknown_symptoms}")

    # If no valid symptoms were found at all, return None
    if symptoms_found_count == 0:
         print("Error: None of the provided symptoms matched the model's features.")
         return None

    # Reshape to 2D array (1 sample, n_features) as expected by sklearn models
    return input_vector.reshape(1, -1)


def predict_disease(symptoms_input, model, feature_names, target_names, top_n=3):
    """
    Predicts the disease based on a list of symptoms using the loaded ensemble model.

    Args:
        symptoms_input (list): A list of symptom strings.
        model: The loaded ensemble model (VotingClassifier).
        feature_names (list): List of feature names used during training.
        target_names (list): List of target disease names.
        top_n (int): The number of top predictions to return.

    Returns:
        list: A list of tuples, where each tuple is (disease_name, probability),
              sorted by probability in descending order. Returns an empty list on error.
    """
    if model is None or not feature_names or not target_names:
        print("Error: Model or associated data (feature/target names) is not loaded or invalid.")
        return []

    # Prepare the feature vector from the input symptoms
    input_features = prepare_input_features(symptoms_input, feature_names)

    if input_features is None:
        print("Error: Could not prepare input features from the provided symptoms.")
        return []

    try:
        # Predict probabilities for each class (disease)
        probabilities = model.predict_proba(input_features)[0] # Get probabilities for the first (only) sample

        # Combine disease names with their probabilities
        results = list(zip(target_names, probabilities))

        # Sort by probability in descending order
        results.sort(key=lambda item: item[1], reverse=True)

        # Return the top N predictions
        return results[:top_n]

    except AttributeError:
         print("Error: Model does not support 'predict_proba'. Ensure it's a classifier trained for probabilities (e.g., VotingClassifier with voting='soft').")
         return []
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return []

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    print("--- Testing Ensemble Model Prediction Script ---")

    # 1. Load the model
    loaded_model, f_names, t_names = load_ensemble_model()

    if loaded_model:
        # 2. Define example symptoms
        # Use symptoms that are likely in your feature_names list
        # Check '../../data/processed/symptom_names_v2.txt' or feature_names in the saved model
        # Example for Fungal infection (assuming these features exist)
        example_symptoms = ['itching', 'skin rash', 'nodal skin eruptions', 'dischromic patches']
        # More generic example
        # example_symptoms = ['fever', 'cough', 'headache']
        # Example with unknown symptoms
        # example_symptoms = ['invalid symptom', 'another bad one', 'itching']

        print(f"\nInput Symptoms: {example_symptoms}")

        # 3. Prepare features (optional, for debugging)
        # input_vec = prepare_input_features(example_symptoms, f_names)
        # if input_vec is not None:
        #     print(f"\nPrepared Feature Vector (shape {input_vec.shape}):\n{input_vec}")
        #     # Find indices where value is 1
        #     active_indices = np.where(input_vec[0] == 1)[0]
        #     print(f"Indices set to 1: {list(active_indices)}")
        #     print(f"Corresponding feature names: {[f_names[i] for i in active_indices]}")


        # 4. Make prediction
        predictions = predict_disease(example_symptoms, loaded_model, f_names, t_names, top_n=5)

        # 5. Print results
        if predictions:
            print("\nTop Predicted Diseases:")
            for disease, probability in predictions:
                print(f"- {disease}: {probability:.4f}")
        else:
            print("\nPrediction failed or returned no results.")
    else:
        print("\nCould not load the model. Prediction cannot proceed.")
