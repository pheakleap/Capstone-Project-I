import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import joblib

def validate_model_predictions(model, X_test, y_test, feature_names, target_names):
    """Comprehensive validation of model predictions"""
    #basic prediction accuracy
    y_pred = model.predict(X_test)
    print("Basic Performance:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Analyze prediction confidence
    y_pred_proba = model.predict_proba(X_test)
    confidence_scores = np.max(y_pred_proba, axis=1)
    
    print("\nConfidence Analysis:")
    print(f"Mean confidence: {confidence_scores.mean():.2f}")
    print(f"Min confidence: {confidence_scores.min():.2f}")
    print(f"Max confidence: {confidence_scores.max():.2f}")
    
    # Analyze misclassifications
    misclassified = X_test[y_pred != y_test]
    misclassified_true = y_test[y_pred != y_test]
    misclassified_pred = y_pred[y_pred != y_test]
    
    print("\nMisclassification Analysis:")
    for true, pred in zip(misclassified_true, misclassified_pred):
        print(f"True: {target_names[true]}, Predicted: {target_names[pred]}")

# Run validation
model = joblib.load('../../models/ensemble_models.joblib')['voting_classifier']
processed_data = joblib.load('../../data/processed/processed_data.joblib')

validate_model_predictions(
    model,
    processed_data['X_test'],
    processed_data['y_test'],
    processed_data['feature_names'],
    processed_data['target_names']
)