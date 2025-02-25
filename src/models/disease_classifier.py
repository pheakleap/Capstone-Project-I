import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseaseClassifier:
    """
    Disease classification model using Logistic Regression.
    Part of Capstone Project I - Disease Prediction System.
    """
    
    def __init__(self):
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        logger.info("DiseaseClassifier initialized")
    
    def fit(self, X, y):
        """Train the model with given features and labels."""
        try:
            y_encoded = self.label_encoder.fit_transform(y)
            self.model.fit(X, y_encoded)
            logger.info("Model training completed successfully")
            return self
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, symptoms):
        prediction = self.model.predict(symptoms)
        return self.label_encoder.inverse_transform(prediction)[0]
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)