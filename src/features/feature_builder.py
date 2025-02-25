import pandas as pd 
import numpy as np

class FeatureBuilder:
    def __init__(self, severity_data):
        self.severity_map = self._create_severity_map(severity_data)
    
    def _create_severity_map(self, severity_data):
        return dict(zip(severity_data['Symptom'], severity_data['weight']))

    def build_features(self, symptoms_data):
        enhanced_features = []
        return pd.Dataframe(enhanced_features)