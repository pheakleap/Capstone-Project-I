class DiseasePredictor:
    def __init__(self, model, severity_data, description_data, precaution_data):
        self.model = model
        self.severity_data = severity_data
        self.description_data = description_data
        self.precaution_data = precaution_data
    
    def predict(self, symptoms):
        """Make predictions with detailed information."""
        # Your prediction code here
        pass