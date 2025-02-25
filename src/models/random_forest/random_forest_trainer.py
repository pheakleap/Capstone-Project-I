from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self):
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.pipeline.fit(X_train, y_train)
        return self.pipeline