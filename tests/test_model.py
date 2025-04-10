import unittest
from src.models.disease_classifier import DiseaseClassifier
import pandas as pd

class TestDiseaseClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = DiseaseClassifier()
        
    def test_prediction_shape(self):
        # Test code here
        pass