import pandas as pd 

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
    def load_all_datasets(self):
        datasets = {
            'main': pd.read_csv(f'{self.data_path}/../../data/raw/dataset.csv'),
            'severity': pd.read_csv(f'{self.data_path}/../../data/raw/Symptom-severity.csv'),
            'description': pd.read_csv(f'{self.data_path}/../../data/raw/symptom_Description.csv'),
            'precaution': pd.read_csv(f'{self.data_path}/../../data/raw/symptom_precaution.csv'),
        }
        return datasets