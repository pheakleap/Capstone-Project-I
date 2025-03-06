import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../data/processed/Top15diseases_encode_label.csv')

X = df.drop(columns=[col for col in df.columns if col.startswith('Disease')])
y = df[[col for col in df.columns if col.startswith('Disease')]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 12, 15],  
    'min_samples_split': [2, 5], 
    'min_samples_leaf': [2, 3], 
    'max_features': ['sqrt', 'log2'],  
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


import joblib
joblib.dump(grid_search, '../src/models/random_forest_modelV168.pkl')
