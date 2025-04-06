import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap

# Load data
df = pd.read_csv('D:/Term7/Capstone-Project-I/data/processed/preprocessed_data.csv')

# Separate features and target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Split into train/test before applying SMOTE!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check balance after SMOTE
print("After SMOTE:", pd.Series(y_train_resampled).value_counts())

# Build models with proper pipelines
lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
rf_pipeline = RandomForestClassifier(n_estimators=100, random_state=42)
catboost_pipeline = cb.CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0)
lgb_pipeline = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
nb_pipeline = GaussianNB()
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', lr_pipeline),
    ('rf', rf_pipeline),
    ('catboost', catboost_pipeline),
    ('lgb', lgb_pipeline),
    ('nb', nb_pipeline),
    ('svm', svm_pipeline)
], voting='soft', n_jobs=-1)

# Hyperparameter grid
param_grid = {
    'lr__logisticregression__C': [0.1, 1, 10],
    'rf__n_estimators': [100, 200],
    'catboost__iterations': [300, 500],
    'lgb__learning_rate': [0.01, 0.1],
    'svm__svc__C': [0.1, 1, 10]
}

# Grid Search CV
grid_search = GridSearchCV(voting_clf, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# SHAP Analysis (optional but awesome)
# Pick a tree-based model from the voting ensemble (e.g., CatBoost)
explainer = shap.TreeExplainer(catboost_pipeline)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")

