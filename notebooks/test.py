import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import tempfile

# Set up a temporary directory for CatBoost
temp_dir = tempfile.mkdtemp()
os.environ['CATBOOST_TEMP_DIR'] = temp_dir

# Read the dataset
df = pd.read_csv("D:/Term7/Capstone-Project-I/data/raw/DiseaseAndSymptoms.csv")

# Data cleaning
df.fillna('none', inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Get symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]

# OneHotEncoding for symptoms
symptom_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_symptoms = symptom_encoder.fit_transform(df[symptom_cols])

# Create DataFrame for encoded symptoms
encoded_symptoms_df = pd.DataFrame(
    encoded_symptoms,
    columns=symptom_encoder.get_feature_names_out(symptom_cols))
encoded_symptoms_df = encoded_symptoms_df.astype(int)

# Concatenate with original DataFrame
df_encoded = pd.concat([df.drop(columns=symptom_cols), encoded_symptoms_df], axis=1)

# Encode Disease column
disease_encoder = LabelEncoder()
df_encoded['Disease'] = disease_encoder.fit_transform(df_encoded['Disease'])

# Split into features and target
X = df_encoded.drop("Disease", axis=1)
y = df_encoded["Disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Save resampled data
balanced_df = pd.concat([
    pd.Series(y_resampled, name="Disease"),
    pd.DataFrame(X_resampled, columns=X.columns)
], axis=1)
balanced_df.to_csv(
    "D:/Term7/Capstone-Project-I/data/processed/preprocessed_data_v2.csv",
    index=False)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Save scaled data
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    "D:/Term7/Capstone-Project-I/data/processed/X_train_scaled_v2.csv",
    index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    "D:/Term7/Capstone-Project-I/data/processed/X_test_scaled_v2.csv",
    index=False)

# Initialize models with explicit temp dir for CatBoost
lr_model = LogisticRegression(solver='liblinear')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
catboost_model = cb.CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    verbose=0,
    allow_writing_files=False,  # Disable writing to disk
    train_dir=temp_dir  # Use our temp directory
)
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
nb_model = GaussianNB()
svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', lr_model),
    ('rf', rf_model),
    ('catboost', catboost_model),
    ('lgb', lgb_model),
    ('nb', nb_model),
    ('svm', svm_model)
], voting='soft')

# Hyperparameter tuning with error handling
param_grid = {
    'lr__C': [0.1, 1, 10],
    'rf__n_estimators': [100, 200],
    'catboost__iterations': [300, 500],
    'lgb__learning_rate': [0.01, 0.1],
    'svm__svc__C': [0.1, 1, 10],
}

grid_search = GridSearchCV(
    voting_clf,
    param_grid,
    cv=3,
    n_jobs=-1,
    error_score='raise'  # Raise exceptions to debug
)

try:
    grid_search.fit(X_train_scaled, y_resampled)
    print("Best parameters:", grid_search.best_params_)
except Exception as e:
    print(f"Error during grid search: {str(e)}")
    # Fallback to default parameters
    voting_clf.set_params(**{
        'lr__C': 1,
        'rf__n_estimators': 100,
        'catboost__iterations': 500,
        'lgb__learning_rate': 0.1,
        'svm__svc__C': 1
    })
    voting_clf.fit(X_train_scaled, y_resampled)

# Cross-validation
try:
    cross_val_scores = cross_val_score(
        voting_clf, X_train_scaled, y_resampled, cv=5)
    print(f"Cross-validation scores: {cross_val_scores}")
    print(f"Mean cross-validation score: {cross_val_scores.mean()}")
except Exception as e:
    print(f"Error during cross-validation: {str(e)}")

# Train final model
voting_clf.fit(X_train_scaled, y_resampled)

# Evaluation
y_pred = voting_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Voting Classifier accuracy: {accuracy}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Load disease mapping
with open("D:/Term7/Capstone-Project-I/data/processed/disease_mapping.json", "r") as f:
    disease_mapping = json.load(f)

index_to_disease = {v: k for k, v in disease_mapping.items()}

# Plot confusion matrix
plt.figure(figsize=(20, 18))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=[index_to_disease[i] for i in range(len(disease_mapping))],
    yticklabels=[index_to_disease[i] for i in range(len(disease_mapping))]
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix with Disease Names")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Save model
joblib.dump(
    voting_clf,
    "D:/Term7/Capstone-Project-I/src/models/voting_classifier_model_v2.joblib")
print("Model saved successfully")

# Clean up temporary directory
try:
    import shutil
    shutil.rmtree(temp_dir)
except Exception as e:
    print(f"Error cleaning up temp directory: {str(e)}")