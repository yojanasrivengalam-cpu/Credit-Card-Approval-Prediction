"""
Credit Card Approval Prediction - Machine Learning Project
===========================================================
A complete, self-contained Python script for predicting credit card approvals.
Runs without any errors - no external dataset required (uses synthetic data).

Requirements:
    pip install pandas numpy scikit-learn

Author: Generated for ML Project
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# STEP 1: CREATE SYNTHETIC DATASET
# ==========================================

def create_synthetic_credit_data(n_samples=5000, random_state=42):
    """
    Creates a realistic synthetic credit card application dataset.
    In a real project, replace this with: df = pd.read_csv('your_data.csv')
    """
    np.random.seed(random_state)

    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1]),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'Employment_Status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], n_samples, p=[0.6, 0.15, 0.15, 0.1]),
        'Annual_Income': np.random.normal(50000, 20000, n_samples).astype(int),
        'Years_at_Current_Job': np.random.randint(0, 25, n_samples),
        'Total_Debt': np.random.normal(15000, 10000, n_samples).astype(int),
        'Credit_Score': np.random.normal(650, 100, n_samples).astype(int),
        'Num_Credit_Cards': np.random.randint(0, 10, n_samples),
        'Num_Loans': np.random.randint(0, 5, n_samples),
        'Has_Mortgage': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
        'Has_Car_Loan': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'Has_Default_History': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'Num_Late_Payments_12M': np.random.poisson(1, n_samples),
        'Bank_Account_Balance': np.random.normal(5000, 3000, n_samples).astype(int),
        'Property_Owned': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
        'Application_Type': np.random.choice(['New', 'Upgrade'], n_samples, p=[0.8, 0.2]),
    }

    df = pd.DataFrame(data)

    # Ensure realistic constraints
    df['Annual_Income'] = np.clip(df['Annual_Income'], 15000, 200000)
    df['Total_Debt'] = np.clip(df['Total_Debt'], 0, 100000)
    df['Credit_Score'] = np.clip(df['Credit_Score'], 300, 850)
    df['Bank_Account_Balance'] = np.clip(df['Bank_Account_Balance'], 0, 50000)
    df['Years_at_Current_Job'] = np.clip(df['Years_at_Current_Job'], 0, df['Age'] - 18)

    # Generate target variable based on realistic rules
    approval_score = (
        (df['Credit_Score'] - 600) * 2 +
        (df['Annual_Income'] - 50000) / 1000 * 3 +
        df['Years_at_Current_Job'] * 5 -
        df['Total_Debt'] / 1000 * 2 -
        df['Num_Late_Payments_12M'] * 20 -
        (df['Has_Default_History'] == 'Yes').astype(int) * 50 +
        (df['Employment_Status'] == 'Employed').astype(int) * 20 +
        (df['Employment_Status'] == 'Unemployed').astype(int) * -30 +
        df['Bank_Account_Balance'] / 1000 * 2 +
        (df['Property_Owned'] == 'Yes').astype(int) * 15
    )

    approval_score += np.random.normal(0, 30, n_samples)
    df['Approved'] = (approval_score > 0).astype(int)

    return df

# Create dataset
df = create_synthetic_credit_data(n_samples=5000)

print("=" * 60)
print("CREDIT CARD APPROVAL PREDICTION - MACHINE LEARNING")
print("=" * 60)
print(f"Dataset Shape: {df.shape}")
print(f"Approval Rate: {df['Approved'].mean()*100:.1f}%")

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================

X = df.drop('Approved', axis=1)
y = df['Approved']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Encode categorical variables
label_encoders = {}
X_processed = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col])
    label_encoders[col] = le

# ==========================================
# STEP 3: TRAIN-TEST SPLIT & SCALING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ==========================================
# STEP 4: MODEL TRAINING
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest Classifier...")
model.fit(X_train_scaled, y_train)
print("Model trained successfully!")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ==========================================
# STEP 5: MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 Rejected  Approved")
print(f"Actual Rejected    {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"Actual Approved    {cm[1,0]:4d}     {cm[1,1]:4d}")

# ==========================================
# STEP 6: SAVE MODEL ARTIFACTS
# ==========================================

# Create model directory if it doesn't exist
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Save the trained model
model_path = os.path.join(model_dir, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved to: {model_path}")

# Save the scaler
scaler_path = os.path.join(model_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to: {scaler_path}")

# Save the label encoders
label_encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
with open(label_encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"Label encoders saved to: {label_encoders_path}")

# Also save feature metadata for inference
feature_metadata = {
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'feature_columns': list(X_processed.columns)
}
metadata_path = os.path.join(model_dir, 'feature_metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(feature_metadata, f)
print(f"Feature metadata saved to: {metadata_path}")

# ==========================================
# STEP 7: LOAD MODEL FUNCTION (for inference)
# ==========================================

def load_model(model_dir='model'):
    """
    Loads the saved model, scaler, and label encoders from disk.
    
    Returns:
    --------
    tuple: (model, scaler, label_encoders, feature_metadata)
    """
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    label_encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
    metadata_path = os.path.join(model_dir, 'feature_metadata.pkl')
    
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)
    
    with open(label_encoders_path, 'rb') as f:
        loaded_label_encoders = pickle.load(f)
    
    with open(metadata_path, 'rb') as f:
        loaded_metadata = pickle.load(f)
    
    print(f"Model loaded from: {model_dir}")
    return loaded_model, loaded_scaler, loaded_label_encoders, loaded_metadata

# ==========================================
# STEP 8: PREDICTION FUNCTION
# ==========================================

def predict_credit_approval(application_data, model=None, scaler=None, label_encoders=None, 
                            categorical_cols=None, numerical_cols=None):
    """
    Predicts whether a credit card application will be approved.
    If model/scaler/encoders not provided, loads them from disk.

    Parameters:
    -----------
    application_data : dict
        Dictionary containing applicant information
    model : trained model (optional)
    scaler : fitted scaler (optional)
    label_encoders : dict of fitted label encoders (optional)
    categorical_cols : list of categorical column names (optional)
    numerical_cols : list of numerical column names (optional)

    Returns:
    --------
    dict : Prediction result with approval status and confidence
    """
    # Load from disk if not provided
    if model is None or scaler is None or label_encoders is None:
        model, scaler, label_encoders, metadata = load_model()
        categorical_cols = metadata['categorical_cols']
        numerical_cols = metadata['numerical_cols']
    
    input_df = pd.DataFrame([application_data])
    input_encoded = input_df.copy()

    for col in categorical_cols:
        if col in input_encoded.columns:
            input_encoded[col] = input_encoded[col].apply(
                lambda x: label_encoders[col].transform([x])[0] 
                if x in label_encoders[col].classes_ 
                else -1
            )

    input_scaled = input_encoded.copy()
    input_scaled[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    return {
        'Approved': bool(prediction),
        'Confidence': f"{max(probability)*100:.2f}%",
        'Approval_Probability': f"{probability[1]*100:.2f}%",
        'Rejection_Probability': f"{probability[0]*100:.2f}%"
    }

# ==========================================
# STEP 9: EXAMPLE PREDICTIONS
# ==========================================

print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)

# Good applicant
new_application = {
    'Gender': 'Male',
    'Age': 35,
    'Marital_Status': 'Married',
    'Education': 'Bachelor',
    'Employment_Status': 'Employed',
    'Annual_Income': 75000,
    'Years_at_Current_Job': 5,
    'Total_Debt': 20000,
    'Credit_Score': 720,
    'Num_Credit_Cards': 3,
    'Num_Loans': 1,
    'Has_Mortgage': 'Yes',
    'Has_Car_Loan': 'No',
    'Has_Default_History': 'No',
    'Num_Late_Payments_12M': 0,
    'Bank_Account_Balance': 15000,
    'Property_Owned': 'Yes',
    'Application_Type': 'New'
}

result = predict_credit_approval(new_application)
print("\nGood Applicant Prediction:")
for key, value in result.items():
    print(f"  {key}: {value}")

# Risky applicant
risky_application = {
    'Gender': 'Female',
    'Age': 25,
    'Marital_Status': 'Single',
    'Education': 'High School',
    'Employment_Status': 'Unemployed',
    'Annual_Income': 20000,
    'Years_at_Current_Job': 0,
    'Total_Debt': 45000,
    'Credit_Score': 520,
    'Num_Credit_Cards': 7,
    'Num_Loans': 3,
    'Has_Mortgage': 'No',
    'Has_Car_Loan': 'Yes',
    'Has_Default_History': 'Yes',
    'Num_Late_Payments_12M': 4,
    'Bank_Account_Balance': 500,
    'Property_Owned': 'No',
    'Application_Type': 'New'
}

result2 = predict_credit_approval(risky_application)
print("\nRisky Applicant Prediction:")
for key, value in result2.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)