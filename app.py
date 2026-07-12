"""
Credit Card Approval Prediction - Flask Web App
===============================================
Loads trained model from model/ folder and serves predictions via web UI.

Requirements:
    pip install flask pandas numpy scikit-learn

Run:
    python app.py
    Then open browser: http://127.0.0.1:5000
"""

import os
import sys
import pickle
import warnings

# Suppress ALL warnings before any imports
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'feature_metadata.pkl')

# ==========================================
# LOAD MODEL ARTIFACTS
# ==========================================

def load_model_artifacts():
    """Load model, scaler, label encoders, and metadata from model/ folder."""
    
    required_files = {
        'Model': MODEL_PATH,
        'Scaler': SCALER_PATH,
        'Label Encoders': LABEL_ENCODERS_PATH,
        'Metadata': METADATA_PATH
    }
    
    missing = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing.append(f"  - {name}: {path}")
    
    if missing:
        print("\n" + "=" * 60)
        print("❌ MODEL FILES NOT FOUND!")
        print("=" * 60)
        print("Missing files:")
        for m in missing:
            print(m)
        print("\n💡 To fix this:")
        print("   1. First run: python train_model.py")
        print("   2. Then run:  python app.py")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    print("✅ Model artifacts loaded successfully!")
    return model, scaler, label_encoders, metadata

print("Loading model artifacts...")
model, scaler, label_encoders, metadata = load_model_artifacts()
categorical_cols = metadata['categorical_cols']
numerical_cols = metadata['numerical_cols']
feature_columns = metadata['feature_columns']

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_credit_approval(application_data):
    input_df = pd.DataFrame([application_data])
    input_df = input_df[feature_columns]
    input_encoded = input_df.copy()

    for col in categorical_cols:
        if col in input_encoded.columns:
            input_encoded[col] = input_encoded[col].apply(
                lambda x: label_encoders[col].transform([x])[0] 
                if x in label_encoders[col].classes_ else -1
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
# HTML TEMPLATE
# ==========================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Approval Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .form-section { padding: 40px; }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        .form-group { display: flex; flex-direction: column; }
        .form-group label {
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            font-size: 0.95em;
        }
        .form-group input, .form-group select {
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 20px;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .result-section { padding: 0 40px 40px; }
        .result-card {
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            animation: slideUp 0.5s ease;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result-approved {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        .result-rejected {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
        }
        .result-icon { font-size: 4em; margin-bottom: 15px; }
        .result-title { font-size: 2em; font-weight: 700; margin-bottom: 10px; }
        .result-probability { font-size: 1.3em; margin-bottom: 5px; }
        .result-confidence { font-size: 1em; opacity: 0.9; }
        .probability-bars {
            margin-top: 20px;
            display: flex;
            gap: 15px;
            justify-content: center;
        }
        .prob-bar {
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 15px 25px;
        }
        .prob-bar-label { font-size: 0.85em; opacity: 0.8; }
        .prob-bar-value { font-size: 1.5em; font-weight: 700; }
        .error-message {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin: 0 40px 40px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💳 Credit Card Approval Predictor</h1>
            <p>Enter applicant details to predict approval status</p>
        </div>
        <div class="form-section">
            <form method="POST" action="/predict">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Gender</label>
                        <select name="Gender" required>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" name="Age" min="18" max="100" value="35" required>
                    </div>
                    <div class="form-group">
                        <label>Marital Status</label>
                        <select name="Marital_Status" required>
                            <option value="Single">Single</option>
                            <option value="Married" selected>Married</option>
                            <option value="Divorced">Divorced</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Education</label>
                        <select name="Education" required>
                            <option value="High School">High School</option>
                            <option value="Bachelor" selected>Bachelor</option>
                            <option value="Master">Master</option>
                            <option value="PhD">PhD</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Employment Status</label>
                        <select name="Employment_Status" required>
                            <option value="Employed" selected>Employed</option>
                            <option value="Self-Employed">Self-Employed</option>
                            <option value="Unemployed">Unemployed</option>
                            <option value="Retired">Retired</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Annual Income ($)</label>
                        <input type="number" name="Annual_Income" min="0" value="75000" required>
                    </div>
                    <div class="form-group">
                        <label>Years at Current Job</label>
                        <input type="number" name="Years_at_Current_Job" min="0" value="5" required>
                    </div>
                    <div class="form-group">
                        <label>Total Debt ($)</label>
                        <input type="number" name="Total_Debt" min="0" value="20000" required>
                    </div>
                    <div class="form-group">
                        <label>Credit Score</label>
                        <input type="number" name="Credit_Score" min="300" max="850" value="720" required>
                    </div>
                    <div class="form-group">
                        <label>Number of Credit Cards</label>
                        <input type="number" name="Num_Credit_Cards" min="0" value="3" required>
                    </div>
                    <div class="form-group">
                        <label>Number of Loans</label>
                        <input type="number" name="Num_Loans" min="0" value="1" required>
                    </div>
                    <div class="form-group">
                        <label>Has Mortgage?</label>
                        <select name="Has_Mortgage" required>
                            <option value="Yes" selected>Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Has Car Loan?</label>
                        <select name="Has_Car_Loan" required>
                            <option value="Yes">Yes</option>
                            <option value="No" selected>No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Has Default History?</label>
                        <select name="Has_Default_History" required>
                            <option value="Yes">Yes</option>
                            <option value="No" selected>No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Late Payments (12 months)</label>
                        <input type="number" name="Num_Late_Payments_12M" min="0" value="0" required>
                    </div>
                    <div class="form-group">
                        <label>Bank Account Balance ($)</label>
                        <input type="number" name="Bank_Account_Balance" min="0" value="15000" required>
                    </div>
                    <div class="form-group">
                        <label>Property Owned?</label>
                        <select name="Property_Owned" required>
                            <option value="Yes" selected>Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Application Type</label>
                        <select name="Application_Type" required>
                            <option value="New" selected>New</option>
                            <option value="Upgrade">Upgrade</option>
                        </select>
                    </div>
                    <button type="submit" class="submit-btn">🔮 Predict Approval</button>
                </div>
            </form>
        </div>
        {% if result %}
        <div class="result-section">
            {% if result.Approved %}
            <div class="result-card result-approved">
                <div class="result-icon">✅</div>
                <div class="result-title">APPROVED</div>
            {% else %}
            <div class="result-card result-rejected">
                <div class="result-icon">❌</div>
                <div class="result-title">REJECTED</div>
            {% endif %}
                <div class="result-probability">Approval Probability: {{ result.Approval_Probability }}</div>
                <div class="result-confidence">Confidence: {{ result.Confidence }}</div>
                <div class="probability-bars">
                    <div class="prob-bar">
                        <div class="prob-bar-label">Approval</div>
                        <div class="prob-bar-value">{{ result.Approval_Probability }}</div>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-bar-label">Rejection</div>
                        <div class="prob-bar-value">{{ result.Rejection_Probability }}</div>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
        {% if error %}
        <div class="error-message">⚠️ {{ error }}</div>
        {% endif %}
    </div>
</body>
</html>'''

# ==========================================
# FLASK ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        application_data = {
            'Gender': request.form.get('Gender'),
            'Age': int(request.form.get('Age')),
            'Marital_Status': request.form.get('Marital_Status'),
            'Education': request.form.get('Education'),
            'Employment_Status': request.form.get('Employment_Status'),
            'Annual_Income': int(request.form.get('Annual_Income')),
            'Years_at_Current_Job': int(request.form.get('Years_at_Current_Job')),
            'Total_Debt': int(request.form.get('Total_Debt')),
            'Credit_Score': int(request.form.get('Credit_Score')),
            'Num_Credit_Cards': int(request.form.get('Num_Credit_Cards')),
            'Num_Loans': int(request.form.get('Num_Loans')),
            'Has_Mortgage': request.form.get('Has_Mortgage'),
            'Has_Car_Loan': request.form.get('Has_Car_Loan'),
            'Has_Default_History': request.form.get('Has_Default_History'),
            'Num_Late_Payments_12M': int(request.form.get('Num_Late_Payments_12M')),
            'Bank_Account_Balance': int(request.form.get('Bank_Account_Balance')),
            'Property_Owned': request.form.get('Property_Owned'),
            'Application_Type': request.form.get('Application_Type'),
        }
        
        result = predict_credit_approval(application_data)
        return render_template_string(HTML_TEMPLATE, result=result)
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        application_data = request.get_json()
        numeric_fields = ['Age', 'Annual_Income', 'Years_at_Current_Job', 'Total_Debt',
                         'Credit_Score', 'Num_Credit_Cards', 'Num_Loans', 
                         'Num_Late_Payments_12M', 'Bank_Account_Balance']
        for field in numeric_fields:
            if field in application_data:
                application_data[field] = int(application_data[field])
        
        result = predict_credit_approval(application_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("💳 CREDIT CARD APPROVAL PREDICTION APP")
    print("=" * 60)
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)