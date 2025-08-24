import joblib
import json
import pandas as pd
import numpy as np

# Load the model and feature list
try:
    model = joblib.load('../../model/model.pkl')
    with open('../../model/features.json', 'r') as f:
        feature_columns = json.load(f)
except FileNotFoundError:
    print("Model files not found. Please run mock_data.py first.")
    model = None
    feature_columns = []

def predict_risk(merchant_data):
    """Predicts risk for a merchant and returns the score and top reason."""
    if model is None:
        return 0.5, "Model not loaded"

    # Prepare the features for prediction
    X_input = merchant_data[feature_columns]
    prediction_proba = model.predict_proba(X_input)[0]
    risk_score = prediction_proba[1]

    # Get feature importance for the top reason
    feature_importance = model.feature_importances_
    top_feature_index = np.argmax(feature_importance)
    top_feature_name = feature_columns[top_feature_index]
    top_feature_value = X_input.iloc[0][top_feature_index]

    reason = f"High {top_feature_name.replace('_', ' ')} ({top_feature_value})"

    return round(risk_score, 2), reason