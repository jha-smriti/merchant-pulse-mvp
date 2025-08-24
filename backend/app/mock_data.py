import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import joblib
import os

def generate_mock_data(n_samples=1000):
    """Generates a fake dataset of merchant features."""
    np.random.seed(42)
    data = {
        'merchant_id': [f'M{1000 + i}' for i in range(n_samples)],
        'avg_transaction_value': np.random.normal(50, 15, n_samples).round(2),
        'transaction_count_30d': np.random.poisson(100, n_samples),
        'chargeback_count_30d': np.random.poisson(1, n_samples),
        'days_since_first_transaction': np.random.randint(100, 2000, n_samples),
    }
    df = pd.DataFrame(data)
    # Create a target variable: 1 for high risk, 0 for low risk
    # Risk increases with high chargebacks and high transaction value
    df['chargeback_ratio'] = df['chargeback_count_30d'] / df['transaction_count_30d']
    df['high_risk'] = (
        (df['chargeback_ratio'] > 0.02) |
        ((df['avg_transaction_value'] > 80) & (df['chargeback_count_30d'] > 1))
    ).astype(int)
    # Drop the target-based feature before training
    df = df.drop('chargeback_ratio', axis=1)
    return df

def train_and_save_model(df):
    """Trains a simple model and saves it."""
    # Features to use for prediction
    feature_columns = [
        'avg_transaction_value',
        'transaction_count_30d',
        'chargeback_count_30d',
        'days_since_first_transaction'
    ]
    X = df[feature_columns]
    y = df['high_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and the feature names
    os.makedirs('../../model', exist_ok=True)
    joblib.dump(model, '../../model/model.pkl')
    with open('../../model/features.json', 'w') as f:
        json.dump(feature_columns, f)

    print(f"Model trained. Accuracy: {model.score(X_test, y_test):.2f}")
    return model, feature_columns

# Generate data and train the model if this script is run directly
if __name__ == "__main__":
    print("Generating mock data...")
    df = generate_mock_data()
    print("Training model...")
    train_and_save_model(df)
    print("Model saved to /model directory.")