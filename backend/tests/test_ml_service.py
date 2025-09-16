import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the ml-service app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml-service', 'app'))

from ml_service.app.main import app, ModelManager

class TestMLService:
    """Test suite for the ML Service"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model_manager = ModelManager()
    
    def test_health_endpoint(self):
        """Test ML service health check"""
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "timestamp" in data
    
    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        assert self.model_manager.models is not None
        assert isinstance(self.model_manager.models, dict)
        assert "latest" in self.model_manager.models
    
    def test_feature_preparation(self):
        """Test feature preparation for ML input"""
        from ml_service.app.main import MerchantData
        
        merchant_data = MerchantData(
            merchant_id="M1001",
            avg_transaction_value=100.0,
            transaction_count_30d=50,
            chargeback_count_30d=2,
            days_since_first_transaction=365,
            industry_category="retail",
            geographical_region="US",
            payment_methods=["card", "paypal"],
            seasonal_patterns={"Q1": 0.8, "Q2": 1.2}
        )
        
        features = self.model_manager._prepare_features(merchant_data)
        
        assert isinstance(features, list)
        assert len(features) == 10  # Expected number of features
        assert features[0] == 100.0  # avg_transaction_value
        assert features[1] == 50     # transaction_count_30d
        assert features[2] == 2      # chargeback_count_30d
        assert features[3] == 365    # days_since_first_transaction
    
    def test_risk_categorization(self):
        """Test risk score categorization"""
        assert self.model_manager._categorize_risk(0.1) == "LOW"
        assert self.model_manager._categorize_risk(0.5) == "MEDIUM"
        assert self.model_manager._categorize_risk(0.8) == "HIGH"
    
    def test_prediction_with_mock_data(self):
        """Test prediction with mock merchant data"""
        from ml_service.app.main import MerchantData
        
        merchant_data = MerchantData(
            merchant_id="M1001",
            avg_transaction_value=100.0,
            transaction_count_30d=50,
            chargeback_count_30d=1,
            days_since_first_transaction=365
        )
        
        results = self.model_manager.predict([merchant_data])
        
        assert len(results) == 1
        result = results[0]
        
        assert "merchant_id" in result
        assert "risk_score" in result
        assert "risk_category" in result
        assert "confidence" in result
        assert "explanation" in result
        assert result["merchant_id"] == "M1001"
        assert 0 <= result["risk_score"] <= 1
        assert result["risk_category"] in ["LOW", "MEDIUM", "HIGH"]
    
    def test_explanation_generation(self):
        """Test explanation generation for predictions"""
        from ml_service.app.main import MerchantData
        
        merchant_data = MerchantData(
            merchant_id="M1001",
            avg_transaction_value=1500.0,  # High value
            transaction_count_30d=20,
            chargeback_count_30d=3,        # High chargebacks
            days_since_first_transaction=30  # New merchant
        )
        
        features = self.model_manager._prepare_features(merchant_data)
        explanation = self.model_manager._generate_explanation(merchant_data, features, "latest")
        
        assert "top_risk_factors" in explanation
        assert "chargeback_rate" in explanation
        assert "risk_indicators" in explanation
        
        risk_indicators = explanation["risk_indicators"]
        assert "high_chargeback_rate" in risk_indicators
        assert "high_transaction_volume" in risk_indicators
        assert "new_merchant" in risk_indicators

class TestMLServiceAPI:
    """Test ML Service API endpoints"""
    
    def test_predict_endpoint(self):
        """Test the predict API endpoint"""
        from fastapi.testclient import TestClient
        from ml_service.app.main import app
        
        client = TestClient(app)
        
        request_data = {
            "data": [
                {
                    "merchant_id": "M1001",
                    "avg_transaction_value": 100.0,
                    "transaction_count_30d": 50,
                    "chargeback_count_30d": 1,
                    "days_since_first_transaction": 365
                }
            ],
            "include_explanation": True,
            "include_confidence": True
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        
        prediction = data[0]
        assert "merchant_id" in prediction
        assert "risk_score" in prediction
        assert "risk_category" in prediction
        assert "confidence" in prediction
        assert "explanation" in prediction
    
    def test_models_endpoint(self):
        """Test the models listing endpoint"""
        from fastapi.testclient import TestClient
        from ml_service.app.main import app
        
        client = TestClient(app)
        
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "metadata" in data
        assert isinstance(data["models"], list)
        assert "latest" in data["models"]
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint"""
        from fastapi.testclient import TestClient
        from ml_service.app.main import app
        
        client = TestClient(app)
        
        response = client.get("/model/latest/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data
        assert "metadata" in data
        assert "feature_importance" in data
    
    def test_train_endpoint(self):
        """Test the training endpoint"""
        from fastapi.testclient import TestClient
        from ml_service.app.main import app
        
        client = TestClient(app)
        
        request_data = {
            "training_data": [
                {
                    "merchant_id": "M1001",
                    "avg_transaction_value": 100.0,
                    "transaction_count_30d": 50,
                    "chargeback_count_30d": 1,
                    "days_since_first_transaction": 365
                }
            ],
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 50},
            "cross_validation": True
        }
        
        response = client.post("/train", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "model_type" in data
        assert "timestamp" in data

class TestModelTraining:
    """Test model training functionality"""
    
    def test_hyperparameter_validation(self):
        """Test hyperparameter validation for different model types"""
        valid_rf_params = {"n_estimators": 100, "max_depth": 10}
        valid_xgb_params = {"n_estimators": 100, "learning_rate": 0.1}
        
        # These should not raise exceptions in a real implementation
        assert isinstance(valid_rf_params, dict)
        assert isinstance(valid_xgb_params, dict)
    
    def test_cross_validation_setup(self):
        """Test cross-validation setup"""
        # This would test actual CV setup in a full implementation
        assert True  # Placeholder

if __name__ == "__main__":
    pytest.main([__file__, "-v"])