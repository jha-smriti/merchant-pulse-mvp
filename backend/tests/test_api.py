import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.main import app

client = TestClient(app)

class TestMerchantPulseAPI:
    """Test suite for the Merchant Pulse API"""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "features" in data
        assert data["version"] == "2.0.0"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "2.0.0"
    
    def test_get_merchants(self):
        """Test fetching merchant list"""
        response = client.get("/merchants/")
        assert response.status_code == 200
        data = response.json()
        assert "merchant_ids" in data
        assert isinstance(data["merchant_ids"], list)
        assert len(data["merchant_ids"]) > 0
    
    def test_get_merchant_details_valid(self):
        """Test getting details for a valid merchant"""
        # First get a valid merchant ID
        merchants_response = client.get("/merchants/")
        merchant_ids = merchants_response.json()["merchant_ids"]
        
        if merchant_ids:
            merchant_id = merchant_ids[0]
            response = client.get(f"/merchants/{merchant_id}/details")
            assert response.status_code == 200
            data = response.json()
            assert data["merchant_id"] == merchant_id
            assert "details" in data
            assert "derived_metrics" in data
            assert "timestamp" in data
    
    def test_get_merchant_details_invalid(self):
        """Test getting details for an invalid merchant"""
        response = client.get("/merchants/INVALID_ID/details")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    @patch('app.main.get_ml_prediction')
    @patch('app.main.get_ai_explanation')
    @patch('app.main.get_recommendations')
    async def test_predict_endpoint_with_ai(self, mock_recommendations, mock_explanation, mock_ml):
        """Test prediction endpoint with AI features"""
        # Setup mocks
        mock_ml.return_value = {
            "risk_score": 0.75,
            "risk_category": "HIGH",
            "confidence": 0.85,
            "explanation": {"risk_reason": "High chargeback count"}
        }
        mock_explanation.return_value = "This merchant shows high risk due to elevated chargeback rates."
        mock_recommendations.return_value = [
            "Implement additional fraud detection",
            "Monitor transaction patterns closely"
        ]
        
        # Get a valid merchant ID
        merchants_response = client.get("/merchants/")
        merchant_ids = merchants_response.json()["merchant_ids"]
        
        if merchant_ids:
            merchant_id = merchant_ids[0]
            request_data = {
                "merchant_id": merchant_id,
                "include_ai_explanation": True,
                "include_recommendations": True
            }
            
            response = client.post("/predict/", json=request_data)
            assert response.status_code == 200
            data = response.json()
            
            assert data["merchant_id"] == merchant_id
            assert "risk_score" in data
            assert "risk_category" in data
            assert "confidence" in data
            assert "ai_explanation" in data
            assert "recommendations" in data
            assert "timestamp" in data
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        # Get some merchant IDs
        merchants_response = client.get("/merchants/")
        merchant_ids = merchants_response.json()["merchant_ids"][:3]  # Test with first 3
        
        request_data = {
            "merchant_ids": merchant_ids,
            "include_ai_explanation": False
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "total_processed" in data
        assert "total_requested" in data
        assert "results" in data
        assert data["total_requested"] == len(merchant_ids)
    
    def test_batch_prediction_limit(self):
        """Test batch prediction with too many merchants"""
        # Create a list with more than 100 merchants
        merchant_ids = [f"M{i:04d}" for i in range(150)]
        
        request_data = {
            "merchant_ids": merchant_ids,
            "include_ai_explanation": False
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 400
        data = response.json()
        assert "Maximum 100 merchant IDs allowed" in data["detail"]
    
    def test_analytics_dashboard(self):
        """Test analytics dashboard endpoint"""
        response = client.get("/analytics/dashboard")
        assert response.status_code == 200
        data = response.json()
        
        assert "summary" in data
        assert "risk_distribution" in data
        assert "timestamp" in data
        
        summary = data["summary"]
        assert "total_merchants" in summary
        assert "high_risk_merchants" in summary
        assert "avg_chargeback_rate" in summary
        
        risk_dist = data["risk_distribution"]
        assert "low_risk" in risk_dist
        assert "medium_risk" in risk_dist
        assert "high_risk" in risk_dist
    
    @patch('httpx.AsyncClient.post')
    async def test_chat_endpoint(self, mock_post):
        """Test chat endpoint with mocked RAG service"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test AI response",
            "conversation_id": "test-conversation-123",
            "sources": [],
            "timestamp": "2024-01-01T00:00:00"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        request_data = {
            "message": "What are the main risk factors?",
            "merchant_id": None,
            "conversation_id": None
        }
        
        response = client.post("/chat", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "response" in data
        assert "conversation_id" in data
        assert "sources" in data
        assert "timestamp" in data

class TestInputValidation:
    """Test input validation and error handling"""
    
    def test_predict_empty_merchant_id(self):
        """Test prediction with empty merchant ID"""
        request_data = {
            "merchant_id": "",
            "include_ai_explanation": True
        }
        
        response = client.post("/predict/", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_context_type(self):
        """Test prediction with invalid context type"""
        merchants_response = client.get("/merchants/")
        merchant_ids = merchants_response.json()["merchant_ids"]
        
        if merchant_ids:
            request_data = {
                "merchant_id": merchant_ids[0],
                "context_type": "invalid_context"
            }
            
            response = client.post("/predict/", json=request_data)
            assert response.status_code == 422  # Validation error
    
    def test_chat_empty_message(self):
        """Test chat with empty message"""
        request_data = {
            "message": "",
            "merchant_id": None
        }
        
        response = client.post("/chat", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_chat_message_too_long(self):
        """Test chat with message that's too long"""
        request_data = {
            "message": "x" * 1001,  # Exceeds 1000 character limit
            "merchant_id": None
        }
        
        response = client.post("/chat", json=request_data)
        assert response.status_code == 422  # Validation error

class TestLegacyCompatibility:
    """Test legacy endpoint compatibility"""
    
    def test_legacy_predict_endpoint(self):
        """Test legacy prediction endpoint still works"""
        merchants_response = client.get("/merchants/")
        merchant_ids = merchants_response.json()["merchant_ids"]
        
        if merchant_ids:
            request_data = {
                "merchant_id": merchant_ids[0]
            }
            
            response = client.post("/predict_legacy/", json=request_data)
            assert response.status_code == 200
            data = response.json()
            
            assert "merchant_id" in data
            assert "risk_score" in data
            assert "risk_reason" in data
            assert "details" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])