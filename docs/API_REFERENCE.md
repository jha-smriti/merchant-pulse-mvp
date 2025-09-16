# Merchant Pulse Enterprise - API Reference

## Overview

This document provides a comprehensive reference for all API endpoints available in the Merchant Pulse Enterprise platform. The platform consists of three main services:

- **Backend Service** (Port 8000): Main business logic and data management
- **ML Service** (Port 8002): Machine learning model serving and training
- **RAG Service** (Port 8003): AI-powered knowledge base and conversational interface

## Authentication

All API endpoints (except health checks and public endpoints) require authentication using JWT tokens.

### Authentication Header
```
Authorization: Bearer <jwt_token>
```

### Rate Limiting
- Standard endpoints: 100 requests/minute
- ML endpoints: 50 requests/minute  
- RAG endpoints: 30 requests/minute

## Backend Service API

### Base URL: `http://localhost:8000`

#### Health & Monitoring

##### GET /health
Health check endpoint for the backend service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "2.0.0"
}
```

##### GET /metrics
Prometheus metrics endpoint for monitoring.

**Response:** Prometheus metrics format

#### Merchant Management

##### GET /merchants/
Retrieve a list of all merchant IDs.

**Response:**
```json
{
  "merchant_ids": [
    "M1001",
    "M1002", 
    "M1003"
  ]
}
```

##### GET /merchants/{merchant_id}/details
Get detailed information about a specific merchant.

**Parameters:**
- `merchant_id` (string): The merchant identifier

**Response:**
```json
{
  "merchant_id": "M1001",
  "details": {
    "avg_transaction_value": 125.50,
    "transaction_count_30d": 150,
    "chargeback_count_30d": 2,
    "days_since_first_transaction": 365
  },
  "derived_metrics": {
    "chargeback_rate": 0.0133,
    "avg_daily_transactions": 5.0,
    "total_volume_30d": 18825.0,
    "risk_indicators": {
      "high_chargeback_rate": false,
      "high_value_transactions": false,
      "new_merchant": false
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Error Responses:**
- `404`: Merchant not found

#### Risk Assessment

##### POST /predict/
Get risk assessment for a single merchant with AI enhancements.

**Request Body:**
```json
{
  "merchant_id": "M1001",
  "include_ai_explanation": true,
  "include_recommendations": true,
  "context_type": "risk_analysis"
}
```

**Parameters:**
- `merchant_id` (string): Merchant identifier
- `include_ai_explanation` (boolean, optional): Include AI explanation (default: true)
- `include_recommendations` (boolean, optional): Include recommendations (default: true)
- `context_type` (string, optional): Context type for AI analysis (default: "risk_analysis")
  - Options: "risk_analysis", "compliance", "fraud_detection"

**Response:**
```json
{
  "merchant_id": "M1001",
  "risk_score": 0.25,
  "risk_category": "LOW",
  "risk_reason": "Low chargeback count (2)",
  "ai_explanation": "This merchant demonstrates low risk profile with consistent transaction patterns and minimal chargebacks. The transaction volume is within normal ranges and shows steady growth.",
  "recommendations": [
    "Continue monitoring transaction patterns",
    "Maintain current fraud prevention measures",
    "Consider offering volume discounts for continued growth"
  ],
  "confidence": 0.87,
  "details": {
    "avg_transaction_value": 125.50,
    "transaction_count_30d": 150,
    "chargeback_count_30d": 2,
    "days_since_first_transaction": 365,
    "chargeback_rate": 0.0133
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### POST /predict/batch
Perform batch risk assessment for multiple merchants.

**Request Body:**
```json
{
  "merchant_ids": ["M1001", "M1002", "M1003"],
  "include_ai_explanation": false
}
```

**Parameters:**
- `merchant_ids` (array): List of merchant IDs (max 100)
- `include_ai_explanation` (boolean, optional): Include AI explanations (default: false)

**Response:**
```json
{
  "total_processed": 3,
  "total_requested": 3,
  "results": [
    {
      "merchant_id": "M1001",
      "risk_score": 0.25,
      "risk_category": "LOW",
      "confidence": 0.87,
      "timestamp": "2024-01-01T12:00:00Z"
    }
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### POST /predict_legacy/
Legacy prediction endpoint for backward compatibility.

**Request Body:**
```json
{
  "merchant_id": "M1001"
}
```

**Response:**
```json
{
  "merchant_id": "M1001",
  "risk_score": 0.25,
  "risk_reason": "Low chargeback count (2)",
  "details": {
    "avg_transaction_value": 125.50,
    "transaction_count_30d": 150,
    "chargeback_count_30d": 2
  }
}
```

#### AI Chat Interface

##### POST /chat
Interact with the AI assistant for merchant intelligence.

**Request Body:**
```json
{
  "message": "What are the main risk factors for merchant M1001?",
  "merchant_id": "M1001",
  "conversation_id": "chat_123456"
}
```

**Parameters:**
- `message` (string): User message (1-1000 characters)
- `merchant_id` (string, optional): Merchant context for the conversation
- `conversation_id` (string, optional): Conversation identifier for context

**Response:**
```json
{
  "response": "Based on the analysis of merchant M1001, the main risk factors include...",
  "conversation_id": "chat_123456",
  "sources": [
    {
      "content": "Risk assessment guidelines...",
      "metadata": {
        "source": "knowledge_base",
        "category": "risk_analysis"
      }
    }
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Analytics

##### GET /analytics/dashboard
Get dashboard analytics and metrics.

**Response:**
```json
{
  "summary": {
    "total_merchants": 1000,
    "high_risk_merchants": 45,
    "high_risk_percentage": 4.5,
    "avg_chargeback_rate": 0.015,
    "total_transactions_30d": 50000,
    "total_volume_30d": 2500000.00
  },
  "risk_distribution": {
    "low_risk": 850,
    "medium_risk": 105,
    "high_risk": 45
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## ML Service API

### Base URL: `http://localhost:8002`

#### Health & Monitoring

##### GET /health
Health check for ML service.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 3,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### GET /metrics
Prometheus metrics for ML service monitoring.

#### Model Management

##### GET /models
List all available models.

**Response:**
```json
{
  "models": ["latest", "random_forest_20240101", "xgboost_20240102"],
  "metadata": {
    "latest": {
      "model_type": "RandomForestClassifier",
      "accuracy": 0.85,
      "created_at": "2024-01-01T00:00:00Z",
      "features": ["avg_transaction_value", "transaction_count_30d"]
    }
  }
}
```

##### GET /model/{version}/info
Get detailed information about a specific model version.

**Parameters:**
- `version` (string): Model version identifier

**Response:**
```json
{
  "version": "latest",
  "metadata": {
    "model_type": "RandomForestClassifier",
    "accuracy": 0.85,
    "created_at": "2024-01-01T00:00:00Z",
    "features": ["avg_transaction_value", "transaction_count_30d"]
  },
  "feature_importance": {
    "avg_transaction_value": 0.35,
    "transaction_count_30d": 0.25,
    "chargeback_count_30d": 0.40
  }
}
```

#### Prediction

##### POST /predict
Generate risk predictions using ML models.

**Request Body:**
```json
{
  "data": [
    {
      "merchant_id": "M1001",
      "avg_transaction_value": 125.50,
      "transaction_count_30d": 150,
      "chargeback_count_30d": 2,
      "days_since_first_transaction": 365,
      "industry_category": "retail",
      "geographical_region": "US",
      "payment_methods": ["card", "paypal"],
      "seasonal_patterns": {"Q1": 0.8, "Q2": 1.2}
    }
  ],
  "model_version": "latest",
  "include_explanation": true,
  "include_confidence": true
}
```

**Response:**
```json
[
  {
    "merchant_id": "M1001",
    "risk_score": 0.25,
    "risk_category": "LOW",
    "confidence": 0.87,
    "explanation": {
      "top_risk_factors": [
        {
          "factor": "chargeback_count_30d",
          "value": 2,
          "importance": 0.40,
          "contribution": 0.15
        }
      ],
      "chargeback_rate": 0.0133,
      "risk_indicators": {
        "high_chargeback_rate": false,
        "high_transaction_volume": false,
        "new_merchant": false
      }
    },
    "model_version": "latest",
    "timestamp": "2024-01-01T12:00:00Z"
  }
]
```

#### Model Training

##### POST /train
Train a new machine learning model.

**Request Body:**
```json
{
  "training_data": [
    {
      "merchant_id": "M1001",
      "avg_transaction_value": 125.50,
      "transaction_count_30d": 150,
      "chargeback_count_30d": 2,
      "days_since_first_transaction": 365
    }
  ],
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5
  },
  "cross_validation": true
}
```

**Parameters:**
- `training_data` (array): Training dataset
- `model_type` (string): Model type ("random_forest", "xgboost", "lightgbm", "neural_network")
- `hyperparameters` (object, optional): Model hyperparameters
- `cross_validation` (boolean, optional): Enable cross-validation (default: true)

**Response:**
```json
{
  "message": "Model training started",
  "model_type": "random_forest",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## RAG Service API

### Base URL: `http://localhost:8003`

#### Health & Monitoring

##### GET /health
Health check for RAG service.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "embeddings": true,
    "vectorstore": true,
    "llm": true
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### GET /metrics
Prometheus metrics for RAG service.

#### Knowledge Base Query

##### POST /query
Query the knowledge base using RAG.

**Request Body:**
```json
{
  "query": "What are the PCI DSS compliance requirements?",
  "merchant_id": "M1001",
  "context_type": "compliance",
  "include_sources": true,
  "max_results": 5
}
```

**Parameters:**
- `query` (string): Search query (1-1000 characters)
- `merchant_id` (string, optional): Merchant context
- `context_type` (string, optional): Query context ("general", "risk_analysis", "compliance", "fraud_detection")
- `include_sources` (boolean, optional): Include source documents (default: true)
- `max_results` (integer, optional): Maximum results to return (1-20, default: 5)

**Response:**
```json
{
  "answer": "PCI DSS (Payment Card Industry Data Security Standard) compliance requirements include...",
  "sources": [
    {
      "content": "PCI DSS requires merchants to...",
      "metadata": {
        "category": "compliance",
        "source": "internal_guidelines"
      },
      "relevance_score": 0.9
    }
  ],
  "confidence": 0.85,
  "query_id": "query_1704110400",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Conversational AI

##### POST /chat
Chat with the AI assistant.

**Request Body:**
```json
{
  "message": "Explain fraud detection patterns",
  "conversation_id": "conv_123456",
  "merchant_context": {
    "merchant_id": "M1001",
    "avg_transaction_value": 125.50,
    "chargeback_count_30d": 2
  }
}
```

**Parameters:**
- `message` (string): Chat message (1-1000 characters)
- `conversation_id` (string, optional): Conversation identifier
- `merchant_context` (object, optional): Merchant data for context

**Response:**
```json
{
  "response": "Fraud detection patterns include velocity fraud, account takeover, and friendly fraud...",
  "conversation_id": "conv_123456",
  "sources": [
    {
      "content": "Fraud detection patterns overview...",
      "metadata": {
        "category": "fraud_patterns",
        "source": "fraud_team"
      }
    }
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Document Management

##### POST /documents/upload
Upload and index a document in the knowledge base.

**Request Body:** `multipart/form-data`
- `file`: Document file (PDF, TXT)
- `title`: Document title
- `category`: Document category ("compliance", "risk_guidelines", "fraud_patterns", "industry_reports")
- `tags`: JSON array of tags

**Response:**
```json
{
  "message": "Document uploaded successfully",
  "filename": "compliance_guide.pdf",
  "chunks_added": 15,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Conversation Management

##### GET /conversations/{conversation_id}/history
Get conversation history.

**Parameters:**
- `conversation_id` (string): Conversation identifier

**Response:**
```json
{
  "conversation_id": "conv_123456",
  "history": [
    {
      "role": "user",
      "content": "What are fraud patterns?",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "role": "assistant", 
      "content": "Fraud patterns include...",
      "timestamp": "2024-01-01T12:00:05Z"
    }
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### DELETE /conversations/{conversation_id}
Clear conversation history.

**Parameters:**
- `conversation_id` (string): Conversation identifier

**Response:**
```json
{
  "message": "Conversation cleared",
  "conversation_id": "conv_123456",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Error Handling

### Standard Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "MERCHANT_NOT_FOUND",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### HTTP Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `422`: Validation Error
- `429`: Too Many Requests
- `500`: Internal Server Error
- `502`: Bad Gateway
- `503`: Service Unavailable

### Rate Limiting Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704110460
```

## SDK Examples

### Python Example

```python
import requests

# Risk assessment
response = requests.post('http://localhost:8000/predict/', 
    json={
        'merchant_id': 'M1001',
        'include_ai_explanation': True
    },
    headers={'Authorization': 'Bearer <token>'}
)

risk_data = response.json()
print(f"Risk Score: {risk_data['risk_score']}")
```

### JavaScript Example

```javascript
// AI Chat
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer <token>'
    },
    body: JSON.stringify({
        message: 'What are the risk factors for this merchant?',
        merchant_id: 'M1001'
    })
});

const chatData = await response.json();
console.log('AI Response:', chatData.response);
```

### cURL Examples

```bash
# Get merchant details
curl -X GET "http://localhost:8000/merchants/M1001/details" \
  -H "Authorization: Bearer <token>"

# Risk prediction
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"merchant_id": "M1001", "include_ai_explanation": true}'

# Query knowledge base
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "PCI DSS requirements", "context_type": "compliance"}'
```

## Webhooks

### Risk Alert Webhook

When a high-risk merchant is detected, the system can send webhooks to configured endpoints.

**Webhook Payload:**
```json
{
  "event": "high_risk_detected",
  "merchant_id": "M1001",
  "risk_score": 0.85,
  "risk_category": "HIGH",
  "timestamp": "2024-01-01T12:00:00Z",
  "webhook_id": "wh_123456"
}
```

## Changelog

### Version 2.0.0 (Current)
- Added AI-powered explanations and recommendations
- Implemented RAG service for conversational AI
- Enhanced ML service with multiple algorithms
- Added comprehensive monitoring and metrics
- Implemented cloud-native deployment features

### Version 1.0.0
- Basic risk assessment functionality
- Simple ML model integration
- React frontend
- Docker containerization

---

For more information, see the [Technical Documentation](TECHNICAL_DOCUMENTATION.md).