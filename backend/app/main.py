from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import logging
import asyncio
import httpx
import os
from datetime import datetime
import json

# Import local modules
from .model import predict_risk
from .mock_data import generate_mock_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Merchant Pulse API - Enterprise Edition",
    description="Advanced merchant risk intelligence platform with AI/ML and RAG capabilities",
    version="2.0.0"
)

# Security
security = HTTPBearer()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://merchant-pulse.example.com",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mock data into a DataFrame
df = generate_mock_data()

# Service URLs
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'http://ml-service:8000')
RAG_SERVICE_URL = os.getenv('RAG_SERVICE_URL', 'http://rag-service:8000')

class MerchantQuery(BaseModel):
    merchant_id: str

class EnhancedMerchantQuery(BaseModel):
    merchant_id: str
    include_ai_explanation: bool = True
    include_recommendations: bool = True
    context_type: str = Field(default="risk_analysis", regex="^(risk_analysis|compliance|fraud_detection)$")

class ChatQuery(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    merchant_id: Optional[str] = None
    conversation_id: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    merchant_ids: List[str] = Field(..., max_items=100)
    include_ai_explanation: bool = False

class RiskAnalysisResponse(BaseModel):
    merchant_id: str
    risk_score: float
    risk_category: str
    risk_reason: str
    ai_explanation: Optional[str] = None
    recommendations: Optional[List[str]] = None
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]]
    timestamp: datetime

async def get_ml_prediction(merchant_data: Dict[str, Any], include_explanation: bool = True) -> Dict[str, Any]:
    """Get prediction from ML service"""
    try:
        async with httpx.AsyncClient() as client:
            ml_request = {
                "data": [merchant_data],
                "include_explanation": include_explanation,
                "include_confidence": True
            }
            response = await client.post(f"{ML_SERVICE_URL}/predict", json=ml_request, timeout=30.0)
            response.raise_for_status()
            return response.json()[0]  # Return first prediction
    except Exception as e:
        logger.error(f"ML service error: {e}")
        # Fallback to local prediction
        merchant_df = pd.DataFrame([merchant_data])
        prediction, reason = predict_risk(merchant_df)
        return {
            "risk_score": prediction,
            "risk_category": "MEDIUM" if prediction > 0.5 else "LOW",
            "confidence": 0.7,
            "explanation": {"risk_reason": reason}
        }

async def get_ai_explanation(merchant_id: str, risk_data: Dict[str, Any], context_type: str = "risk_analysis") -> Optional[str]:
    """Get AI explanation from RAG service"""
    try:
        async with httpx.AsyncClient() as client:
            query = f"Explain the risk factors for merchant {merchant_id} with risk score {risk_data.get('risk_score', 0):.2f}"
            rag_request = {
                "query": query,
                "merchant_id": merchant_id,
                "context_type": context_type,
                "include_sources": False
            }
            response = await client.post(f"{RAG_SERVICE_URL}/query", json=rag_request, timeout=30.0)
            response.raise_for_status()
            return response.json()["answer"]
    except Exception as e:
        logger.error(f"RAG service error: {e}")
        return None

async def get_recommendations(merchant_id: str, risk_data: Dict[str, Any]) -> List[str]:
    """Get AI-powered recommendations"""
    try:
        async with httpx.AsyncClient() as client:
            query = f"Provide risk mitigation recommendations for merchant {merchant_id} with {risk_data.get('risk_category', 'MEDIUM')} risk"
            rag_request = {
                "query": query,
                "merchant_id": merchant_id,
                "context_type": "risk_analysis",
                "include_sources": False
            }
            response = await client.post(f"{RAG_SERVICE_URL}/query", json=rag_request, timeout=30.0)
            response.raise_for_status()
            recommendations_text = response.json()["answer"]
            
            # Parse recommendations (simple implementation)
            recommendations = []
            for line in recommendations_text.split('\n'):
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    recommendations.append(line.strip()[1:].strip())
            
            return recommendations[:5]  # Limit to 5 recommendations
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return [
            "Monitor transaction patterns closely",
            "Implement additional fraud detection measures",
            "Review chargeback prevention strategies"
        ]

@app.get("/")
def read_root():
    return {
        "message": "Merchant Pulse API - Enterprise Edition",
        "version": "2.0.0",
        "features": [
            "Advanced ML Risk Prediction",
            "AI-Powered Explanations",
            "RAG-based Intelligence",
            "Real-time Analytics",
            "Microservices Architecture"
        ],
        "status": "operational"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "2.0.0"
    }

@app.post("/predict/", response_model=RiskAnalysisResponse)
async def get_merchant_risk(query: EnhancedMerchantQuery):
    """Enhanced API endpoint with AI explanations and recommendations."""
    merchant_data = df[df['merchant_id'] == query.merchant_id]
    if merchant_data.empty:
        raise HTTPException(status_code=404, detail="Merchant ID not found")

    merchant_info = merchant_data.iloc[0].to_dict()
    
    # Prepare data for ML service
    ml_data = {
        "merchant_id": query.merchant_id,
        "avg_transaction_value": merchant_info['avg_transaction_value'],
        "transaction_count_30d": merchant_info['transaction_count_30d'],
        "chargeback_count_30d": merchant_info['chargeback_count_30d'],
        "days_since_first_transaction": merchant_info['days_since_first_transaction']
    }
    
    # Get ML prediction
    ml_result = await get_ml_prediction(ml_data, query.include_ai_explanation)
    
    # Get AI explanation if requested
    ai_explanation = None
    if query.include_ai_explanation:
        ai_explanation = await get_ai_explanation(query.merchant_id, ml_result, query.context_type)
    
    # Get recommendations if requested
    recommendations = None
    if query.include_recommendations:
        recommendations = await get_recommendations(query.merchant_id, ml_result)
    
    return RiskAnalysisResponse(
        merchant_id=query.merchant_id,
        risk_score=ml_result["risk_score"],
        risk_category=ml_result["risk_category"],
        risk_reason=ml_result.get("explanation", {}).get("risk_reason", "Unknown"),
        ai_explanation=ai_explanation,
        recommendations=recommendations,
        confidence=ml_result.get("confidence", 0.5),
        details={
            "avg_transaction_value": merchant_info['avg_transaction_value'],
            "transaction_count_30d": merchant_info['transaction_count_30d'],
            "chargeback_count_30d": merchant_info['chargeback_count_30d'],
            "days_since_first_transaction": merchant_info['days_since_first_transaction'],
            "chargeback_rate": merchant_info['chargeback_count_30d'] / max(merchant_info['transaction_count_30d'], 1)
        },
        timestamp=datetime.now()
    )

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Batch prediction endpoint for multiple merchants"""
    if len(request.merchant_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 merchant IDs allowed")
    
    results = []
    for merchant_id in request.merchant_ids:
        try:
            query = EnhancedMerchantQuery(
                merchant_id=merchant_id,
                include_ai_explanation=request.include_ai_explanation,
                include_recommendations=False
            )
            result = await get_merchant_risk(query)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing merchant {merchant_id}: {e}")
            continue
    
    return {
        "total_processed": len(results),
        "total_requested": len(request.merchant_ids),
        "results": results,
        "timestamp": datetime.now()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(query: ChatQuery):
    """Chat with AI assistant about merchant intelligence"""
    try:
        async with httpx.AsyncClient() as client:
            merchant_context = None
            if query.merchant_id:
                merchant_data = df[df['merchant_id'] == query.merchant_id]
                if not merchant_data.empty:
                    merchant_context = merchant_data.iloc[0].to_dict()
            
            chat_request = {
                "message": query.message,
                "conversation_id": query.conversation_id,
                "merchant_context": merchant_context
            }
            
            response = await client.post(f"{RAG_SERVICE_URL}/chat", json=chat_request, timeout=30.0)
            response.raise_for_status()
            return ChatResponse(**response.json())
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat service temporarily unavailable")

@app.get("/merchants/")
def get_all_merchants():
    """Get a list of all merchant IDs for the frontend dropdown."""
    return {"merchant_ids": df['merchant_id'].tolist()}

@app.get("/merchants/{merchant_id}/details")
def get_merchant_details(merchant_id: str):
    """Get detailed merchant information"""
    merchant_data = df[df['merchant_id'] == merchant_id]
    if merchant_data.empty:
        raise HTTPException(status_code=404, detail="Merchant ID not found")
    
    merchant_info = merchant_data.iloc[0].to_dict()
    
    return {
        "merchant_id": merchant_id,
        "details": merchant_info,
        "derived_metrics": {
            "chargeback_rate": merchant_info['chargeback_count_30d'] / max(merchant_info['transaction_count_30d'], 1),
            "avg_daily_transactions": merchant_info['transaction_count_30d'] / 30,
            "total_volume_30d": merchant_info['avg_transaction_value'] * merchant_info['transaction_count_30d'],
            "risk_indicators": {
                "high_chargeback_rate": merchant_info['chargeback_count_30d'] / max(merchant_info['transaction_count_30d'], 1) > 0.02,
                "high_value_transactions": merchant_info['avg_transaction_value'] > 1000,
                "new_merchant": merchant_info['days_since_first_transaction'] < 90
            }
        },
        "timestamp": datetime.now()
    }

@app.get("/analytics/dashboard")
def get_dashboard_analytics():
    """Get dashboard analytics data"""
    total_merchants = len(df)
    high_risk_merchants = len(df[df['chargeback_count_30d'] > 2])
    avg_chargeback_rate = (df['chargeback_count_30d'] / df['transaction_count_30d'].replace(0, 1)).mean()
    
    return {
        "summary": {
            "total_merchants": total_merchants,
            "high_risk_merchants": high_risk_merchants,
            "high_risk_percentage": (high_risk_merchants / total_merchants) * 100,
            "avg_chargeback_rate": avg_chargeback_rate,
            "total_transactions_30d": df['transaction_count_30d'].sum(),
            "total_volume_30d": (df['avg_transaction_value'] * df['transaction_count_30d']).sum()
        },
        "risk_distribution": {
            "low_risk": len(df[df['chargeback_count_30d'] == 0]),
            "medium_risk": len(df[(df['chargeback_count_30d'] > 0) & (df['chargeback_count_30d'] <= 2)]),
            "high_risk": high_risk_merchants
        },
        "timestamp": datetime.now()
    }

# Original endpoint for backward compatibility
@app.post("/predict_legacy/")
async def get_merchant_risk_legacy(query: MerchantQuery):
    """Legacy API endpoint for backward compatibility."""
    merchant_data = df[df['merchant_id'] == query.merchant_id]
    if merchant_data.empty:
        return {"error": "Merchant ID not found"}

    prediction, top_feature = predict_risk(merchant_data)
    merchant_info = merchant_data.iloc[0].to_dict()

    return {
        "merchant_id": query.merchant_id,
        "risk_score": prediction,
        "risk_reason": top_feature,
        "details": {
            "avg_transaction_value": merchant_info['avg_transaction_value'],
            "transaction_count_30d": merchant_info['transaction_count_30d'],
            "chargeback_count_30d": merchant_info['chargeback_count_30d'],
        }
    }