from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
import redis
import json
import logging
from datetime import datetime
import mlflow
import mlflow.sklearn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import asyncio
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Service - Merchant Risk Intelligence",
    description="Advanced ML service for merchant risk prediction with real-time inference",
    version="1.0.0"
)

# Metrics
prediction_counter = Counter('ml_predictions_total', 'Total number of predictions made')
prediction_duration = Histogram('ml_prediction_duration_seconds', 'Time spent on predictions')
model_accuracy = Histogram('ml_model_accuracy', 'Model accuracy metrics')

# Redis connection
redis_client = None
try:
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    redis_client = redis.from_url(redis_url, decode_responses=True)
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")

class MerchantData(BaseModel):
    merchant_id: str
    avg_transaction_value: float = Field(..., gt=0)
    transaction_count_30d: int = Field(..., ge=0)
    chargeback_count_30d: int = Field(..., ge=0)
    days_since_first_transaction: int = Field(..., gt=0)
    industry_category: Optional[str] = "unknown"
    geographical_region: Optional[str] = "unknown"
    payment_methods: Optional[List[str]] = []
    seasonal_patterns: Optional[Dict[str, float]] = {}

class PredictionRequest(BaseModel):
    data: List[MerchantData]
    model_version: Optional[str] = "latest"
    include_explanation: bool = True
    include_confidence: bool = True

class PredictionResponse(BaseModel):
    merchant_id: str
    risk_score: float
    risk_category: str
    confidence: float
    explanation: Dict[str, Any]
    model_version: str
    timestamp: datetime

class ModelTrainingRequest(BaseModel):
    training_data: List[MerchantData]
    model_type: str = Field(..., regex="^(random_forest|xgboost|lightgbm|neural_network)$")
    hyperparameters: Optional[Dict[str, Any]] = {}
    cross_validation: bool = True

class ModelManager:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.model_metadata = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        model_path = os.getenv('MODEL_PATH', '/app/models')
        try:
            # Load default model
            self.models['latest'] = joblib.load(f'{model_path}/risk_model.pkl')
            
            # Load feature importance
            with open(f'{model_path}/feature_importance.json', 'r') as f:
                self.feature_importance['latest'] = json.load(f)
            
            # Load model metadata
            with open(f'{model_path}/model_metadata.json', 'r') as f:
                self.model_metadata['latest'] = json.load(f)
                
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for testing"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        self.models['latest'] = model
        self.feature_importance['latest'] = {
            f'feature_{i}': importance 
            for i, importance in enumerate(model.feature_importances_)
        }
        self.model_metadata['latest'] = {
            'model_type': 'RandomForestClassifier',
            'accuracy': 0.85,
            'created_at': datetime.now().isoformat(),
            'features': [f'feature_{i}' for i in range(10)]
        }
    
    def predict(self, data: List[MerchantData], model_version: str = "latest") -> List[Dict]:
        """Make predictions with the specified model"""
        if model_version not in self.models:
            raise ValueError(f"Model version {model_version} not found")
        
        model = self.models[model_version]
        results = []
        
        for merchant in data:
            # Prepare features
            features = self._prepare_features(merchant)
            
            # Make prediction
            with prediction_duration.time():
                risk_proba = model.predict_proba([features])[0]
                risk_score = risk_proba[1] if len(risk_proba) > 1 else risk_proba[0]
            
            # Calculate confidence
            confidence = max(risk_proba) if len(risk_proba) > 1 else 0.5
            
            # Determine risk category
            risk_category = self._categorize_risk(risk_score)
            
            # Generate explanation
            explanation = self._generate_explanation(merchant, features, model_version)
            
            results.append({
                'merchant_id': merchant.merchant_id,
                'risk_score': float(risk_score),
                'risk_category': risk_category,
                'confidence': float(confidence),
                'explanation': explanation,
                'model_version': model_version,
                'timestamp': datetime.now()
            })
            
            prediction_counter.inc()
        
        return results
    
    def _prepare_features(self, merchant: MerchantData) -> List[float]:
        """Prepare features for model input"""
        features = [
            merchant.avg_transaction_value,
            merchant.transaction_count_30d,
            merchant.chargeback_count_30d,
            merchant.days_since_first_transaction,
            # Add derived features
            merchant.chargeback_count_30d / max(merchant.transaction_count_30d, 1),  # chargeback rate
            merchant.avg_transaction_value * merchant.transaction_count_30d,  # total volume
            # Add categorical features (encoded)
            hash(merchant.industry_category) % 100 / 100,  # simple hash encoding
            hash(merchant.geographical_region) % 100 / 100,
            len(merchant.payment_methods or []),
            sum(merchant.seasonal_patterns.values()) if merchant.seasonal_patterns else 0
        ]
        return features
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels"""
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_explanation(self, merchant: MerchantData, features: List[float], model_version: str) -> Dict:
        """Generate explanation for the prediction"""
        feature_names = [
            'avg_transaction_value', 'transaction_count_30d', 'chargeback_count_30d',
            'days_since_first_transaction', 'chargeback_rate', 'total_volume',
            'industry_category_encoded', 'geographical_region_encoded',
            'payment_methods_count', 'seasonal_score'
        ]
        
        importance = self.feature_importance.get(model_version, {})
        
        # Find top contributing factors
        top_factors = []
        for i, (name, value) in enumerate(zip(feature_names, features)):
            factor_importance = importance.get(name, 0.1)
            contribution = factor_importance * value
            top_factors.append({
                'factor': name,
                'value': value,
                'importance': factor_importance,
                'contribution': contribution
            })
        
        top_factors.sort(key=lambda x: x['contribution'], reverse=True)
        
        return {
            'top_risk_factors': top_factors[:3],
            'chargeback_rate': merchant.chargeback_count_30d / max(merchant.transaction_count_30d, 1),
            'risk_indicators': {
                'high_chargeback_rate': merchant.chargeback_count_30d / max(merchant.transaction_count_30d, 1) > 0.02,
                'high_transaction_volume': merchant.avg_transaction_value > 1000,
                'new_merchant': merchant.days_since_first_transaction < 90
            }
        }

# Initialize model manager
model_manager = ModelManager()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.models),
        "timestamp": datetime.now()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_risk(request: PredictionRequest):
    """Predict risk for merchants"""
    try:
        # Check cache if available
        cache_key = None
        if redis_client and len(request.data) == 1:
            cache_key = f"prediction:{request.data[0].merchant_id}:{request.model_version}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {cache_key}")
                return [PredictionResponse(**json.loads(cached_result))]
        
        # Make predictions
        results = model_manager.predict(request.data, request.model_version)
        
        # Cache result if single prediction
        if redis_client and cache_key and len(results) == 1:
            redis_client.setex(cache_key, 300, json.dumps(results[0], default=str))  # 5 min cache
        
        return [PredictionResponse(**result) for result in results]
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train a new model"""
    try:
        # Add training task to background
        background_tasks.add_task(
            _train_model_background,
            request.training_data,
            request.model_type,
            request.hyperparameters,
            request.cross_validation
        )
        
        return {
            "message": "Model training started",
            "model_type": request.model_type,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _train_model_background(training_data, model_type, hyperparameters, cross_validation):
    """Background task for model training"""
    try:
        logger.info(f"Starting training for {model_type}")
        
        # Prepare training data
        X = []
        y = []
        for merchant in training_data:
            features = model_manager._prepare_features(merchant)
            X.append(features)
            # For demo, generate synthetic labels
            y.append(1 if merchant.chargeback_count_30d > 2 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model based on type
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**hyperparameters)
        elif model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier(**hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X, y)
        
        # Save model with versioning
        version = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_manager.models[version] = model
        
        logger.info(f"Model {version} trained successfully")
        
    except Exception as e:
        logger.error(f"Background training error: {e}")

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": list(model_manager.models.keys()),
        "metadata": model_manager.model_metadata
    }

@app.get("/model/{version}/info")
async def get_model_info(version: str):
    """Get information about a specific model"""
    if version not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "version": version,
        "metadata": model_manager.model_metadata.get(version, {}),
        "feature_importance": model_manager.feature_importance.get(version, {})
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)