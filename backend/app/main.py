from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model import predict_risk
from .mock_data import generate_mock_data
import pandas as pd

app = FastAPI(title="Merchant Pulse API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mock data into a DataFrame
df = generate_mock_data()

class MerchantQuery(BaseModel):
    merchant_id: str

@app.get("/")
def read_root():
    return {"message": "Merchant Pulse API is running!"}

@app.post("/predict/")
async def get_merchant_risk(query: MerchantQuery):
    """API endpoint to get a risk prediction for a merchant ID."""
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

@app.get("/merchants/")
def get_all_merchants():
    """Get a list of all merchant IDs for the frontend dropdown."""
    return {"merchant_ids": df['merchant_id'].tolist()}