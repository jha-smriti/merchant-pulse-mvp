**Merchant Pulse Lite** 🚀

A full-stack AI-powered merchant risk intelligence platform built under a 12-hour constraint. This application demonstrates the ability to create a functional MVP that predicts merchant risk scores using machine learning.

https://img.shields.io/badge/Python-3.9%252B-blue
https://img.shields.io/badge/FastAPI-0.104.1-green
https://img.shields.io/badge/React-18.2.0-blue
https://img.shields.io/badge/PostgreSQL-13-blue
https://img.shields.io/badge/Docker-Compose-blue

🎯 Problem Statement
Payment networks like Visa and Mastercard lose billions annually to payment fraud and merchant chargebacks. This solution provides:

AI-powered risk prediction for merchants

Real-time analytics and visualization

Proactive risk identification before issues escalate

Explainable AI with clear risk reasons

✨ Features
Machine Learning Integration: Predicts merchant risk scores using Random Forest algorithm

Real-time Dashboard: Interactive React frontend with risk visualization

RESTful API: FastAPI backend with JWT authentication

Database Storage: PostgreSQL for persistent data storage

Dockerized Deployment: Easy setup with Docker Compose

🏗️ Architecture
text
merchant-pulse-platform/
├── backend/          # FastAPI application
├── frontend/         # React application  
├── model/            # ML model artifacts
└── docker-compose.yml # Container orchestration
🚀 Quick Start
Prerequisites
Docker and Docker Compose

Python 3.9+ (for local development)

Node.js 16+ (for local development)

Installation
Clone and setup the project:

bash
git clone <your-repo-url>
cd merchant-pulse-platform
Start the application:

bash
docker-compose up -d
Initialize the database:

bash
docker-compose exec backend python -c "
from app.utils.database import engine, Base
from app.services.data_service import initialize_database
from app.utils.database import SessionLocal

Base.metadata.create_all(bind=engine)
db = SessionLocal()
initialize_database(db)
db.close()
print('Database initialized successfully')
"
Access the application:

Frontend: http://localhost:3000

Backend API: http://localhost:8000

API Documentation: http://localhost:8000/docs

🛠️ Manual Setup (Without Docker)
Backend Setup
bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate the ML model
python app/mock_data.py

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Frontend Setup
bash
cd frontend
npm install
npm start
📊 How It Works
Data Generation: Creates synthetic merchant data with realistic risk patterns

Machine Learning: Uses a Random Forest classifier trained on historical patterns

Risk Prediction: Calculates probability scores from 0-100% for each merchant

Explainable AI: Provides clear reasons for risk assessments (e.g., "High chargeback count (5)")

Visualization: Displays results in an intuitive dashboard interface

🔍 API Endpoints
Endpoint	Method	Description
/auth/token	POST	User authentication
/merchants/	GET	Get all merchants with filtering
/merchants/{id}	GET	Get specific merchant details
/merchants/predict	POST	Predict risk for a new merchant
/health	GET	System health check
🧠 Machine Learning Model
Algorithm: Random Forest Classifier

Features: Transaction value, volume, chargeback history, business age

Output: Risk probability (0-1) with explainable reasoning

Training: Synthetic data with realistic risk patterns

🏆 Key Achievements
Built in under 12 hours, this project demonstrates:

✅ End-to-end full-stack development

✅ Machine learning integration

✅ Clean architecture and code organization

✅ Docker containerization

✅ Production-ready API design

✅ Responsive frontend design

✅ Database schema design and management

📈 Future Enhancements
Real payment data integration

Advanced ML models (XGBoost, Neural Networks)

Real-time data streaming

Advanced visualization and reporting

Multi-tenant architecture

Automated model retraining pipeline

🤝 Contributing
This project was built under extreme time constraints as a demonstration of rapid development skills. For a more production-ready version, see the full Merchant Pulse Platform.

📄 License
This project is created for demonstration purposes as part of a technical assessment.

