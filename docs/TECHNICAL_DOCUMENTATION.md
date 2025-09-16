# Merchant Pulse Enterprise - Technical Documentation

## Overview

Merchant Pulse Enterprise is a comprehensive, AI-powered merchant risk intelligence platform built as a deployment-ready system with advanced machine learning capabilities, Retrieval-Augmented Generation (RAG), and cloud-native architecture. This project demonstrates enterprise-level software development practices suitable for a 4th-year graduate computer science project.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   Load Balancer │
│   (React)       │◄──►│   (Kong)        │◄──►│   (Nginx)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
        │   Backend    │ │ ML Service  │ │RAG Service │
        │   (FastAPI)  │ │ (FastAPI)   │ │ (FastAPI) │
        └───────┬──────┘ └──────┬──────┘ └─────┬──────┘
                │               │              │
        ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
        │  PostgreSQL  │ │   Redis     │ │ ChromaDB   │
        │  (Database)  │ │  (Cache)    │ │ (Vector)   │
        └──────────────┘ └─────────────┘ └────────────┘
```

### Microservices Architecture

#### 1. Backend Service (FastAPI)
- **Purpose**: Main API gateway and business logic
- **Responsibilities**:
  - User authentication and authorization
  - Merchant data management
  - Risk assessment orchestration
  - Analytics and reporting
- **Technologies**: FastAPI, PostgreSQL, Redis, SQLAlchemy
- **Port**: 8000

#### 2. ML Service (FastAPI)
- **Purpose**: Machine learning model serving and training
- **Responsibilities**:
  - Risk prediction models
  - Model training and versioning
  - Feature engineering
  - Model performance monitoring
- **Technologies**: FastAPI, scikit-learn, XGBoost, TensorFlow, MLflow
- **Port**: 8002

#### 3. RAG Service (FastAPI)
- **Purpose**: Retrieval-Augmented Generation for AI assistance
- **Responsibilities**:
  - Natural language processing
  - Knowledge base management
  - Conversational AI
  - Document processing and indexing
- **Technologies**: FastAPI, OpenAI API, LangChain, ChromaDB, HuggingFace
- **Port**: 8003

#### 4. Frontend (React)
- **Purpose**: User interface and experience
- **Responsibilities**:
  - Risk assessment dashboard
  - AI chat interface
  - Analytics visualization
  - Responsive design
- **Technologies**: React, Axios, CSS3, HTML5
- **Port**: 3000 (development), 80 (production)

## Key Features

### 1. Advanced Risk Intelligence
- **Real-time Risk Assessment**: Immediate risk scoring for merchants
- **Predictive Analytics**: Forecast potential risks before they materialize
- **Explainable AI**: Clear explanations for risk decisions
- **Batch Processing**: Handle multiple merchant assessments simultaneously

### 2. AI-Powered Assistance
- **Conversational AI**: Natural language interface for risk queries
- **RAG System**: Knowledge-based question answering
- **Contextual Responses**: Merchant-specific advice and recommendations
- **Multi-turn Conversations**: Persistent conversation history

### 3. Machine Learning Pipeline
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM, Neural Networks
- **Model Versioning**: MLflow integration for model lifecycle management
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Real-time Inference**: Low-latency prediction serving
- **Model Monitoring**: Performance tracking and drift detection

### 4. Cloud-Native Deployment
- **Containerization**: Docker containers for all services
- **Orchestration**: Kubernetes manifests with auto-scaling
- **Service Mesh**: Kong API Gateway for traffic management
- **Monitoring**: Prometheus and Grafana for observability
- **CI/CD**: GitHub Actions for automated deployment

## Technology Stack

### Backend Technologies
- **FastAPI**: Modern, fast web framework for APIs
- **PostgreSQL**: Robust relational database
- **Redis**: In-memory cache and session store
- **SQLAlchemy**: Python ORM for database operations
- **Alembic**: Database migration tool

### Machine Learning Stack
- **scikit-learn**: Traditional ML algorithms
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Fast gradient boosting
- **TensorFlow**: Deep learning framework
- **MLflow**: ML lifecycle management
- **Optuna**: Hyperparameter optimization

### AI/NLP Technologies
- **OpenAI API**: Large language models
- **LangChain**: LLM application framework
- **ChromaDB**: Vector database for embeddings
- **HuggingFace**: Open-source NLP models
- **Sentence Transformers**: Text embeddings

### Frontend Technologies
- **React**: Component-based UI library
- **Axios**: HTTP client for API calls
- **Modern CSS**: Flexbox, Grid, Animations
- **Responsive Design**: Mobile-first approach

### Infrastructure & DevOps
- **Docker**: Containerization platform
- **Kubernetes**: Container orchestration
- **Kong**: API Gateway and service mesh
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **GitHub Actions**: CI/CD pipeline

## API Documentation

### Core Endpoints

#### Authentication & Health
```
GET  /health                    - Health check
GET  /metrics                   - Prometheus metrics
```

#### Merchant Management
```
GET  /merchants/                - List all merchants
GET  /merchants/{id}/details    - Get merchant details
POST /predict/                  - Single merchant risk prediction
POST /predict/batch             - Batch risk prediction
```

#### AI & Analytics
```
POST /chat                      - AI chat interface
GET  /analytics/dashboard       - Dashboard analytics
```

### ML Service Endpoints
```
POST /predict                   - ML model prediction
POST /train                     - Train new model
GET  /models                    - List available models
GET  /model/{version}/info      - Model information
GET  /health                    - Service health
GET  /metrics                   - Model metrics
```

### RAG Service Endpoints
```
POST /query                     - Knowledge base query
POST /chat                      - Conversational AI
POST /documents/upload          - Upload documents
GET  /conversations/{id}/history - Conversation history
DELETE /conversations/{id}      - Clear conversation
```

## Deployment

### Local Development

#### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend development)

#### Quick Start
```bash
# Clone the repository
git clone https://github.com/jha-smriti/merchant-pulse-mvp.git
cd merchant-pulse-mvp

# Start all services
docker-compose up -d

# Initialize the database (if needed)
docker-compose exec backend python -c "
from app.mock_data import generate_mock_data, train_and_save_model
df = generate_mock_data()
train_and_save_model(df)
"

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (v1.24+)
- kubectl configured
- Helm (optional, for advanced deployments)

#### Deploy to Kubernetes
```bash
# Deploy to development environment
kubectl apply -k k8s/overlays/development

# Deploy to production environment
kubectl apply -k k8s/overlays/production

# Check deployment status
kubectl get pods -n merchant-pulse
kubectl get services -n merchant-pulse
```

### Environment Configuration

#### Environment Variables

**Backend Service:**
```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379
ML_SERVICE_URL=http://ml-service:8000
RAG_SERVICE_URL=http://rag-service:8000
LOG_LEVEL=INFO
```

**ML Service:**
```bash
REDIS_URL=redis://host:6379
MODEL_PATH=/app/models
MLFLOW_TRACKING_URI=http://mlflow:5000
```

**RAG Service:**
```bash
OPENAI_API_KEY=your-openai-api-key
CHROMA_URL=http://chromadb:8000
HUGGINGFACE_API_TOKEN=your-hf-token
```

## Security

### API Security
- **Rate Limiting**: Kong-based rate limiting
- **Authentication**: JWT-based authentication
- **CORS**: Configured for allowed origins
- **Input Validation**: Pydantic models for request validation

### Container Security
- **Non-root Users**: All containers run as non-root
- **Security Scanning**: Trivy vulnerability scanning
- **Minimal Images**: Alpine-based images where possible
- **Secret Management**: Kubernetes secrets for sensitive data

### Database Security
- **Connection Pooling**: SQLAlchemy connection pooling
- **SQL Injection Prevention**: ORM-based queries
- **Encryption**: TLS connections enforced
- **Backup Strategy**: Automated backup procedures

## Monitoring & Observability

### Metrics Collection
- **Application Metrics**: Custom Prometheus metrics
- **System Metrics**: Node exporter metrics
- **Database Metrics**: PostgreSQL exporter
- **Cache Metrics**: Redis exporter

### Alerting
- **High Error Rates**: > 5% error rate for 2 minutes
- **High Response Times**: > 2 seconds for 95th percentile
- **Service Downtime**: Service unavailable for > 1 minute
- **High Memory Usage**: > 90% memory usage for 5 minutes
- **High CPU Usage**: > 80% CPU usage for 5 minutes

### Logging
- **Structured Logging**: JSON format for all services
- **Correlation IDs**: Request tracing across services
- **Log Aggregation**: Centralized log collection
- **Log Retention**: 30-day retention policy

## Testing

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **End-to-End Tests**: Full workflow testing
- **Performance Tests**: Load and stress testing

### Running Tests

#### Backend Tests
```bash
cd backend
python -m pytest tests/ -v --cov=app
```

#### Frontend Tests
```bash
cd frontend
npm test -- --coverage --watchAll=false
```

#### ML Service Tests
```bash
cd ml-service
python -m pytest tests/ -v --cov=app
```

## Performance Optimization

### Database Optimization
- **Indexing**: Optimized database indexes
- **Query Optimization**: Efficient SQL queries
- **Connection Pooling**: Reduced connection overhead
- **Read Replicas**: Scaled read operations

### Caching Strategy
- **Redis Caching**: API response caching
- **Model Caching**: ML model result caching
- **CDN**: Static asset caching
- **Browser Caching**: Client-side caching

### Scalability
- **Horizontal Pod Autoscaling**: Kubernetes HPA
- **Load Balancing**: Multi-instance load distribution
- **Database Sharding**: Horizontal database scaling
- **Microservices**: Independent service scaling

## Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run quality checks (linting, testing)
5. Submit a pull request
6. Code review and approval
7. Automated deployment

### Code Quality Standards
- **Linting**: Black, isort, flake8 for Python; ESLint for JavaScript
- **Type Checking**: mypy for Python; TypeScript for JavaScript
- **Test Coverage**: Minimum 80% coverage required
- **Documentation**: Comprehensive docstrings and comments

## Future Enhancements

### Planned Features
- **Real-time Streaming**: Kafka-based event streaming
- **Advanced Analytics**: Time-series analysis and forecasting
- **Mobile Application**: React Native mobile app
- **Multi-tenancy**: Support for multiple organizations
- **Blockchain Integration**: Decentralized identity verification

### Performance Improvements
- **GraphQL API**: More efficient data fetching
- **Edge Computing**: CDN-based edge deployments
- **Database Optimization**: Advanced indexing strategies
- **Caching Layers**: Multi-level caching architecture

## License

This project is created for educational purposes as part of a 4th-year graduate computer science program. All code is provided under the MIT License for academic use.

## Contact

For questions, issues, or contributions, please contact the development team or create an issue in the GitHub repository.

---

*This documentation is maintained alongside the codebase and updated with each release.*