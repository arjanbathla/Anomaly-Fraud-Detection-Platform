# Anomaly & Fraud Detection Platform

A comprehensive anomaly and fraud detection platform built with FastAPI, Streamlit, and modern ML libraries. This platform provides end-to-end capabilities for detecting suspicious events in streaming or batch datasets, with a web UI for analysts and API endpoints for integration.

## �� Features

### Core Capabilities
- **Multi-source Data Ingestion**: CSV, Parquet, Kafka, MQTT, Webhooks, REST API
- **Advanced ML Models**: Isolation Forest, One-Class SVM, Autoencoders, XGBoost, LightGBM
- **Real-time Scoring**: Sub-100ms model inference with REST API
- **Feature Engineering**: Automated transformations, time-window aggregations, PII handling
- **Alert Management**: Configurable routing, escalation, case management
- **Explainability**: SHAP/LIME integration for model interpretability
- **Monitoring**: Drift detection, performance metrics, cost analysis

### User Roles
- **Analyst**: Review alerts, label outcomes, export reports
- **Data Engineer**: Configure data sources, schemas, pipelines
- **Model Owner**: Train/tune models, promote versions, monitor drift
- **Admin**: Manage users, roles, compliance settings

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature Store  │    │   ML Models     │
│  (Kafka, CSV,   │───▶│   (Feast)       │───▶│  (MLflow)       │
│   MQTT, etc.)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │    │ Feature Pipeline│    │ Model Registry  │
│  & Validation   │    │ & Transformations│    │ & Versioning    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Streamlit     │    │   Monitoring    │
│   (REST API)    │    │   (Web UI)      │    │ (Prometheus)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd anomaly-detection-platform
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the platform**
   ```bash
   docker-compose up -d
   ```

4. **Access the services**
   - API Documentation: http://localhost:8000/docs
   - Web UI: http://localhost:8501
   - MLflow UI: http://localhost:5000
   - Grafana: http://localhost:3000 (admin/admin)

### Manual Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up database**
   ```bash
   # Start PostgreSQL and Redis
   # Run the initialization script
   psql -U postgres -d anomaly_detection -f scripts/init_db.sql
   ```

3. **Start the services**
   ```bash
   # Start API server
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   
   # Start Streamlit UI (in another terminal)
   streamlit run src/ui/streamlit_app.py --server.port 8501
   ```

## 📖 Usage

### API Endpoints

#### Authentication
```bash
# Get access token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

#### Model Scoring
```bash
# Score a single record
curl -X POST "http://localhost:8000/api/v1/scoring/score" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "entity_id": "user_123",
      "amount": 1500.00,
      "merchant_category": "electronics",
      "time_of_day": "23:45"
    },
    "model_name": "credit_card_fraud",
    "include_explanations": true
  }'
```

#### Batch Scoring
```bash
# Score multiple records
curl -X POST "http://localhost:8000/api/v1/scoring/score/batch" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"entity_id": "user_123", "amount": 1500.00},
      {"entity_id": "user_456", "amount": 50.00}
    ],
    "model_name": "credit_card_fraud"
  }'
```

### Web UI

Access the Streamlit web interface at http://localhost:8501

**Features:**
- **Dashboard**: Overview of alerts, cases, and system metrics
- **Alerts**: Review and manage anomaly alerts
- **Cases**: Create and track investigation cases
- **Models**: Monitor model performance and training
- **Analytics**: Generate reports and visualizations

### Configuration

#### Data Source Configuration
```yaml
# Example data source config
name: "Credit Card Transactions"
source_type: "kafka"
config:
  topic: "transactions"
  bootstrap_servers: ["localhost:9092"]
schema:
  type: "object"
  properties:
    user_id:
      type: "string"
    amount:
      type: "number"
    merchant_category:
      type: "string"
  required: ["user_id", "amount"]
pii_fields: ["user_id"]
```

#### Feature Pipeline Configuration
```yaml
# Example feature pipeline
name: "Transaction Features"
transformations:
  - type: "imputation"
    config:
      strategy: "mean"
  - type: "scaling"
    config:
      method: "standard"
  - type: "time_window_aggregation"
    config:
      entity_key: "user_id"
      windows: ["1h", "24h", "7d"]
      aggregations: ["mean", "std", "count"]
```

## 🔧 Development

### Project Structure
```
anomaly-detection-platform/
├── src/
│   ├── api/                    # FastAPI endpoints
│   ├── core/                   # Core models and configuration
│   ├── data/                   # Data ingestion and processing
│   ├── models/                 # ML algorithms and registry
│   ├── ui/                     # Streamlit web interface
│   └── monitoring/             # Monitoring and alerting
├── tests/                      # Test suites
├── scripts/                    # Utility scripts
├── docker/                     # Docker configuration
├── docs/                       # Documentation
└── requirements.txt            # Python dependencies
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/
```

## 📊 Monitoring

### Metrics
- **Model Performance**: Precision, Recall, F1, ROC-AUC, PR-AUC
- **System Metrics**: Latency, throughput, error rates
- **Business Metrics**: Cost savings, false positive rates
- **Data Drift**: PSI, KS statistics for feature drift

### Alerting
- **Slack Integration**: Real-time notifications
- **Email Alerts**: Escalation for critical issues
- **Webhook Support**: Custom integrations
- **Escalation Policies**: Configurable alert routing

## 🔒 Security

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- API key management
- Audit logging

### Data Privacy
- PII detection and masking
- Configurable data retention
- GDPR compliance features
- Encryption at rest and in transit

## 🚀 Deployment

### Production Considerations
- Use environment-specific configurations
- Set up proper secrets management
- Configure monitoring and alerting
- Implement backup and disaster recovery
- Set up CI/CD pipelines

### Scaling
- Horizontal scaling with load balancers
- Database read replicas
- Caching with Redis
- Message queue scaling

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [User Guide](docs/user-guide.md) - Platform usage guide
- [Developer Guide](docs/developer-guide.md) - Development setup
- [Deployment Guide](docs/deployment-guide.md) - Production deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- Create an issue for bug reports
- Check the documentation
- Contact the development team

## 🔄 Changelog

### v1.0.0
- Initial release
- Core anomaly detection capabilities
- Web UI and API
- Multi-source data ingestion
- Model registry and versioning
