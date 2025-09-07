# 🚀 Anomaly Detection Platform - Demo Guide

This is a simplified demo version of the anomaly detection platform that you can run immediately to test the functionality.

## ✅ What's Working

The demo includes:
- **FastAPI Backend**: REST API with model training and scoring endpoints
- **Streamlit Frontend**: Interactive web UI for data visualization and model testing
- **Machine Learning**: Isolation Forest algorithm for anomaly detection
- **Real-time Scoring**: API endpoints for scoring individual records

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)
```bash
./start_demo.sh
```

This will start both the API server and web UI automatically.

### Option 2: Manual Startup

**Terminal 1 - Start API Server:**
```bash
python simple_api.py
```

**Terminal 2 - Start Web UI:**
```bash
streamlit run simple_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🌐 Access Points

Once running, you can access:

- **🌐 API Server**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **🎨 Web UI**: http://localhost:8501 (Interactive Streamlit interface)

## 🧪 Testing the Platform

### 1. Test API Endpoints
```bash
python test_platform.py
```

### 2. Manual API Testing

**Train the model:**
```bash
curl -X POST http://localhost:8000/train
```

**Score a normal record:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature_1": 0.1, "feature_2": -0.2, "feature_3": 0.3, "feature_4": -0.1}}'
```

**Score an anomaly record:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature_1": 4.0, "feature_2": 5.0, "feature_3": 3.5, "feature_4": 4.2}}'
```

### 3. Web UI Features

The Streamlit interface provides:
- **Data Generation**: Generate sample datasets with configurable anomaly ratios
- **Model Training**: Train isolation forest models on the generated data
- **Visualization**: Interactive plots showing data distribution and anomaly scores
- **Real-time Scoring**: Test individual records through the web interface
- **Performance Metrics**: Confusion matrix and accuracy metrics

## 📊 Demo Data

The platform generates synthetic data with:
- **4 Features**: Multivariate normal distributions with correlations
- **Normal Records**: Mean around [0,0,0,0] with low variance
- **Anomaly Records**: Mean around [3,3,3,3] with higher variance
- **Configurable Ratio**: Adjustable anomaly percentage (default 10%)

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with basic info |
| `/health` | GET | Health check |
| `/train` | POST | Train the anomaly detection model |
| `/score` | POST | Score a single record |
| `/models` | GET | List available models |

## 🎯 Key Features Demonstrated

1. **Data Ingestion**: Synthetic data generation with realistic patterns
2. **Feature Engineering**: Standard scaling and preprocessing
3. **Model Training**: Isolation Forest for unsupervised anomaly detection
4. **Real-time Scoring**: Sub-100ms API response times
5. **Web Interface**: Interactive data visualization and model testing
6. **API Documentation**: Auto-generated Swagger/OpenAPI docs

## 🐛 Troubleshooting

**If services don't start:**
1. Check if ports 8000 and 8501 are available
2. Install required packages: `pip install fastapi uvicorn streamlit plotly pandas numpy scikit-learn`
3. Kill existing processes: `pkill -f streamlit` and `pkill -f uvicorn`

**If API returns errors:**
1. Make sure to train the model first: `curl -X POST http://localhost:8000/train`
2. Check the API documentation at http://localhost:8000/docs

**If Streamlit UI doesn't load:**
1. Check the terminal output for error messages
2. Try accessing http://localhost:8501 directly in your browser
3. Restart the Streamlit process

## 🚀 Next Steps

This demo shows the core functionality. The full platform includes:
- Multiple data source connectors (Kafka, MQTT, etc.)
- Advanced ML algorithms (XGBoost, Autoencoders, etc.)
- Model registry and versioning
- Alert management and case tracking
- User authentication and RBAC
- Monitoring and drift detection

To explore the full platform, check the main README.md file.

## 📝 Notes

- This is a simplified demo for testing purposes
- Data is generated synthetically and not persisted
- Models are trained in-memory and reset on restart
- No authentication is required for the demo
- All data and models are temporary

Enjoy exploring the anomaly detection platform! 🎉
