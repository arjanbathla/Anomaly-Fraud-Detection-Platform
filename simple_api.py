#!/usr/bin/env python3
"""Simplified FastAPI version for testing."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

app = FastAPI(title="Anomaly Detection API - Demo", version="1.0.0")

# Global model (in production, this would be loaded from model registry)
model = None
scaler = StandardScaler()

class ScoreRequest(BaseModel):
    data: Dict[str, Any]
    model_name: str = "default"

class ScoreResponse(BaseModel):
    prediction_id: str
    score: float
    is_anomaly: bool
    threshold: float

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Anomaly Detection API - Demo",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/train", response_model=Dict[str, Any])
async def train_model():
    """Train a simple anomaly detection model."""
    global model, scaler
    
    # Generate sample training data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0, 0],
        cov=[[1, 0.3, 0.2, 0.1], [0.3, 1, 0.4, 0.2], [0.2, 0.4, 1, 0.3], [0.1, 0.2, 0.3, 1]],
        size=int(n_samples * 0.9)
    )
    
    # Anomaly data
    anomaly_data = np.random.multivariate_normal(
        mean=[3, 3, 3, 3],
        cov=[[2, 0.5, 0.3, 0.2], [0.5, 2, 0.6, 0.4], [0.3, 0.6, 2, 0.5], [0.2, 0.4, 0.5, 2]],
        size=int(n_samples * 0.1)
    )
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    
    # Scale data
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_scaled)
    
    return {
        "message": "Model trained successfully",
        "n_samples": len(X),
        "n_features": X.shape[1]
    }

@app.post("/score", response_model=ScoreResponse)
async def score_record(request: ScoreRequest):
    """Score a single record."""
    global model, scaler
    
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
    
    try:
        # Extract features from request
        features = [
            request.data.get("feature_1", 0.0),
            request.data.get("feature_2", 0.0),
            request.data.get("feature_3", 0.0),
            request.data.get("feature_4", 0.0)
        ]
        
        # Convert to numpy array and scale
        X = np.array([features])
        X_scaled = scaler.transform(X)
        
        # Get prediction
        score = -model.decision_function(X_scaled)[0]  # Convert to positive score
        is_anomaly = model.predict(X_scaled)[0] == -1
        threshold = 0.5  # Simple threshold
        
        return ScoreResponse(
            prediction_id=f"pred_{np.random.randint(10000, 99999)}",
            score=float(score),
            is_anomaly=bool(is_anomaly),
            threshold=threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring record: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "default",
                "type": "isolation_forest",
                "status": "trained" if model is not None else "not_trained"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
