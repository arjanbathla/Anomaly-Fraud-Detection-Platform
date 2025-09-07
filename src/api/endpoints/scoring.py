"""API endpoints for model scoring and predictions."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from ...core.models import Prediction, Alert, AlertStatus
from ...core.exceptions import ModelInferenceError, AuthenticationError
from ...core.auth import get_current_user
from ...models.registry.model_registry import ModelRegistry
from ...models.algorithms.anomaly_detectors import AnomalyDetectorFactory
from ...monitoring.alerts.alert_manager import AlertManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scoring", tags=["scoring"])

# Initialize components
model_registry = ModelRegistry()
alert_manager = AlertManager()


@router.post("/score", response_model=Dict[str, Any])
async def score_single_record(
    data: Dict[str, Any],
    model_name: str,
    model_version: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Score a single record for anomalies."""
    try:
        # Load model
        model, metadata = model_registry.load_model(model_name, model_version)
        
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # Apply feature transformations if needed
        # This would typically involve loading the feature pipeline
        # and applying the same transformations used during training
        
        # Convert to numpy array for prediction
        X = df.values
        
        # Create detector instance
        detector = AnomalyDetectorFactory.create_detector(
            metadata.model_type, 
            metadata.config
        )
        detector.model = model  # Set the loaded model
        
        # Get prediction
        score = detector.predict(X)[0]
        is_anomaly = score > metadata.config.get("threshold", 0.5)
        
        # Create prediction record
        prediction = Prediction(
            model_id=metadata.id,
            model_version=metadata.version,
            entity_id=data.get("entity_id", "unknown"),
            features=data,
            score=float(score),
            threshold=metadata.config.get("threshold", 0.5),
            is_anomaly=bool(is_anomaly)
        )
        
        # Generate explanations if requested
        if data.get("include_explanations", False):
            # This would integrate with SHAP/LIME
            prediction.explanations = {
                "top_features": ["feature1", "feature2", "feature3"],
                "feature_importance": [0.3, 0.2, 0.1]
            }
        
        # Create alert if anomaly detected
        if is_anomaly:
            alert = Alert(
                prediction_id=prediction.id,
                entity_id=prediction.entity_id,
                score=prediction.score,
                status=AlertStatus.OPEN
            )
            # Send alert
            await alert_manager.send_alert(alert)
        
        return {
            "prediction_id": prediction.id,
            "score": prediction.score,
            "is_anomaly": prediction.is_anomaly,
            "threshold": prediction.threshold,
            "explanations": prediction.explanations,
            "timestamp": prediction.created_at.isoformat()
        }
        
    except ModelInferenceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in scoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/score/batch", response_model=Dict[str, Any])
async def score_batch_records(
    data: List[Dict[str, Any]],
    model_name: str,
    model_version: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Score multiple records for anomalies."""
    try:
        # Load model
        model, metadata = model_registry.load_model(model_name, model_version)
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Apply feature transformations
        X = df.values
        
        # Create detector instance
        detector = AnomalyDetectorFactory.create_detector(
            metadata.model_type,
            metadata.config
        )
        detector.model = model
        
        # Get predictions
        scores = detector.predict(X)
        threshold = metadata.config.get("threshold", 0.5)
        is_anomalies = scores > threshold
        
        # Create prediction records
        predictions = []
        alerts = []
        
        for i, (record, score, is_anomaly) in enumerate(zip(data, scores, is_anomalies)):
            prediction = Prediction(
                model_id=metadata.id,
                model_version=metadata.version,
                entity_id=record.get("entity_id", f"record_{i}"),
                features=record,
                score=float(score),
                threshold=threshold,
                is_anomaly=bool(is_anomaly)
            )
            predictions.append(prediction)
            
            # Create alert if anomaly detected
            if is_anomaly:
                alert = Alert(
                    prediction_id=prediction.id,
                    entity_id=prediction.entity_id,
                    score=prediction.score,
                    status=AlertStatus.OPEN
                )
                alerts.append(alert)
        
        # Send alerts
        if alerts:
            await alert_manager.send_batch_alerts(alerts)
        
        return {
            "predictions": [
                {
                    "prediction_id": p.id,
                    "entity_id": p.entity_id,
                    "score": p.score,
                    "is_anomaly": p.is_anomaly,
                    "threshold": p.threshold
                }
                for p in predictions
            ],
            "summary": {
                "total_records": len(data),
                "anomalies_detected": sum(is_anomalies),
                "anomaly_rate": float(sum(is_anomalies) / len(data))
            }
        }
        
    except ModelInferenceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch scoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_available_models(current_user: dict = Depends(get_current_user)):
    """List all available models."""
    try:
        models = model_registry.list_models()
        return [
            {
                "id": model.id,
                "name": model.name,
                "model_type": model.model_type,
                "version": model.version,
                "is_production": model.is_production,
                "tags": model.tags,
                "created_at": model.created_at.isoformat()
            }
            for model in models
        ]
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models/{model_name}/versions", response_model=List[str])
async def list_model_versions(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    """List all versions of a specific model."""
    try:
        versions = model_registry.get_model_versions(model_name)
        return versions
    except Exception as e:
        logger.error(f"Error listing model versions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/models/{model_name}/promote")
async def promote_model(
    model_name: str,
    version: str,
    environment: str = "production",
    current_user: dict = Depends(get_current_user)
):
    """Promote a model version to an environment."""
    try:
        # Check if user has permission to promote models
        if current_user.get("role") not in ["model_owner", "admin"]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        model_registry.promote_model(model_name, version, environment)
        return {"message": f"Model {model_name} v{version} promoted to {environment}"}
        
    except Exception as e:
        logger.error(f"Error promoting model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
