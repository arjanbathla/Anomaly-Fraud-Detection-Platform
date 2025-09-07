"""Model registry for versioning and managing ML models."""

import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import logging

from ...core.models import Model, ModelMetrics
from ...core.exceptions import ModelTrainingError, ModelInferenceError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model versioning, storage, and metadata."""
    
    def __init__(self, registry_path: str = "./model_registry", mlflow_uri: str = "http://localhost:5000"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        self.experiment_name = "anomaly_detection"
        
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not initialize MLflow: {e}")
            self.experiment_id = None
    
    def register_model(self, 
                      model: Any,
                      model_metadata: Model,
                      metrics: Optional[ModelMetrics] = None,
                      artifacts: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model version."""
        try:
            # Generate model hash for reproducibility
            model_hash = self._generate_model_hash(model, model_metadata)
            
            # Create model directory
            model_dir = self.registry_path / model_metadata.name / model_metadata.version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata.dict(), f, default=str)
            
            # Save metrics if provided
            if metrics:
                metrics_path = model_dir / "metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics.dict(), f, default=str)
            
            # Save artifacts if provided
            if artifacts:
                artifacts_dir = model_dir / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                for name, artifact in artifacts.items():
                    artifact_path = artifacts_dir / f"{name}.pkl"
                    with open(artifact_path, 'wb') as f:
                        pickle.dump(artifact, f)
            
            # Log to MLflow if available
            if self.experiment_id:
                self._log_to_mlflow(model, model_metadata, metrics, artifacts)
            
            logger.info(f"Model {model_metadata.name} v{model_metadata.version} registered successfully")
            return str(model_dir)
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to register model: {str(e)}")
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Any, Model]:
        """Load a model and its metadata."""
        try:
            if version is None:
                # Load latest version
                model_versions = self._get_model_versions(model_name)
                if not model_versions:
                    raise ModelInferenceError(f"No versions found for model {model_name}")
                version = max(model_versions)
            
            model_dir = self.registry_path / model_name / version
            
            # Load model
            model_path = model_dir / "model.pkl"
            if not model_path.exists():
                raise ModelInferenceError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = Model(**metadata_dict)
            
            return model, metadata
            
        except Exception as e:
            raise ModelInferenceError(f"Failed to load model: {str(e)}")
    
    def list_models(self) -> List[Model]:
        """List all registered models."""
        models = []
        
        for model_dir in self.registry_path.iterdir():
            if model_dir.is_dir():
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir():
                        metadata_path = version_dir / "metadata.json"
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata_dict = json.load(f)
                                models.append(Model(**metadata_dict))
                            except Exception as e:
                                logger.warning(f"Could not load metadata for {version_dir}: {e}")
        
        return models
    
    def get_model_versions(self, model_name: str) -> List[str]:
        """Get all versions of a model."""
        return self._get_model_versions(model_name)
    
    def promote_model(self, model_name: str, version: str, environment: str = "production") -> None:
        """Promote a model version to an environment."""
        try:
            model_dir = self.registry_path / model_name / version
            metadata_path = model_dir / "metadata.json"
            
            if not metadata_path.exists():
                raise ModelInferenceError(f"Model {model_name} v{version} not found")
            
            # Load and update metadata
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = Model(**metadata_dict)
            metadata.tags.append(environment)
            metadata.is_production = (environment == "production")
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata.dict(), f, default=str)
            
            logger.info(f"Model {model_name} v{version} promoted to {environment}")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to promote model: {str(e)}")
    
    def delete_model(self, model_name: str, version: str) -> None:
        """Delete a model version."""
        try:
            model_dir = self.registry_path / model_name / version
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Model {model_name} v{version} deleted")
            else:
                raise ModelInferenceError(f"Model {model_name} v{version} not found")
                
        except Exception as e:
            raise ModelTrainingError(f"Failed to delete model: {str(e)}")
    
    def _generate_model_hash(self, model: Any, metadata: Model) -> str:
        """Generate a hash for model reproducibility."""
        # Create a string representation of model and metadata
        model_str = str(model)
        metadata_str = json.dumps(metadata.dict(), sort_keys=True)
        combined = f"{model_str}_{metadata_str}"
        
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _get_model_versions(self, model_name: str) -> List[str]:
        """Get all versions for a model."""
        model_path = self.registry_path / model_name
        if not model_path.exists():
            return []
        
        versions = []
        for version_dir in model_path.iterdir():
            if version_dir.is_dir():
                versions.append(version_dir.name)
        
        return sorted(versions)
    
    def _log_to_mlflow(self, 
                      model: Any,
                      metadata: Model,
                      metrics: Optional[ModelMetrics],
                      artifacts: Optional[Dict[str, Any]]) -> None:
        """Log model to MLflow."""
        try:
            with mlflow.start_run(experiment_id=self.experiment_id):
                # Log parameters
                mlflow.log_params(metadata.config)
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics({
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "roc_auc": metrics.roc_auc,
                        "pr_auc": metrics.pr_auc,
                        "mcc": metrics.mcc,
                        "calibration_score": metrics.calibration_score,
                        "cost_savings": metrics.cost_savings
                    })
                
                # Log model
                if hasattr(model, 'predict_proba'):
                    mlflow.sklearn.log_model(model, "model")
                else:
                    # For custom models, save as pickle
                    model_path = "model.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(model_path)
                
                # Log artifacts
                if artifacts:
                    for name, artifact in artifacts.items():
                        artifact_path = f"{name}.pkl"
                        with open(artifact_path, 'wb') as f:
                            pickle.dump(artifact, f)
                        mlflow.log_artifact(artifact_path)
                
                # Set tags
                mlflow.set_tags({
                    "model_name": metadata.name,
                    "model_type": metadata.model_type,
                    "version": metadata.version,
                    "is_production": str(metadata.is_production)
                })
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
