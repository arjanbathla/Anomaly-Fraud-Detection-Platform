#!/usr/bin/env python3
"""Script to train anomaly detection models."""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.algorithms.anomaly_detectors import AnomalyDetectorFactory
from models.registry.model_registry import ModelRegistry
from core.models import Model, ModelType, ModelMetrics
from data.processing.transformations import FeaturePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000, n_features: int = 10, anomaly_ratio: float = 0.1):
    """Generate sample data for training."""
    # Generate normal data
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomaly = n_samples - n_normal
    
    # Normal data (multivariate normal distribution)
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    # Anomaly data (outliers)
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,  # Shifted mean
        cov=np.eye(n_features) * 2,    # Higher variance
        size=n_anomaly
    )
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def train_model(algorithm: str, X: np.ndarray, y: np.ndarray = None):
    """Train a model with the given algorithm."""
    logger.info(f"Training {algorithm} model...")
    
    # Model configuration
    config = {
        "n_estimators": 100,
        "random_state": 42,
        "contamination": 0.1 if algorithm in ["isolation_forest", "one_class_svm"] else None
    }
    
    # Create detector
    detector = AnomalyDetectorFactory.create_detector(algorithm, config)
    
    # Train model
    detector.fit(X, y)
    
    # Get predictions for evaluation
    scores = detector.predict(X)
    
    # Calculate metrics if we have labels
    metrics = None
    if y is not None:
        # For unsupervised models, we need to set a threshold
        if algorithm in ["isolation_forest", "one_class_svm", "autoencoder"]:
            # Use 90th percentile as threshold
            threshold = np.percentile(scores, 90)
            predictions = scores > threshold
        else:
            # For supervised models, use 0.5 threshold
            predictions = scores > 0.5
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        roc_auc = roc_auc_score(y, scores)
        
        metrics = ModelMetrics(
            model_id="",  # Will be set when model is registered
            model_version="1.0.0",
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            pr_auc=0.0,  # Would need precision_recall_curve
            mcc=0.0,     # Would need matthews_corrcoef
            calibration_score=0.0,
            cost_savings=0.0
        )
    
    return detector, metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--algorithm", choices=["isolation_forest", "one_class_svm", "autoencoder", "supervised"], 
                       default="isolation_forest", help="Algorithm to use")
    parser.add_argument("--samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--features", type=int, default=10, help="Number of features")
    parser.add_argument("--anomaly-ratio", type=float, default=0.1, help="Ratio of anomalies")
    parser.add_argument("--output-dir", default="./model_registry", help="Output directory for models")
    
    args = parser.parse_args()
    
    # Generate sample data
    logger.info("Generating sample data...")
    X, y = generate_sample_data(args.samples, args.features, args.anomaly_ratio)
    
    # Train model
    detector, metrics = train_model(args.algorithm, X, y)
    
    # Create model metadata
    model_metadata = Model(
        name=f"sample_{args.algorithm}_model",
        model_type=args.algorithm,
        version="1.0.0",
        config={
            "n_estimators": 100,
            "random_state": 42,
            "contamination": 0.1,
            "threshold": 0.5
        },
        metrics=metrics.dict() if metrics else {},
        tags=["sample", "demo"],
        is_production=False
    )
    
    # Register model
    logger.info("Registering model...")
    registry = ModelRegistry(registry_path=args.output_dir)
    model_path = registry.register_model(
        model=detector,
        model_metadata=model_metadata,
        metrics=metrics
    )
    
    logger.info(f"Model registered successfully at: {model_path}")
    logger.info(f"Model ID: {model_metadata.id}")
    logger.info(f"Model Name: {model_metadata.name}")
    logger.info(f"Model Version: {model_metadata.version}")
    
    if metrics:
        logger.info(f"Precision: {metrics.precision:.3f}")
        logger.info(f"Recall: {metrics.recall:.3f}")
        logger.info(f"F1 Score: {metrics.f1_score:.3f}")
        logger.info(f"ROC AUC: {metrics.roc_auc:.3f}")


if __name__ == "__main__":
    main()
