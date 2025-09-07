"""Anomaly detection algorithms for the platform."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)


class BaseAnomalyDetector(ABC):
    """Base class for anomaly detection algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False
        self.threshold = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseAnomalyDetector':
        """Fit the anomaly detector."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        pass
    
    def set_threshold(self, threshold: float) -> None:
        """Set the anomaly threshold."""
        self.threshold = threshold
    
    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Determine if samples are anomalies based on threshold."""
        scores = self.predict(X)
        if self.threshold is None:
            raise ValueError("Threshold must be set before calling is_anomaly")
        return scores > self.threshold


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = IsolationForest(
            n_estimators=config.get("n_estimators", 100),
            max_samples=config.get("max_samples", "auto"),
            contamination=config.get("contamination", 0.1),
            max_features=config.get("max_features", 1.0),
            bootstrap=config.get("bootstrap", False),
            random_state=config.get("random_state", 42)
        )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model."""
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (higher = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        # We convert to positive scores where higher = more anomalous
        scores = -self.model.decision_function(X)
        return scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        scores = self.predict(X)
        # Normalize scores to probabilities (0-1)
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            proba = (scores - min_score) / (max_score - min_score)
        else:
            proba = np.zeros_like(scores)
        return proba


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = OneClassSVM(
            kernel=config.get("kernel", "rbf"),
            gamma=config.get("gamma", "scale"),
            nu=config.get("nu", 0.1),
            degree=config.get("degree", 3),
            coef0=config.get("coef0", 0.0)
        )
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OneClassSVMDetector':
        """Fit the One-Class SVM model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        # One-Class SVM returns -1 for anomalies, 1 for normal
        # We convert to positive scores where higher = more anomalous
        scores = -self.model.decision_function(X_scaled)
        return scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        scores = self.predict(X)
        # Normalize scores to probabilities
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            proba = (scores - min_score) / (max_score - min_score)
        else:
            proba = np.zeros_like(scores)
        return proba


class AutoencoderDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.encoding_dim = config.get("encoding_dim", 32)
        self.hidden_dims = config.get("hidden_dims", [64, 32])
        self.activation = config.get("activation", "relu")
        self.optimizer = config.get("optimizer", "adam")
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        self.validation_split = config.get("validation_split", 0.2)
        self.model = None
        self.scaler = StandardScaler()
    
    def _build_model(self, input_dim: int) -> keras.Model:
        """Build the autoencoder model."""
        input_layer = keras.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for dim in self.hidden_dims:
            encoded = keras.layers.Dense(dim, activation=self.activation)(encoded)
        encoded = keras.layers.Dense(self.encoding_dim, activation=self.activation)(encoded)
        
        # Decoder
        decoded = encoded
        for dim in reversed(self.hidden_dims):
            decoded = keras.layers.Dense(dim, activation=self.activation)(decoded)
        decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer=self.optimizer, loss='mse')
        
        return autoencoder
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AutoencoderDetector':
        """Fit the autoencoder model."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = self._build_model(X_scaled.shape[1])
        
        # Train the autoencoder
        self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=0
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (reconstruction error)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_pred = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction error
        mse = np.mean((X_scaled - X_pred) ** 2, axis=1)
        return mse
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        scores = self.predict(X)
        # Normalize scores to probabilities
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            proba = (scores - min_score) / (max_score - min_score)
        else:
            proba = np.zeros_like(scores)
        return proba


class SupervisedAnomalyDetector(BaseAnomalyDetector):
    """Supervised anomaly detector using traditional ML algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.algorithm = config.get("algorithm", "xgboost")
        self.model = None
        self.scaler = StandardScaler()
        
        if self.algorithm == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                random_state=config.get("random_state", 42)
            )
        elif self.algorithm == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                random_state=config.get("random_state", 42)
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                random_state=config.get("random_state", 42)
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupervisedAnomalyDetector':
        """Fit the supervised model."""
        if y is None:
            raise ValueError("Supervised model requires target variable")
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        # For supervised models, we use the probability of the positive class
        proba = self.model.predict_proba(X_scaled)
        if proba.shape[1] == 2:
            return proba[:, 1]  # Probability of positive class (anomaly)
        else:
            return proba[:, -1]  # Probability of last class
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        return self.predict(X)


class AnomalyDetectorFactory:
    """Factory for creating anomaly detectors."""
    
    _detectors = {
        "isolation_forest": IsolationForestDetector,
        "one_class_svm": OneClassSVMDetector,
        "autoencoder": AutoencoderDetector,
        "supervised": SupervisedAnomalyDetector,
    }
    
    @classmethod
    def create_detector(cls, algorithm: str, config: Dict[str, Any]) -> BaseAnomalyDetector:
        """Create an anomaly detector."""
        detector_class = cls._detectors.get(algorithm)
        if not detector_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return detector_class(config)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithms."""
        return list(cls._detectors.keys())
