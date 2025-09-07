"""Core data models for the anomaly detection platform."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum
import uuid


class UserRole(str, Enum):
    """User roles in the system."""
    ANALYST = "analyst"
    DATA_ENGINEER = "data_engineer"
    MODEL_OWNER = "model_owner"
    ADMIN = "admin"


class AlertStatus(str, Enum):
    """Alert status values."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class ModelType(str, Enum):
    """Model types."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    GRADIENT_BOOSTING = "gradient_boosting"


class DataSourceType(str, Enum):
    """Data source types."""
    CSV = "csv"
    PARQUET = "parquet"
    KAFKA = "kafka"
    MQTT = "mqtt"
    WEBHOOK = "webhook"
    REST_API = "rest_api"


class User(BaseModel):
    """User model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DataSource(BaseModel):
    """Data source configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    source_type: DataSourceType
    config: Dict[str, Any]
    schema: Dict[str, Any]  # JSONSchema
    pii_fields: List[str] = []
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class FeaturePipeline(BaseModel):
    """Feature pipeline configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    transformations: List[Dict[str, Any]]
    entity_keys: List[str]  # e.g., ["user_id", "device_id"]
    window_configs: List[Dict[str, Any]]  # time window aggregations
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Model(BaseModel):
    """Model configuration and metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    model_type: ModelType
    version: str
    config: Dict[str, Any]
    metrics: Dict[str, float] = {}
    tags: List[str] = []
    is_production: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Prediction(BaseModel):
    """Model prediction result."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_version: str
    entity_id: str
    features: Dict[str, Any]
    score: float
    threshold: float
    is_anomaly: bool
    explanations: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Alert(BaseModel):
    """Alert generated from predictions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction_id: str
    entity_id: str
    score: float
    status: AlertStatus = AlertStatus.OPEN
    assignee: Optional[str] = None
    notes: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Case(BaseModel):
    """Case management for alerts."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str
    title: str
    description: str
    status: AlertStatus = AlertStatus.OPEN
    assignee: Optional[str] = None
    priority: int = 1  # 1-5 scale
    evidence: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AuditLog(BaseModel):
    """Audit log entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DataRecord(BaseModel):
    """Raw data record."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = []
    is_valid: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureValue(BaseModel):
    """Feature value in the feature store."""
    entity_id: str
    feature_name: str
    value: Union[float, int, str, bool]
    timestamp: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_id: str
    model_version: str
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    mcc: float
    calibration_score: float
    cost_savings: float
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)


class DriftMetrics(BaseModel):
    """Data and prediction drift metrics."""
    model_id: str
    feature_name: str
    psi_score: float
    ks_statistic: float
    drift_detected: bool
    threshold: float
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
