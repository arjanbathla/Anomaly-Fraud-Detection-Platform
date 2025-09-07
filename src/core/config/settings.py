"""Configuration settings for the anomaly detection platform."""

from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "Anomaly & Fraud Detection Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/anomaly_detection"
    redis_url: str = "redis://localhost:6379"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "anomaly_detection"
    
    # Feature Store
    feature_store_path: str = "./feature_store"
    
    # Monitoring
    prometheus_port: int = 9090
    grafana_url: str = "http://localhost:3000"
    
    # Alerting
    slack_webhook_url: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    
    # Data Sources
    kafka_bootstrap_servers: List[str] = ["localhost:9092"]
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    
    # PII Handling
    pii_fields: List[str] = ["pan", "email", "phone", "ssn", "credit_card"]
    pii_masking_strategy: str = "hash"  # hash, tokenize, truncate
    
    # Model Settings
    model_retrain_frequency_hours: int = 24
    drift_threshold: float = 0.1
    alert_threshold: float = 0.8
    
    # Data Retention
    data_retention_days: int = 395  # 13 months
    audit_log_retention_days: int = 2555  # 7 years
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
