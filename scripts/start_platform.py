#!/usr/bin/env python3
"""Startup script for the anomaly detection platform."""

import subprocess
import sys
import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import streamlit
        import pandas
        import numpy
        import sklearn
        logger.info("✓ Core dependencies found")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "logs", 
        "model_registry",
        "feature_store",
        "monitoring"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")


def start_services():
    """Start the platform services."""
    logger.info("Starting Anomaly Detection Platform...")
    
    # Start API server
    logger.info("Starting API server...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ])
    
    # Wait a bit for API to start
    time.sleep(5)
    
    # Start Streamlit UI
    logger.info("Starting Streamlit UI...")
    ui_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", 
        "run", "src/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
    
    logger.info("Platform started successfully!")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Web UI: http://localhost:8501")
    
    try:
        # Wait for processes
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down platform...")
        api_process.terminate()
        ui_process.terminate()
        logger.info("Platform stopped.")


def main():
    """Main startup function."""
    logger.info("Anomaly Detection Platform Startup")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Start services
    start_services()


if __name__ == "__main__":
    main()
