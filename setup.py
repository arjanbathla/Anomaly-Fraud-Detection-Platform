#!/usr/bin/env python3
"""Setup script for the Anomaly Detection Platform."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Anomaly Detection Platform...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "logs", "model_registry", "feature_store", "monitoring"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Copy environment file
    if not Path(".env").exists():
        if Path(".env.example").exists():
            run_command("cp .env.example .env", "Creating environment file")
            print("✅ Environment file created. Please edit .env with your configuration.")
        else:
            print("⚠️  No .env.example found. Please create .env manually.")
    
    # Make scripts executable
    scripts = ["scripts/train_model.py", "scripts/start_platform.py"]
    for script in scripts:
        if Path(script).exists():
            os.chmod(script, 0o755)
            print(f"✅ Made executable: {script}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your configuration")
    print("2. Start the platform: python scripts/start_platform.py")
    print("3. Or use Docker: docker-compose up -d")
    print("\nAccess points:")
    print("- API Documentation: http://localhost:8000/docs")
    print("- Web UI: http://localhost:8501")
    print("- MLflow UI: http://localhost:5000")

if __name__ == "__main__":
    main()
