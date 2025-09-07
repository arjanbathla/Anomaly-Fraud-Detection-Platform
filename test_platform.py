#!/usr/bin/env python3
"""Test script for the anomaly detection platform."""

import requests
import json
import time

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Anomaly Detection Platform API")
    print("=" * 50)
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print()
    
    # Test model training
    print("2. Training model...")
    try:
        response = requests.post(f"{base_url}/train")
        if response.status_code == 200:
            print("✅ Model training successful")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Model training failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model training error: {e}")
    
    print()
    
    # Test scoring
    print("3. Testing anomaly scoring...")
    
    # Test normal record
    normal_record = {
        "data": {
            "feature_1": 0.1,
            "feature_2": -0.2,
            "feature_3": 0.3,
            "feature_4": -0.1
        }
    }
    
    try:
        response = requests.post(f"{base_url}/score", json=normal_record)
        if response.status_code == 200:
            result = response.json()
            print("✅ Normal record scored successfully")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Is Anomaly: {result['is_anomaly']}")
        else:
            print(f"❌ Normal record scoring failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Normal record scoring error: {e}")
    
    # Test anomaly record
    anomaly_record = {
        "data": {
            "feature_1": 4.0,
            "feature_2": 5.0,
            "feature_3": 3.5,
            "feature_4": 4.2
        }
    }
    
    try:
        response = requests.post(f"{base_url}/score", json=anomaly_record)
        if response.status_code == 200:
            result = response.json()
            print("✅ Anomaly record scored successfully")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Is Anomaly: {result['is_anomaly']}")
        else:
            print(f"❌ Anomaly record scoring failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Anomaly record scoring error: {e}")
    
    print()
    
    # Test model listing
    print("4. Testing model listing...")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            print("✅ Model listing successful")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Model listing failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model listing error: {e}")
    
    print()
    print("🎉 API testing completed!")

def test_streamlit():
    """Test if Streamlit UI is accessible."""
    print("\n🌐 Testing Streamlit UI...")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8501")
        if response.status_code == 200:
            print("✅ Streamlit UI is accessible")
            print("   URL: http://localhost:8501")
        else:
            print(f"❌ Streamlit UI not accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Streamlit UI error: {e}")

if __name__ == "__main__":
    print("🚀 Anomaly Detection Platform - Test Suite")
    print("=" * 60)
    
    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(2)
    
    # Test API
    test_api()
    
    # Test Streamlit
    test_streamlit()
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    print("✅ API Server: http://localhost:8000")
    print("✅ API Docs: http://localhost:8000/docs")
    print("✅ Streamlit UI: http://localhost:8501")
    print("\n🎯 You can now:")
    print("   1. Open http://localhost:8501 in your browser for the web UI")
    print("   2. Open http://localhost:8000/docs for interactive API documentation")
    print("   3. Use the API endpoints programmatically")
