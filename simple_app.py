#!/usr/bin/env python3
"""Simplified version of the anomaly detection platform for testing."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

# Page configuration
st.set_page_config(
    page_title="Anomaly Detection Platform - Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_sample_data(n_samples=1000, anomaly_ratio=0.1):
    """Generate sample data for demonstration."""
    # Generate normal data
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomaly = n_samples - n_normal
    
    # Normal data (multivariate normal distribution)
    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0, 0],
        cov=[[1, 0.3, 0.2, 0.1], [0.3, 1, 0.4, 0.2], [0.2, 0.4, 1, 0.3], [0.1, 0.2, 0.3, 1]],
        size=n_normal
    )
    
    # Anomaly data (outliers)
    anomaly_data = np.random.multivariate_normal(
        mean=[3, 3, 3, 3],
        cov=[[2, 0.5, 0.3, 0.2], [0.5, 2, 0.6, 0.4], [0.3, 0.6, 2, 0.5], [0.2, 0.4, 0.5, 2]],
        size=n_anomaly
    )
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    df['is_anomaly'] = y
    df['entity_id'] = [f'entity_{i}' for i in range(len(df))]
    df['timestamp'] = pd.date_range(start=datetime.now() - timedelta(days=30), periods=len(df), freq='H')
    
    return df

def train_anomaly_model(data):
    """Train an isolation forest model."""
    # Prepare features
    features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    X = data[features].values
    
    # Train model
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    
    # Get predictions
    scores = -model.decision_function(X)  # Convert to positive scores
    predictions = model.predict(X)
    
    # Convert predictions to anomaly labels
    anomaly_labels = (predictions == -1).astype(int)
    
    return model, scores, anomaly_labels

def main():
    """Main application."""
    st.title("🔍 Anomaly Detection Platform - Demo")
    st.markdown("This is a simplified demo of the anomaly detection platform.")
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Data generation controls
    st.sidebar.subheader("Data Generation")
    n_samples = st.sidebar.slider("Number of samples", 100, 5000, 1000)
    anomaly_ratio = st.sidebar.slider("Anomaly ratio", 0.01, 0.3, 0.1)
    
    # Generate data
    if st.sidebar.button("Generate New Data"):
        st.session_state.data = generate_sample_data(n_samples, anomaly_ratio)
        st.session_state.model_trained = False
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = generate_sample_data(n_samples, anomaly_ratio)
        st.session_state.model_trained = False
    
    # Train model
    if st.sidebar.button("Train Model") or not st.session_state.model_trained:
        with st.spinner("Training anomaly detection model..."):
            model, scores, predictions = train_anomaly_model(st.session_state.data)
            st.session_state.model = model
            st.session_state.scores = scores
            st.session_state.predictions = predictions
            st.session_state.model_trained = True
        st.success("Model trained successfully!")
    
    # Main content
    data = st.session_state.data
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
    
    with col2:
        st.metric("True Anomalies", int(data['is_anomaly'].sum()))
    
    with col3:
        if st.session_state.model_trained:
            st.metric("Detected Anomalies", int(st.session_state.predictions.sum()))
        else:
            st.metric("Detected Anomalies", "Train model first")
    
    with col4:
        if st.session_state.model_trained:
            accuracy = (st.session_state.predictions == data['is_anomaly']).mean()
            st.metric("Accuracy", f"{accuracy:.2%}")
        else:
            st.metric("Accuracy", "Train model first")
    
    # Data visualization
    st.subheader("Data Distribution")
    
    # Feature distributions
    fig = px.histogram(data, x='feature_1', color='is_anomaly', 
                       title='Feature 1 Distribution by Anomaly Status',
                       labels={'is_anomaly': 'Is Anomaly'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 2D scatter plot
    fig = px.scatter(data, x='feature_1', y='feature_2', color='is_anomaly',
                     title='Feature 1 vs Feature 2',
                     labels={'is_anomaly': 'Is Anomaly'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly scores if model is trained
    if st.session_state.model_trained:
        st.subheader("Anomaly Scores")
        
        # Add scores to data
        data_with_scores = data.copy()
        data_with_scores['anomaly_score'] = st.session_state.scores
        data_with_scores['predicted_anomaly'] = st.session_state.predictions
        
        # Score distribution
        fig = px.histogram(data_with_scores, x='anomaly_score', 
                           title='Distribution of Anomaly Scores')
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series of scores
        fig = px.line(data_with_scores, x='timestamp', y='anomaly_score',
                      title='Anomaly Scores Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top anomalies
        st.subheader("Top Anomalies")
        top_anomalies = data_with_scores.nlargest(10, 'anomaly_score')
        st.dataframe(top_anomalies[['entity_id', 'anomaly_score', 'predicted_anomaly', 'is_anomaly']])
    
    # Model performance
    if st.session_state.model_trained:
        st.subheader("Model Performance")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(data['is_anomaly'], st.session_state.predictions)
        
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(data['is_anomaly'], st.session_state.predictions)
        recall = recall_score(data['is_anomaly'], st.session_state.predictions)
        f1 = f1_score(data['is_anomaly'], st.session_state.predictions)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision", f"{precision:.3f}")
        with col2:
            st.metric("Recall", f"{recall:.3f}")
        with col3:
            st.metric("F1 Score", f"{f1:.3f}")
    
    # Real-time scoring demo
    st.subheader("Real-time Scoring Demo")
    
    with st.form("scoring_form"):
        st.write("Enter values for a new record to score:")
        
        col1, col2 = st.columns(2)
        with col1:
            feature_1 = st.number_input("Feature 1", value=0.0)
            feature_2 = st.number_input("Feature 2", value=0.0)
        with col2:
            feature_3 = st.number_input("Feature 3", value=0.0)
            feature_4 = st.number_input("Feature 4", value=0.0)
        
        submitted = st.form_submit_button("Score Record")
        
        if submitted and st.session_state.model_trained:
            # Score the new record
            new_record = np.array([[feature_1, feature_2, feature_3, feature_4]])
            score = -st.session_state.model.decision_function(new_record)[0]
            is_anomaly = st.session_state.model.predict(new_record)[0] == -1
            
            st.success(f"Anomaly Score: {score:.3f}")
            if is_anomaly:
                st.error("🚨 ANOMALY DETECTED!")
            else:
                st.success("✅ Normal record")
        elif submitted:
            st.warning("Please train the model first!")

if __name__ == "__main__":
    main()
