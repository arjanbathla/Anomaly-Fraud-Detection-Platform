"""Streamlit web UI for the anomaly detection platform."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="Anomaly Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000/api/v1"

# Authentication
def authenticate():
    """Simple authentication for demo purposes."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            # Simple demo authentication
            if username in ["admin", "analyst", "data_engineer", "model_owner"]:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        return False
    
    return True

def main():
    """Main application."""
    if not authenticate():
        return
    
    # Sidebar
    st.sidebar.title("🔍 Anomaly Detection Platform")
    st.sidebar.write(f"Welcome, {st.session_state.username}!")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Dashboard", "Alerts", "Cases", "Models", "Data Sources", "Analytics"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Alerts":
        show_alerts()
    elif page == "Cases":
        show_cases()
    elif page == "Models":
        show_models()
    elif page == "Data Sources":
        show_data_sources()
    elif page == "Analytics":
        show_analytics()

def show_dashboard():
    """Show the main dashboard."""
    st.title("📊 Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", "1,234", "12%")
    
    with col2:
        st.metric("Open Cases", "56", "3%")
    
    with col3:
        st.metric("Models Active", "8", "0%")
    
    with col4:
        st.metric("Data Sources", "12", "1%")
    
    # Charts
    st.subheader("Alert Trends")
    
    # Mock data for demo
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    alert_counts = [10, 15, 8, 20, 12, 18, 25, 30, 22, 16, 14, 19, 23, 17, 21, 26, 28, 24, 20, 18, 15, 12, 16, 19, 22, 25, 27, 24, 20, 18]
    
    df_trends = pd.DataFrame({
        'Date': dates,
        'Alerts': alert_counts
    })
    
    fig = px.line(df_trends, x='Date', y='Alerts', title='Daily Alert Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent alerts
    st.subheader("Recent Alerts")
    
    alerts_data = [
        {"ID": "ALERT-001", "Entity": "user_123", "Score": 0.95, "Status": "Open", "Time": "2 min ago"},
        {"ID": "ALERT-002", "Entity": "user_456", "Score": 0.87, "Status": "Investigating", "Time": "15 min ago"},
        {"ID": "ALERT-003", "Entity": "user_789", "Score": 0.92, "Status": "Open", "Time": "1 hour ago"},
    ]
    
    df_alerts = pd.DataFrame(alerts_data)
    st.dataframe(df_alerts, use_container_width=True)

def show_alerts():
    """Show alerts management page."""
    st.title("🚨 Alerts")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("Status", ["All", "Open", "Investigating", "Resolved"])
    
    with col2:
        score_filter = st.slider("Min Score", 0.0, 1.0, 0.0)
    
    with col3:
        date_filter = st.date_input("Date", datetime.now())
    
    # Alerts table
    st.subheader("Alert List")
    
    # Mock alerts data
    alerts_data = []
    for i in range(50):
        alerts_data.append({
            "ID": f"ALERT-{i+1:03d}",
            "Entity ID": f"user_{i+1}",
            "Score": round(0.5 + (i % 50) * 0.01, 3),
            "Status": ["Open", "Investigating", "Resolved"][i % 3],
            "Created": (datetime.now() - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M"),
            "Assignee": f"analyst_{i % 5 + 1}" if i % 3 != 2 else None
        })
    
    df_alerts = pd.DataFrame(alerts_data)
    
    # Apply filters
    if status_filter != "All":
        df_alerts = df_alerts[df_alerts["Status"] == status_filter]
    
    df_alerts = df_alerts[df_alerts["Score"] >= score_filter]
    
    st.dataframe(df_alerts, use_container_width=True)
    
    # Alert details
    if st.checkbox("Show Alert Details"):
        selected_alert = st.selectbox("Select Alert", df_alerts["ID"].tolist())
        
        if selected_alert:
            st.subheader(f"Alert Details: {selected_alert}")
            
            alert_info = df_alerts[df_alerts["ID"] == selected_alert].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Entity ID:** {alert_info['Entity ID']}")
                st.write(f"**Score:** {alert_info['Score']}")
                st.write(f"**Status:** {alert_info['Status']}")
            
            with col2:
                st.write(f"**Created:** {alert_info['Created']}")
                st.write(f"**Assignee:** {alert_info['Assignee'] or 'Unassigned'}")
            
            # Feature importance chart
            st.subheader("Feature Importance")
            
            features = ["amount", "frequency", "location", "time_of_day", "merchant_category"]
            importance = [0.35, 0.25, 0.20, 0.15, 0.05]
            
            fig = px.bar(
                x=importance, 
                y=features, 
                orientation='h',
                title="Top Contributing Features"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

def show_cases():
    """Show case management page."""
    st.title("📋 Cases")
    
    # Case creation
    with st.expander("Create New Case"):
        case_title = st.text_input("Case Title")
        case_description = st.text_area("Description")
        priority = st.selectbox("Priority", [1, 2, 3, 4, 5])
        assignee = st.selectbox("Assignee", ["analyst_1", "analyst_2", "analyst_3"])
        
        if st.button("Create Case"):
            st.success("Case created successfully!")
    
    # Cases list
    st.subheader("Active Cases")
    
    cases_data = [
        {"ID": "CASE-001", "Title": "High Value Transaction Investigation", "Priority": 1, "Status": "Open", "Assignee": "analyst_1"},
        {"ID": "CASE-002", "Title": "Suspicious Login Pattern", "Priority": 2, "Status": "Investigating", "Assignee": "analyst_2"},
        {"ID": "CASE-003", "Title": "Unusual Merchant Activity", "Priority": 3, "Status": "Open", "Assignee": None},
    ]
    
    df_cases = pd.DataFrame(cases_data)
    st.dataframe(df_cases, use_container_width=True)

def show_models():
    """Show model management page."""
    st.title("🤖 Models")
    
    # Model overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        
        # Mock performance data
        models = ["Credit Card Fraud", "Network Anomaly", "Transaction Risk"]
        precision = [0.95, 0.87, 0.92]
        recall = [0.88, 0.91, 0.85]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Precision', x=models, y=precision))
        fig.add_trace(go.Bar(name='Recall', x=models, y=recall))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Status")
        
        model_status = [
            {"Model": "Credit Card Fraud", "Status": "Production", "Version": "2.1.0"},
            {"Model": "Network Anomaly", "Status": "Staging", "Version": "1.5.2"},
            {"Model": "Transaction Risk", "Status": "Production", "Version": "3.0.1"},
        ]
        
        df_status = pd.DataFrame(model_status)
        st.dataframe(df_status, use_container_width=True)
    
    # Model training
    st.subheader("Train New Model")
    
    with st.form("model_training"):
        model_name = st.text_input("Model Name")
        algorithm = st.selectbox("Algorithm", ["Isolation Forest", "One-Class SVM", "Autoencoder", "XGBoost"])
        training_data = st.file_uploader("Training Data", type=['csv', 'parquet'])
        
        submitted = st.form_submit_button("Start Training")
        
        if submitted:
            st.success("Model training started!")

def show_data_sources():
    """Show data sources page."""
    st.title("📊 Data Sources")
    
    # Data source status
    sources_data = [
        {"Name": "Credit Card Transactions", "Type": "Kafka", "Status": "Active", "Records/min": 1250},
        {"Name": "Network Logs", "Type": "CSV", "Status": "Active", "Records/min": 890},
        {"Name": "User Behavior", "Type": "API", "Status": "Inactive", "Records/min": 0},
    ]
    
    df_sources = pd.DataFrame(sources_data)
    st.dataframe(df_sources, use_container_width=True)
    
    # Data flow diagram
    st.subheader("Data Flow")
    
    # Simple diagram using text
    st.code("""
    Kafka Topic → Feature Pipeline → Model Scoring → Alert Generation
         ↓              ↓                ↓              ↓
    Raw Data → Processed Features → Predictions → Notifications
    """)

def show_analytics():
    """Show analytics and reporting page."""
    st.title("📈 Analytics")
    
    # Time period selector
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Analytics charts
    st.subheader("Alert Distribution by Score")
    
    # Mock data
    score_ranges = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    counts = [45, 32, 28, 15, 8]
    
    fig = px.pie(values=counts, names=score_ranges, title="Alert Score Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # False positive analysis
    st.subheader("False Positive Analysis")
    
    fp_data = pd.DataFrame({
        'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        'False Positives': [12, 8, 15, 10],
        'True Positives': [88, 92, 85, 90]
    })
    
    fig = px.bar(fp_data, x='Week', y=['False Positives', 'True Positives'], 
                 title='False Positive vs True Positive Trends')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
