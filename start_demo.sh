#!/bin/bash

echo "🚀 Starting Anomaly Detection Platform Demo"
echo "=========================================="

# Kill any existing processes
echo "🔄 Stopping existing processes..."
pkill -f "streamlit run simple_app.py" 2>/dev/null
pkill -f "python simple_api.py" 2>/dev/null
sleep 2

# Start API server
echo "🌐 Starting API server on port 8000..."
python simple_api.py &
API_PID=$!

# Wait for API to start
sleep 3

# Start Streamlit UI
echo "🎨 Starting Streamlit UI on port 8501..."
streamlit run simple_app.py --server.port 8501 --server.address 0.0.0.0 &
UI_PID=$!

# Wait for UI to start
sleep 5

echo ""
echo "✅ Platform started successfully!"
echo ""
echo "📋 Access Points:"
echo "   🌐 API Server: http://localhost:8000"
echo "   📚 API Docs: http://localhost:8000/docs"
echo "   🎨 Web UI: http://localhost:8501"
echo ""
echo "🧪 Run tests: python test_platform.py"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID 2>/dev/null
    kill $UI_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
