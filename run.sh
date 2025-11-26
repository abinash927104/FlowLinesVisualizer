#!/bin/bash

# Flow Line Visualizer - Quick Start Script

echo "🌊 Starting Flow Line Visualizer..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run Streamlit app
echo ""
echo "🚀 Launching Streamlit app..."
echo "The app will open in your default browser."
echo ""
streamlit run app.py

