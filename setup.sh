#!/bin/bash

# Update package lists (optional, uncomment if needed)
# echo "Updating package lists..."
# sudo apt-get update

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start Docker services
echo "Starting Docker services..."
sudo docker-compose up -d

# Run Streamlit application
echo "Running Streamlit application..."
streamlit run streamlit_app.py
