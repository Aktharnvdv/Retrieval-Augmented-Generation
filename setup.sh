#!/bin/bash

# Install Python packages
pip install -r requirements.txt

# Build and start Docker containers
docker-compose up --build
