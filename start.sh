#!/bin/bash

# Railway startup script with verbose logging
echo "=== Railway Startup Script ==="
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Files in directory: $(ls -la)"

# Set default port if PORT is not set
if [ -z "$PORT" ]; then
    echo "PORT environment variable not set, using default 8000"
    PORT=8000
else
    echo "Using PORT from environment: $PORT"
fi

# Check if src directory exists
if [ ! -d "src" ]; then
    echo "ERROR: src directory not found!"
    exit 1
fi

# List Python packages
echo "Installed packages:"
pip list | head -10

# Start the application with more verbose output
echo "Starting uvicorn on port $PORT with host 0.0.0.0"
exec python -m uvicorn src.web_service:app --host 0.0.0.0 --port $PORT --log-level info