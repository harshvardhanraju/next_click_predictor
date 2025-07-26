#!/bin/bash

# Railway startup script
# Handles PORT environment variable with fallback

# Set default port if PORT is not set
if [ -z "$PORT" ]; then
    echo "PORT environment variable not set, using default 8000"
    PORT=8000
else
    echo "Using PORT from environment: $PORT"
fi

# Start the application
echo "Starting uvicorn on port $PORT"
exec python -m uvicorn src.web_service:app --host 0.0.0.0 --port $PORT