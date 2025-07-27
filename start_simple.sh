#!/bin/bash

echo "=== Simple Railway Test ==="
echo "PORT: $PORT"
echo "PWD: $(pwd)"

# Use Railway's PORT directly
if [ -z "$PORT" ]; then
    echo "ERROR: PORT not set by Railway!"
    PORT=8000
fi

echo "Starting on port: $PORT"
python -m uvicorn src.test_app:app --host 0.0.0.0 --port $PORT