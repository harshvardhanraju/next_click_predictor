#!/bin/bash

echo "=== Basic Python Server Test ==="
echo "PORT from Railway: $PORT"
echo "Current working directory: $(pwd)"

if [ -z "$PORT" ]; then
    echo "WARNING: PORT not set by Railway, using 8000"
    export PORT=8000
fi

echo "Starting basic Python server on port $PORT"
python main.py