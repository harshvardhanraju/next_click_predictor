"""
Ultra-minimal FastAPI server for Railway testing
"""

from fastapi import FastAPI
import os
import json
from datetime import datetime

app = FastAPI(title="Railway Test API")

@app.get("/")
def root():
    return {
        "message": "Railway FastAPI working!",
        "port": os.environ.get("PORT", "8000"),
        "timestamp": datetime.now().isoformat(),
        "status": "healthy"
    }

@app.get("/health")  
def health():
    return {"status": "healthy"}

@app.get("/test")
def test():
    return {
        "message": "Test endpoint working",
        "env_vars": dict(os.environ)
    }

print("Simple Railway server loaded successfully")