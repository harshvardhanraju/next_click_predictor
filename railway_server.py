"""
Railway-optimized FastAPI server for Next Click Predictor
Lightweight version that runs without heavy ML dependencies
"""

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print("=== RAILWAY MINIMAL SERVER STARTUP ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"PORT environment variable: {os.environ.get('PORT', 'NOT_SET')}")

app = FastAPI(
    title="Next Click Predictor API (Railway)", 
    version="1.0.0",
    description="Lightweight Railway deployment with mock ML predictions"
)

# CORS configuration for Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "https://next-click-predictor-frontend.vercel.app",
        "https://*.vercel.app",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"📨 {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"✅ Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"❌ Request failed: {str(e)}")
        raise

@app.get("/")
def read_root():
    logger.info("🏠 Root endpoint called")
    return {
        "message": "Next Click Predictor API - Railway Deployment",
        "status": "running",
        "version": "1.0.0-railway",
        "port": os.environ.get("PORT", "8000"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    logger.info("🔍 Health check called")
    return {
        "status": "healthy",
        "service": "next-click-predictor-railway",
        "timestamp": datetime.now().isoformat(),
        "port": os.environ.get("PORT", "8000")
    }

@app.post("/predict")
async def predict_click(
    file: UploadFile = File(...),
    user_attributes: str = Form(...),
    task_description: str = Form(...)
):
    logger.info("🔮 Predict endpoint called")
    logger.info(f"📁 File: {file.filename} ({file.content_type})")
    logger.info(f"👤 User: {user_attributes}")
    logger.info(f"📝 Task: {task_description}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="File must be a PNG, JPG, or JPEG image"
        )
    
    try:
        # Parse user attributes
        user_attrs = json.loads(user_attributes)
        logger.info(f"✅ Parsed user attributes: {user_attrs}")
        
        # Read file to ensure it's valid (but don't process it)
        file_content = await file.read()
        logger.info(f"📄 File size: {len(file_content)} bytes")
        
        # Generate realistic mock prediction based on user attributes
        tech_level = user_attrs.get('tech_savviness', 'medium')
        device = user_attrs.get('device_type', 'desktop')
        age_group = user_attrs.get('age_group', '25-34')
        
        # Create mock UI element based on common patterns
        mock_element = {
            "id": "predicted_element_1",
            "type": "button",
            "element_type": "BUTTON",
            "text": "Submit" if "submit" in task_description.lower() else "Continue",
            "x": 350 if device == "desktop" else 150,
            "y": 200 if device == "desktop" else 300,
            "width": 120 if device == "desktop" else 100,
            "height": 40,
            "bbox": [350, 200, 470, 240] if device == "desktop" else [150, 300, 250, 340],
            "center": [410, 220] if device == "desktop" else [200, 320],
            "confidence": 0.87 if tech_level == "high" else 0.75,
            "prominence": 0.9,
            "visibility": True
        }
        
        # Create prediction with confidence based on user profile
        base_confidence = 0.85
        if tech_level == "high":
            base_confidence = 0.90
        elif tech_level == "low":
            base_confidence = 0.70
            
        mock_prediction = {
            "element_id": "predicted_element_1",
            "element_type": "BUTTON",
            "element_text": mock_element["text"],
            "click_probability": base_confidence,
            "confidence": base_confidence,
            "rank": 1,
            "element": mock_element,
            "reasoning": [
                f"High visual prominence for {device} interface",
                f"User's {tech_level} tech level suggests familiarity",
                "Task description indicates action-oriented goal",
                "Button positioning follows UI best practices"
            ]
        }
        
        # Create comprehensive response
        response = {
            "top_prediction": mock_prediction,
            "all_predictions": [mock_prediction],
            "explanation": {
                "main_explanation": f"Based on {device} interface analysis and {tech_level} user tech level, the {mock_element['text']} button has {base_confidence:.0%} click probability.",
                "key_factors": [
                    {
                        "factor": "Visual Prominence",
                        "weight": 0.35,
                        "description": f"Button positioned prominently for {device} viewport",
                        "importance": 0.35
                    },
                    {
                        "factor": "User Experience Level", 
                        "weight": 0.28,
                        "description": f"User's {tech_level} tech savviness indicates interface familiarity",
                        "importance": 0.28
                    },
                    {
                        "factor": "Task Alignment",
                        "weight": 0.25,
                        "description": "Button text and function align with described task",
                        "importance": 0.25
                    },
                    {
                        "factor": "Device Optimization",
                        "weight": 0.12,
                        "description": f"Interface optimized for {device} interaction patterns",
                        "importance": 0.12
                    }
                ],
                "reasoning_chain": [
                    "🔍 Analyzed uploaded interface screenshot",
                    "📱 Detected UI elements and interaction zones", 
                    "👤 Processed user demographic profile",
                    "🎯 Evaluated task description for user intent",
                    "🧠 Applied Bayesian probability modeling",
                    "✨ Generated prediction with confidence metrics"
                ],
                "confidence_analysis": f"High confidence prediction based on clear {device} UI patterns and {tech_level}-level user experience expectations.",
                "confidence_explanation": f"The {base_confidence:.0%} confidence reflects strong alignment between UI design, user profile ({age_group}, {tech_level} tech level), and task requirements."
            },
            "ui_elements": [mock_element],
            "processing_time": 1.4,
            "confidence_score": base_confidence,
            "metadata": {
                "filename": file.filename,
                "file_size": len(file_content),
                "user_attributes": user_attrs,
                "task_description": task_description,
                "model_version": "railway-mock-1.0",
                "deployment": "railway",
                "timestamp": datetime.now().isoformat(),
                "service_type": "lightweight-mock"
            }
        }
        
        logger.info(f"🎯 Generated prediction with {base_confidence:.0%} confidence")
        return response
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_attributes JSON: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/docs")
def api_docs():
    return {"message": "API documentation available at /docs"}

@app.get("/test")
def test_endpoint():
    logger.info("🧪 Test endpoint called")
    return {
        "message": "Railway deployment test successful",
        "status": "working",
        "environment": dict(os.environ),
        "timestamp": datetime.now().isoformat()
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 RAILWAY SERVER STARTUP COMPLETE")
    logger.info(f"🌐 Server running on port {os.environ.get('PORT', '8000')}")
    
@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("🛑 RAILWAY SERVER SHUTDOWN")

print("✅ RAILWAY_SERVER.PY LOADED SUCCESSFULLY")
print("🚀 FastAPI app configured for Railway deployment")