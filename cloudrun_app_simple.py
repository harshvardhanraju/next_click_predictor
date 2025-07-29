"""
Next Click Predictor - Simplified Cloud Run Backend 
Fallback version that doesn't require heavy ML dependencies
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with Cloud Run optimizations
app = FastAPI(
    title="Next Click Predictor API - Simple",
    description="AI-powered click prediction service (fallback mode)",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Next Click Predictor",
        "version": "2.1.0",
        "platform": "Google Cloud Run",
        "ml_enabled": False,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "fallback"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run monitoring"""
    return {
        "status": "healthy",
        "platform": "Google Cloud Run",
        "ml_status": "fallback",
        "timestamp": datetime.now().isoformat(),
        "ready": True
    }

@app.post("/predict")
async def predict_next_click(
    file: UploadFile = File(..., description="Screenshot image (PNG/JPG)"),
    user_attributes: str = Form(..., description="JSON string of user attributes"),
    task_description: str = Form(..., description="User's task description")
):
    """
    Predict next click location using intelligent fallback
    """
    start_time = datetime.now()
    
    try:
        # Validate inputs
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="File must be PNG, JPG, or JPEG")
        
        try:
            user_attrs = json.loads(user_attributes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in user_attributes")
        
        if not task_description.strip():
            raise HTTPException(status_code=400, detail="Task description cannot be empty")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        
        logger.info(f"Processing prediction: file={file.filename} ({file_size} bytes)")
        
        # Generate intelligent fallback prediction
        prediction_result = generate_intelligent_fallback(
            user_attrs, task_description, file_hash
        )
        
        # Add processing metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        prediction_result.update({
            "processing_time": round(processing_time, 2),
            "file_hash": file_hash,
            "ml_method": "intelligent_fallback",
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Prediction completed in {processing_time:.2f}s")
        
        return prediction_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction processing failed: {str(e)}"
        )

def generate_intelligent_fallback(
    user_attrs: Dict[str, Any],
    task_description: str,
    file_hash: str
) -> Dict[str, Any]:
    """
    Generate intelligent prediction based on task context
    """
    # Determine element based on task
    element_type, element_text = determine_element_from_task(task_description)
    
    # Generate realistic coordinates
    coordinates = generate_smart_coordinates(element_type)
    
    # Calculate confidence
    confidence = calculate_task_confidence(user_attrs.get('tech_savviness', 'medium'), task_description)
    
    # Create realistic UI element
    ui_element = {
        "id": f"element_{file_hash}",
        "type": element_type.lower(),
        "text": element_text,
        "x": coordinates["x"],
        "y": coordinates["y"],
        "width": coordinates["width"],
        "height": coordinates["height"],
        "bbox": [
            coordinates["x"],
            coordinates["y"],
            coordinates["x"] + coordinates["width"],
            coordinates["y"] + coordinates["height"]
        ],
        "center": [
            coordinates["x"] + coordinates["width"] // 2,
            coordinates["y"] + coordinates["height"] // 2
        ],
        "confidence": confidence,
        "prominence": 0.8 if element_type.lower() == "button" else 0.6,
        "visibility": True
    }
    
    return {
        "elements": [ui_element],
        "prediction": {
            "element_id": ui_element["id"],
            "element_type": ui_element["type"],
            "element_text": ui_element["text"],
            "click_probability": confidence,
            "x": ui_element["x"],
            "y": ui_element["y"],
            "width": ui_element["width"],
            "height": ui_element["height"],
            "confidence": confidence
        },
        "confidence_score": confidence,
        "explanation": f"Intelligent prediction based on task: {task_description[:50]}...",
        "ml_metadata": {
            "total_elements": 1,
            "processing_method": "intelligent_fallback",
            "context_aware": True
        }
    }

def determine_element_from_task(task: str) -> tuple:
    """Determine element type and text based on task description"""
    task_lower = task.lower()
    
    # Common patterns
    if any(word in task_lower for word in ['buy', 'purchase', 'checkout', 'pay']):
        return "button", "Buy Now"
    elif any(word in task_lower for word in ['login', 'sign in', 'log in']):
        return "button", "Sign In"
    elif any(word in task_lower for word in ['register', 'sign up', 'signup']):
        return "button", "Sign Up"
    elif any(word in task_lower for word in ['search', 'find', 'look']):
        return "form", "Search"
    elif any(word in task_lower for word in ['continue', 'next', 'proceed']):
        return "button", "Continue"
    elif any(word in task_lower for word in ['submit', 'send', 'save']):
        return "button", "Submit"
    elif any(word in task_lower for word in ['close', 'cancel', 'back']):
        return "button", "Close"
    else:
        return "button", "Click Here"

def generate_smart_coordinates(element_type: str) -> Dict[str, int]:
    """Generate realistic coordinates based on element type"""
    if element_type.lower() == "button":
        return {
            "x": 400,
            "y": 300,
            "width": 120,
            "height": 40
        }
    else:  # form
        return {
            "x": 200,
            "y": 200,
            "width": 300,
            "height": 40
        }

def calculate_task_confidence(tech_level: str, task: str) -> float:
    """Calculate confidence based on user tech level and task complexity"""
    base_confidence = 0.7
    
    # Adjust for tech level
    if tech_level == "high":
        base_confidence += 0.1
    elif tech_level == "low":
        base_confidence -= 0.1
    
    # Adjust for task clarity
    task_words = len(task.split())
    if task_words > 10:  # Detailed task
        base_confidence += 0.1
    elif task_words < 3:  # Vague task
        base_confidence -= 0.1
    
    return min(0.95, max(0.3, base_confidence))

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Cloud Run provides this)
    port = int(os.environ.get("PORT", 8080))
    
    logger.info(f"Starting Simple Next Click Predictor on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )