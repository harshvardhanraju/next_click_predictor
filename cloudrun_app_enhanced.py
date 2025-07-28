"""
Next Click Predictor - Enhanced Cloud Run Backend with Real ML
Integrates the advanced UI detection system with Cloud Run optimization
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
import cv2
import numpy as np

# Add src directory to path for ML imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import ML components with error handling
ML_AVAILABLE = False
try:
    from screenshot_processor import ScreenshotProcessor
    from next_click_predictor import NextClickPredictor
    ML_AVAILABLE = True
    logging.info("✅ ML components loaded successfully")
except ImportError as e:
    logging.warning(f"⚠️ ML components not available: {e}")
    logging.info("📦 Falling back to optimized mock predictions")

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with Cloud Run optimizations
app = FastAPI(
    title="Next Click Predictor API - Enhanced",
    description="AI-powered click prediction service with advanced ML capabilities",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3002",
        "https://*.vercel.app",
        "https://next-click-predictor-frontend.vercel.app",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize ML predictor if available
ml_predictor = None
if ML_AVAILABLE:
    try:
        ml_predictor = NextClickPredictor()
        logger.info("🚀 ML predictor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML predictor: {e}")
        ml_predictor = None

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Next Click Predictor Enhanced",
        "version": "3.0.0",
        "platform": "Google Cloud Run",
        "ml_enabled": ML_AVAILABLE and ml_predictor is not None,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "advanced_ui_detection": ML_AVAILABLE,
            "modern_pattern_recognition": ML_AVAILABLE,
            "bayesian_prediction": ML_AVAILABLE,
            "ocr_text_extraction": ML_AVAILABLE
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "analyze": "/analyze-screenshot",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run monitoring"""
    return {
        "status": "healthy",
        "platform": "Google Cloud Run",
        "ml_status": "available" if (ML_AVAILABLE and ml_predictor) else "fallback",
        "timestamp": datetime.now().isoformat(),
        "memory": "Optimized for Cloud Run",
        "ready": True
    }

@app.post("/predict")
async def predict_next_click(
    file: UploadFile = File(..., description="Screenshot image (PNG/JPG)"),
    user_attributes: str = Form(..., description="JSON string of user attributes"),
    task_description: str = Form(..., description="User's task description")
):
    """
    Predict next click location using enhanced ML or intelligent fallback
    """
    start_time = datetime.now()
    
    try:
        # Validate file and inputs
        await validate_inputs(file, user_attributes, task_description)
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        
        # Parse user attributes
        user_attrs = json.loads(user_attributes)
        
        logger.info(f"🔍 Processing prediction: file={file.filename} ({file_size} bytes), "
                   f"hash={file_hash}, ml_enabled={ML_AVAILABLE and ml_predictor is not None}")
        
        # Choose prediction method
        if ML_AVAILABLE and ml_predictor:
            # Use real ML prediction
            prediction_result = await ml_prediction(
                file_content, file.filename, user_attrs, task_description
            )
        else:
            # Use intelligent fallback
            prediction_result = generate_intelligent_fallback(
                file_content, file.filename, file_hash, user_attrs, task_description
            )
        
        # Add processing metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        prediction_result.update({
            "processing_time": round(processing_time, 2),
            "file_hash": file_hash,
            "ml_method": "advanced_detection" if (ML_AVAILABLE and ml_predictor) else "intelligent_fallback",
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"✅ Prediction completed in {processing_time:.2f}s with "
                   f"{prediction_result['confidence_score']:.0%} confidence")
        
        return prediction_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction processing failed: {str(e)}"
        )

@app.post("/analyze-screenshot")
async def analyze_screenshot_only(
    file: UploadFile = File(..., description="Screenshot image (PNG/JPG)")
):
    """
    Analyze screenshot for UI elements without prediction (debugging endpoint)
    """
    try:
        # Validate file
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        file_content = await file.read()
        
        if ML_AVAILABLE and ml_predictor:
            # Use real screenshot analysis
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                # Analyze screenshot only
                analysis_result = ml_predictor.analyze_screenshot_only(temp_path)
                
                return {
                    "analysis_method": "advanced_ml",
                    "ui_elements": analysis_result.get("ui_elements", []),
                    "total_elements": analysis_result.get("total_elements", 0),
                    "screen_dimensions": analysis_result.get("screen_dimensions", []),
                    "element_types": analysis_result.get("element_types", {}),
                    "timestamp": datetime.now().isoformat()
                }
            finally:
                os.unlink(temp_path)
        else:
            # Simple analysis fallback
            return {
                "analysis_method": "fallback",
                "message": "ML components not available",
                "mock_elements": 3,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Screenshot analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def validate_inputs(file: UploadFile, user_attributes: str, task_description: str):
    """Validate API inputs"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="File must be PNG, JPG, or JPEG")
    
    try:
        json.loads(user_attributes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in user_attributes")
    
    if not task_description.strip():
        raise HTTPException(status_code=400, detail="Task description cannot be empty")

async def ml_prediction(
    file_content: bytes,
    filename: str, 
    user_attrs: Dict[str, Any],
    task_description: str
) -> Dict[str, Any]:
    """
    Generate prediction using real ML components
    """
    # Save file temporarily for processing
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name
    
    try:
        logger.info("🤖 Running ML prediction...")
        
        # Use the real predictor
        result = ml_predictor.predict_next_click(
            screenshot_path=temp_path,
            user_attributes=user_attrs,
            task_description=task_description,
            return_detailed=True
        )
        
        # Convert result to API format
        return format_ml_result(result)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def format_ml_result(ml_result) -> Dict[str, Any]:
    """
    Convert ML prediction result to API format
    """
    try:
        # Get top prediction
        top_pred = ml_result.top_prediction
        
        # Format UI elements for frontend
        ui_elements = []
        for element in ml_result.ui_elements:
            ui_elements.append({
                "id": element.get("id", "unknown"),
                "type": element.get("type", "unknown"),
                "text": element.get("text", ""),
                "x": element.get("bbox", [0, 0, 100, 100])[0],
                "y": element.get("bbox", [0, 0, 100, 100])[1],
                "width": element.get("bbox", [0, 0, 100, 100])[2] - element.get("bbox", [0, 0, 100, 100])[0],
                "height": element.get("bbox", [0, 0, 100, 100])[3] - element.get("bbox", [0, 0, 100, 100])[1],
                "bbox": element.get("bbox", [0, 0, 100, 100]),
                "center": element.get("center", [50, 50]),
                "confidence": element.get("prominence", 0.5),
                "prominence": element.get("prominence", 0.5),
                "visibility": element.get("visibility", True)
            })
        
        # Create top prediction element
        top_element = None
        if ui_elements:
            # Find element matching top prediction
            for elem in ui_elements:
                if elem["id"] == top_pred.get("element_id"):
                    top_element = elem
                    break
            
            # If no match, use first element
            if not top_element:
                top_element = ui_elements[0]
        
        if not top_element:
            # Create fallback element
            top_element = {
                "id": "fallback_element",
                "type": "button",
                "text": "Click Here",
                "x": 300, "y": 200, "width": 120, "height": 40,
                "bbox": [300, 200, 420, 240],
                "center": [360, 220],
                "confidence": 0.5,
                "prominence": 0.5,
                "visibility": True
            }
        
        return {
            "elements": ui_elements,
            "prediction": {
                "element_id": top_element["id"],
                "element_type": top_element["type"],
                "element_text": top_element["text"],
                "click_probability": top_pred.get("click_probability", 0.7),
                "x": top_element["x"],
                "y": top_element["y"],
                "width": top_element["width"],
                "height": top_element["height"],
                "confidence": top_pred.get("confidence", 0.7)
            },
            "confidence_score": ml_result.confidence_score,
            "explanation": ml_result.explanation.get("main_explanation", "ML-powered prediction"),
            "ml_metadata": {
                "total_elements": len(ui_elements),
                "processing_method": "advanced_ml_detection",
                "bayesian_inference": True,
                "modern_ui_patterns": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error formatting ML result: {e}")
        # Return safe fallback
        return generate_safe_fallback()

def generate_intelligent_fallback(
    file_content: bytes,
    filename: str,
    file_hash: str,
    user_attrs: Dict[str, Any],
    task_description: str
) -> Dict[str, Any]:
    """
    Generate intelligent prediction when ML is not available
    """
    logger.info("🔄 Using intelligent fallback prediction")
    
    # Try to get image dimensions for better positioning
    image_width, image_height = 1200, 800  # Default
    try:
        # Decode image to get dimensions
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            image_height, image_width = img.shape[:2]
            logger.info(f"📐 Detected image dimensions: {image_width}x{image_height}")
    except Exception as e:
        logger.debug(f"Could not decode image dimensions: {e}")
    
    # Generate context-aware prediction
    tech_level = user_attrs.get('tech_savviness', 'medium')
    device_type = user_attrs.get('device_type', 'desktop')
    
    # Determine element based on task
    element_type, element_text = determine_element_from_task(task_description, device_type)
    coordinates = generate_smart_coordinates(image_width, image_height, element_type, device_type)
    confidence = calculate_task_confidence(tech_level, task_description)
    
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

def determine_element_from_task(task: str, device: str) -> tuple:
    """Determine element type and text based on task description"""
    task_lower = task.lower()
    
    # Common patterns
    if any(word in task_lower for word in ['buy', 'purchase', 'checkout', 'pay']):
        return "button", "Buy Now" if device == "mobile" else "Add to Cart"
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

def generate_smart_coordinates(width: int, height: int, element_type: str, device: str) -> Dict[str, int]:
    """Generate realistic coordinates based on context"""
    if device == "mobile":
        # Mobile-optimized positioning
        if element_type.lower() == "button":
            return {
                "x": max(20, width // 2 - 60),
                "y": max(height - 100, height // 2),
                "width": 120,
                "height": 44
            }
        else:  # form
            return {
                "x": 20,
                "y": height // 3,
                "width": width - 40,
                "height": 40
            }
    else:
        # Desktop positioning
        if element_type.lower() == "button":
            return {
                "x": max(50, width // 2 - 75),
                "y": max(50, height // 2),
                "width": 150,
                "height": 36
            }
        else:  # form
            return {
                "x": width // 4,
                "y": height // 3,
                "width": width // 2,
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

def generate_safe_fallback() -> Dict[str, Any]:
    """Generate safe fallback when everything else fails"""
    return {
        "elements": [{
            "id": "safe_fallback",
            "type": "button",
            "text": "Click Here",
            "x": 300, "y": 200, "width": 100, "height": 40,
            "bbox": [300, 200, 400, 240],
            "center": [350, 220],
            "confidence": 0.5,
            "prominence": 0.5,
            "visibility": True
        }],
        "prediction": {
            "element_id": "safe_fallback",
            "element_type": "button",
            "element_text": "Click Here",
            "click_probability": 0.5,
            "x": 300, "y": 200, "width": 100, "height": 40,
            "confidence": 0.5
        },
        "confidence_score": 0.5,
        "explanation": "Safe fallback prediction",
        "ml_metadata": {
            "total_elements": 1,
            "processing_method": "safe_fallback"
        }
    }

# Health monitoring endpoint
@app.get("/metrics")
async def get_metrics():
    """Get service metrics for monitoring"""
    return {
        "ml_status": "available" if (ML_AVAILABLE and ml_predictor) else "fallback",
        "capabilities": {
            "advanced_ui_detection": ML_AVAILABLE,
            "screenshot_processing": ML_AVAILABLE,
            "bayesian_prediction": ML_AVAILABLE
        },
        "system_info": {
            "python_version": sys.version,
            "platform": "Google Cloud Run"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Cloud Run provides this)
    port = int(os.environ.get("PORT", 8080))
    
    logger.info(f"🚀 Starting Enhanced Next Click Predictor on port {port}")
    logger.info(f"🤖 ML components: {'Available' if ML_AVAILABLE else 'Fallback mode'}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )