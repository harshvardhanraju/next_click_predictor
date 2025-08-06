"""
Ultra-Fast Cloud Run Backend - Production Ready
Minimal dependencies, maximum speed, aggressive timeout protection
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any
import hashlib

# Add local modules
sys.path.insert(0, os.path.dirname(__file__))

# Import our production-optimized predictor
from production_optimized_predictor import ProductionOptimizedPredictor

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Next Click Predictor - Ultra Fast v2.6",
    description="Ultra-fast ML-powered click prediction with aggressive optimization",
    version="2.6.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the ultra-fast predictor"""
    global predictor
    
    try:
        config = {
            'max_elements': 5,  # Process only 5 elements
            'timeout_seconds': 30,  # Hard timeout
            'min_area': 400,
            'max_area': 50000,
        }
        
        predictor = ProductionOptimizedPredictor(config)
        logger.info("✅ Ultra-fast predictor initialized")
        return True
        
    except Exception as e:
        logger.error(f"❌ Predictor initialization failed: {e}")
        return False

# Initialize on startup
initialize_predictor()

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("🚀 Starting Ultra-Fast Next Click Predictor v2.6")
    logger.info(f"🏃 Speed mode: {'Active' if predictor else 'Failed'}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Next Click Predictor Ultra-Fast v2.6",
        "version": "2.6.0",
        "mode": "production_optimized",
        "status": "healthy",
        "predictor_ready": predictor is not None,
        "timestamp": datetime.now().isoformat(),
        "features": {
            "ultra_fast_processing": True,
            "minimal_dependencies": True,
            "aggressive_timeout_protection": True,
            "max_processing_time": "30 seconds",
            "max_elements": 5
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.6.0",
        "predictor_status": "ready" if predictor else "failed",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_next_click(
    file: UploadFile = File(...),
    user_attributes: str = Form(...),
    task_description: str = Form(...)
):
    """Ultra-fast prediction endpoint"""
    start_time = datetime.now()
    
    try:
        # Validate inputs quickly
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        # Parse user attributes
        try:
            user_attrs = json.loads(user_attributes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in user_attributes")
        
        if not task_description.strip():
            raise HTTPException(status_code=400, detail="Task description required")
        
        # Check predictor availability
        if not predictor:
            raise HTTPException(status_code=503, detail="Predictor not available")
        
        # Read and save file
        file_content = await file.read()
        file_size = len(file_content)
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        
        logger.info(f"🔍 Processing: {file.filename} ({file_size} bytes)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            # Run ultra-fast prediction
            prediction_result = predictor.predict_next_click(
                screenshot_path=temp_path,
                user_attributes=user_attrs,
                task_description=task_description
            )
            
            # Format for API response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "elements": [{
                    "id": prediction_result.element_id,
                    "type": prediction_result.element_type,
                    "text": prediction_result.element_text,
                    "x": prediction_result.bbox[0],
                    "y": prediction_result.bbox[1], 
                    "width": prediction_result.bbox[2] - prediction_result.bbox[0],
                    "height": prediction_result.bbox[3] - prediction_result.bbox[1],
                    "bbox": prediction_result.bbox,
                    "center": prediction_result.center,
                    "confidence": prediction_result.confidence,
                    "prominence": prediction_result.click_probability,
                    "visibility": True,
                    "rank": 1
                }],
                "prediction": {
                    "element_id": prediction_result.element_id,
                    "element_type": prediction_result.element_type,
                    "element_text": prediction_result.element_text,
                    "click_probability": prediction_result.click_probability,
                    "x": prediction_result.bbox[0],
                    "y": prediction_result.bbox[1],
                    "width": prediction_result.bbox[2] - prediction_result.bbox[0],
                    "height": prediction_result.bbox[3] - prediction_result.bbox[1],
                    "confidence": prediction_result.confidence
                },
                "confidence_score": prediction_result.confidence,
                "explanation": f"Ultra-fast prediction for: {task_description[:50]}...",
                "ml_metadata": {
                    "total_elements": 1,
                    "processing_method": "ultra_fast_v2.6",
                    "processing_time": prediction_result.processing_time,
                    "ml_method": prediction_result.method,
                    "speed_optimized": True
                },
                "processing_time": processing_time,
                "file_hash": file_hash,
                "timestamp": datetime.now().isoformat(),
                "version": "2.6.0"
            }
            
            logger.info(f"✅ Ultra-fast prediction completed in {processing_time:.2f}s")
            return response
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analyze-screenshot")
async def analyze_screenshot_only(
    file: UploadFile = File(...)
):
    """Ultra-fast screenshot analysis"""
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        # For ultra-fast mode, return minimal analysis
        return {
            "analysis_method": "ultra_fast_v2.6",
            "message": "Ultra-fast mode - use /predict endpoint for full analysis",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Service metrics"""
    return {
        "version": "2.6.0",
        "mode": "ultra_fast",
        "predictor_status": "ready" if predictor else "failed",
        "performance": {
            "max_processing_time": "30 seconds",
            "max_elements_processed": 5,
            "dependencies": "minimal",
            "ocr_enabled": False,
            "ml_complexity": "optimized"
        },
        "features": {
            "contour_detection": True,
            "simple_classification": True,
            "task_matching": True,
            "user_adaptation": True,
            "timeout_protection": True
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    
    logger.info(f"🚀 Starting Ultra-Fast Predictor v2.6 on port {port}")
    logger.info("🏃 Speed mode: Aggressive optimization enabled")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")