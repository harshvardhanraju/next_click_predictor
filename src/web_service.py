"""
FastAPI web service for the Next-Click Prediction System

This provides a REST API interface for the prediction system,
allowing easy integration with web applications.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import tempfile
import os
import json
import logging
from datetime import datetime

from next_click_predictor import NextClickPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Next-Click Prediction API",
    description="Predict what users will click next based on screenshots, user attributes, and tasks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = NextClickPredictor()

# Request/Response models
class UserAttributes(BaseModel):
    age_group: Optional[str] = "25-34"
    tech_savviness: Optional[str] = "medium"
    mood: Optional[str] = "neutral"
    device_type: Optional[str] = "desktop"
    browsing_speed: Optional[str] = "medium"

class PredictionRequest(BaseModel):
    user_attributes: UserAttributes
    task_description: str
    return_detailed: Optional[bool] = True

class PredictionResponse(BaseModel):
    top_prediction: Dict[str, Any]
    all_predictions: List[Dict[str, Any]]
    explanation: Dict[str, Any]
    confidence_score: float
    processing_time: float
    ui_elements_count: int

class AnalysisResponse(BaseModel):
    ui_elements: List[Dict[str, Any]]
    screen_dimensions: List[int]
    total_elements: int
    element_types: Dict[str, int]
    prominence_distribution: Dict[str, float]

class SystemStats(BaseModel):
    total_predictions: int
    avg_processing_time: float
    avg_confidence: float
    uptime: str
    recent_predictions: int

# Global variables for tracking
start_time = datetime.now()
request_count = 0

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Next-Click Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Lightweight health check endpoint for Railway deployment"""
    try:
        # Very simple health check - just return 200 OK
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "next-click-predictor"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/health/ready")
async def readiness_check():
    """More detailed readiness check for full system"""
    try:
        health_data = {
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - start_time),
            "requests_processed": request_count,
            "service": "next-click-predictor",
            "version": "1.0.0"
        }
        
        # Test if predictor is available
        try:
            if predictor is not None:
                health_data["predictor_loaded"] = True
            else:
                health_data["predictor_loaded"] = False
        except Exception:
            health_data["predictor_loaded"] = False
            
        return health_data
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/predict")
async def predict_next_click(
    file: UploadFile = File(...),
    user_attributes: str = Form(...),
    task_description: str = Form(...)
):
    """
    Predict what the user will click next
    
    Args:
        file: Screenshot image file
        user_attributes: JSON string with user attributes
        task_description: Description of the task
        
    Returns:
        Prediction results with explanations
    """
    global request_count
    request_count += 1
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="File must be a PNG, JPG, or JPEG image"
        )
    
    # Parse user attributes
    try:
        user_attrs = json.loads(user_attributes)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format for user_attributes"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        # Run prediction
        result = predictor.predict_next_click(
            tmp_path,
            user_attrs,
            task_description,
            True
        )
        
        # Convert result to response format  
        response = {
            "top_prediction": result.top_prediction,
            "all_predictions": result.all_predictions,
            "explanation": result.explanation,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time,
            "ui_elements": result.ui_elements,
            "metadata": result.metadata
        }
        
        logger.info(f"Prediction completed: confidence {result.confidence_score:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_screenshot(screenshot: UploadFile = File(...)):
    """
    Analyze screenshot without prediction
    
    Args:
        screenshot: PNG screenshot file
        
    Returns:
        Screenshot analysis results
    """
    global request_count
    request_count += 1
    
    # Validate file type
    if not screenshot.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="File must be a PNG, JPG, or JPEG image"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        contents = await screenshot.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        # Run analysis
        analysis = predictor.analyze_screenshot_only(tmp_path)
        
        if 'error' in analysis:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {analysis['error']}"
            )
        
        response = AnalysisResponse(
            ui_elements=analysis['ui_elements'],
            screen_dimensions=analysis['screen_dimensions'],
            total_elements=analysis['total_elements'],
            element_types=analysis['element_types'],
            prominence_distribution=analysis['prominence_distribution']
        )
        
        logger.info(f"Analysis completed: {analysis['total_elements']} elements found")
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = predictor.get_system_stats()
        
        if 'error' in stats:
            return SystemStats(
                total_predictions=0,
                avg_processing_time=0.0,
                avg_confidence=0.0,
                uptime=str(datetime.now() - start_time),
                recent_predictions=0
            )
        
        return SystemStats(
            total_predictions=stats['total_predictions'],
            avg_processing_time=stats['avg_processing_time'],
            avg_confidence=stats['avg_confidence'],
            uptime=str(datetime.now() - start_time),
            recent_predictions=stats['recent_predictions']
        )
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Stats retrieval failed: {str(e)}"
        )

@app.get("/history")
async def get_prediction_history(limit: int = 10):
    """Get recent prediction history"""
    try:
        history = predictor.get_prediction_history(limit)
        
        # Clean up history for JSON response
        clean_history = []
        for item in history:
            clean_item = {
                'timestamp': item['timestamp'],
                'screenshot_path': os.path.basename(item['screenshot_path']),
                'top_prediction': item['top_prediction'],
                'processing_time': item['processing_time'],
                'confidence': item['confidence']
            }
            clean_history.append(clean_item)
        
        return {
            'history': clean_history,
            'total_count': len(history)
        }
        
    except Exception as e:
        logger.error(f"History retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"History retrieval failed: {str(e)}"
        )

@app.post("/batch-predict")
async def batch_predict(
    request: PredictionRequest,
    screenshots: List[UploadFile] = File(...),
    task_descriptions: List[str] = []
):
    """
    Predict clicks for multiple screenshots
    
    Args:
        request: Prediction request with user attributes
        screenshots: List of PNG screenshot files
        task_descriptions: List of task descriptions (one per screenshot)
        
    Returns:
        List of prediction results
    """
    global request_count
    request_count += len(screenshots)
    
    if len(task_descriptions) != len(screenshots):
        raise HTTPException(
            status_code=400,
            detail="Number of task descriptions must match number of screenshots"
        )
    
    tmp_paths = []
    try:
        # Save all uploaded files
        for screenshot in screenshots:
            if not screenshot.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {screenshot.filename} must be a PNG, JPG, or JPEG image"
                )
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                contents = await screenshot.read()
                tmp.write(contents)
                tmp_paths.append(tmp.name)
        
        # Run batch prediction
        results = predictor.predict_batch(
            tmp_paths,
            request.user_attributes.dict(),
            task_descriptions
        )
        
        # Convert results to response format
        response_results = []
        for result in results:
            if result:
                response_results.append({
                    'top_prediction': result.top_prediction,
                    'all_predictions': result.all_predictions,
                    'explanation': result.explanation,
                    'confidence_score': result.confidence_score,
                    'processing_time': result.processing_time,
                    'ui_elements_count': len(result.ui_elements)
                })
            else:
                response_results.append({'error': 'Prediction failed'})
        
        return {
            'results': response_results,
            'total_processed': len(screenshots),
            'successful_predictions': len([r for r in response_results if 'error' not in r])
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

@app.get("/examples")
async def get_examples():
    """Get example requests for API testing"""
    return {
        "predict_example": {
            "user_attributes": {
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused",
                "device_type": "desktop"
            },
            "task_description": "Complete purchase. What would you click next?",
            "return_detailed": True
        },
        "user_profiles": {
            "tech_savvy_user": {
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused",
                "device_type": "desktop"
            },
            "casual_user": {
                "age_group": "35-44",
                "tech_savviness": "medium",
                "mood": "neutral",
                "device_type": "mobile"
            },
            "beginner_user": {
                "age_group": "55-64",
                "tech_savviness": "low",
                "mood": "cautious",
                "device_type": "tablet"
            }
        },
        "task_examples": [
            "Complete purchase. What would you click next?",
            "Find product information. What would you click next?",
            "Sign up for account. What would you click next?",
            "Browse social media. What would you click next?",
            "Generate report. What would you click next?"
        ]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Next-Click Prediction API starting up...")
    logger.info(f"API docs available at: /docs")
    logger.info(f"Health check available at: /health")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Next-Click Prediction API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)