from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
import sys
import tempfile
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ML components disabled for Railway deployment
ML_AVAILABLE = False
print("ML components disabled for Railway lightweight deployment")

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print("=== RAILWAY SERVER STARTUP ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"PORT environment variable: {os.environ.get('PORT', 'NOT_SET')}")
print(f"All environment variables: {dict(os.environ)}")

app = FastAPI(title="Next Click Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "https://next-click-predictor-frontend.vercel.app",
        "https://*.vercel.app",
        "*"  # Allow all for development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor - Force mock mode for Railway deployment
predictor = None
print("⚠️ Running in Railway mock mode - Using lightweight prediction service")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed with error: {str(e)}")
        raise

@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    result = {"status": "working", "port": os.environ.get("PORT", "8000")}
    logger.info(f"Returning: {result}")
    return result

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_attributes: str = Form(...),
    task_description: str = Form(...)
):
    logger.info("Predict endpoint called")
    logger.info(f"File: {file.filename}, Content-Type: {file.content_type}")
    logger.info(f"User attributes: {user_attributes}")
    logger.info(f"Task description: {task_description}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="File must be a PNG, JPG, or JPEG image"
        )
    
    try:
        # Parse user attributes
        user_attrs = json.loads(user_attributes)
        logger.info(f"Parsed user attributes: {user_attrs}")
        
        # Use real ML predictor if available, otherwise use mock
        if predictor is not None:
            logger.info("Using real ML predictor")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            try:
                # Run real prediction
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
                
                logger.info(f"Real ML prediction completed: confidence {result.confidence_score:.2f}")
                return response
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            logger.info("Using mock predictor (ML not available)")
            # Mock prediction result matching frontend types
            mock_element = {
                "x": 300,
                "y": 200,
                "width": 120,
                "height": 40,
                "element_type": "button",
                "text": "Submit",
                "confidence": 0.85,
                "prominence": 0.9
            }
            
            mock_prediction = {
                "element_id": 1,
                "click_probability": 0.85,
                "element": mock_element,
                "reasoning": [
                    "Button positioned prominently in visual hierarchy",
                    "User's experience level suggests familiarity with interface",
                    "Task description indicates specific action needed"
                ]
            }
            
            result = {
                "top_prediction": mock_prediction,
                "all_predictions": [mock_prediction],
                "explanation": {
                    "main_explanation": "Based on visual hierarchy and user attributes, the submit button is most likely to be clicked next.",
                    "key_factors": [
                        {
                            "factor": "Visual prominence",
                            "weight": 0.4,
                            "description": "Button is prominently placed and visually distinct"
                        },
                        {
                            "factor": "User experience level",
                            "weight": 0.3,
                            "description": f"User's {user_attrs.get('tech_savviness', 'intermediate')} tech level"
                        },
                        {
                            "factor": "Task alignment",
                            "weight": 0.3,
                            "description": "Task description suggests need for form submission"
                        }
                    ],
                    "reasoning_chain": [
                        "Analyzed UI elements for clickable components",
                        "Assessed visual hierarchy and prominence",
                        "Considered user attributes and experience level",
                        "Evaluated task description for intent",
                        "Calculated click probabilities using Bayesian network"
                    ],
                    "confidence_analysis": "High confidence due to clear visual hierarchy and task alignment"
                },
                "ui_elements": [mock_element],
                "processing_time": 1.2,
                "confidence_score": 0.85,
                "metadata": {
                    "filename": file.filename,
                    "user_attributes": user_attrs,
                    "task_description": task_description,
                    "model_version": "mock-1.0" if predictor is None else "ml-1.0",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"Returning prediction result: {result}")
            return result
        
    except Exception as e:
        logger.error(f"Predict endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
def health():
    logger.info("Health endpoint called")
    result = {"status": "healthy"}
    logger.info(f"Returning: {result}")
    return result

@app.get("/docs")
def docs_redirect():
    logger.info("Docs endpoint called")
    result = {"message": "API documentation available at /docs"}
    logger.info(f"Returning: {result}")
    return result

# Add startup event for additional logging
@app.on_event("startup")
async def startup_event():
    logger.info("=== FastAPI APPLICATION STARTUP COMPLETE ===")
    logger.info(f"Server should be accessible on port {os.environ.get('PORT', '8000')}")
    
@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("=== FastAPI APPLICATION SHUTDOWN ===")

print("=== SERVER.PY LOADED SUCCESSFULLY ===")
print("FastAPI app created and configured")