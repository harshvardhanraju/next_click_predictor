from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
import sys

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    try:
        # Parse user attributes
        user_attrs = json.loads(user_attributes)
        logger.info(f"Parsed user attributes: {user_attrs}")
        
        # Mock prediction result for now
        result = {
            "predictions": [
                {
                    "x": 300,
                    "y": 200,
                    "confidence": 0.85,
                    "element_type": "button",
                    "description": "Primary action button"
                }
            ],
            "explanation": {
                "key_factors": [
                    "Button positioned in visual hierarchy",
                    "User's experience level suggests familiarity",
                    "Task description indicates specific goal"
                ],
                "confidence_score": 0.85
            },
            "user_attributes": user_attrs,
            "task_description": task_description,
            "filename": file.filename
        }
        
        logger.info(f"Returning prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Predict endpoint error: {str(e)}")
        error_result = {"error": str(e), "status": "failed"}
        logger.info(f"Returning error: {error_result}")
        return error_result

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