"""
Test FastAPI app for Railway with proper /predict endpoint
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from datetime import datetime

app = FastAPI(title="Next-Click Prediction Test API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Next-Click Prediction Test API", 
        "port": os.environ.get("PORT", "not_set"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict_test(
    file: UploadFile = File(...),
    user_attributes: str = Form(...),
    task_description: str = Form(...)
):
    """Test predict endpoint that returns mock data"""
    try:
        # Parse user attributes
        user_attrs = json.loads(user_attributes)
        
        # Mock response that matches frontend expectations
        mock_result = {
            "top_prediction": {
                "element_id": 1,
                "click_probability": 0.85,
                "element": {
                    "x": 100,
                    "y": 200,
                    "width": 120,
                    "height": 40,
                    "element_type": "BUTTON",
                    "text": "Submit",
                    "confidence": 0.9,
                    "prominence": 0.8
                },
                "reasoning": ["High visual prominence", "Clear call-to-action", "Central position"]
            },
            "all_predictions": [
                {
                    "element_id": 1,
                    "click_probability": 0.85,
                    "element": {
                        "x": 100,
                        "y": 200,
                        "width": 120,
                        "height": 40,
                        "element_type": "BUTTON",
                        "text": "Submit",
                        "confidence": 0.9,
                        "prominence": 0.8
                    },
                    "reasoning": ["High visual prominence", "Clear call-to-action"]
                }
            ],
            "explanation": {
                "main_explanation": f"Based on the task '{task_description}' and user profile, the Submit button has the highest click probability.",
                "key_factors": [
                    {
                        "factor": "Visual Prominence",
                        "weight": 0.4,
                        "description": "Button has high contrast and central position"
                    },
                    {
                        "factor": "Task Relevance", 
                        "weight": 0.35,
                        "description": "Submit action aligns with user's goal"
                    }
                ],
                "reasoning_chain": ["User uploaded screenshot", "Analyzed UI elements", "Matched with task goal"],
                "confidence_analysis": "High confidence prediction due to clear visual cues"
            },
            "ui_elements": [
                {
                    "x": 100,
                    "y": 200,
                    "width": 120,
                    "height": 40,
                    "element_type": "BUTTON",
                    "text": "Submit",
                    "confidence": 0.9,
                    "prominence": 0.8
                }
            ],
            "processing_time": 1.2,
            "confidence_score": 0.85,
            "metadata": {
                "model_version": "test",
                "timestamp": datetime.now().isoformat(),
                "user_age_group": user_attrs.get("age_group", "unknown"),
                "device_type": user_attrs.get("device_type", "unknown"),
                "file_size": file.size if file else 0,
                "file_type": file.content_type if file else "unknown"
            }
        }
        
        return mock_result
        
    except Exception as e:
        return {
            "error": f"Test endpoint error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "received_data": {
                "file_info": f"{file.filename} ({file.content_type})" if file else "no file",
                "task": task_description,
                "user_attrs": user_attributes
            }
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)