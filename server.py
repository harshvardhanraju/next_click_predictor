from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "working", "port": os.environ.get("PORT", "8000")}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_attributes: str = Form(...),
    task_description: str = Form(...)
):
    try:
        # Parse user attributes
        user_attrs = json.loads(user_attributes)
        
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
        
        return result
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/docs")
def docs_redirect():
    return {"message": "API documentation available at /docs"}