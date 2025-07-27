from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

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
def predict():
    return {"status": "post_working", "message": "Predict endpoint working"}

@app.get("/health")
def health():
    return {"status": "healthy"}