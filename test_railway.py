#!/usr/bin/env python3

import sys
import os
import time

print("=== RAILWAY TEST SCRIPT STARTING ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"PORT: {os.environ.get('PORT', 'NOT_SET')}")

# List files in current directory
print("\nFiles in current directory:")
for file in os.listdir('.'):
    print(f"  {file}")

print("\nAttempting to import FastAPI...")
try:
    import fastapi
    print(f"✅ FastAPI imported successfully: {fastapi.__version__}")
except Exception as e:
    print(f"❌ FastAPI import failed: {e}")

print("\nAttempting to import uvicorn...")
try:
    import uvicorn
    print(f"✅ Uvicorn imported successfully: {uvicorn.__version__}")
except Exception as e:
    print(f"❌ Uvicorn import failed: {e}")

print("\n=== KEEPING SCRIPT ALIVE FOR 60 SECONDS ===")
time.sleep(60)
print("=== RAILWAY TEST SCRIPT ENDING ===")# Force Railway redeploy
