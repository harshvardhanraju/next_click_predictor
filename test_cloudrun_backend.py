#!/usr/bin/env python3
"""
Test script to verify Cloud Run backend is working
"""

import requests
import json
import os

# Cloud Run URL (replace with your actual URL)
CLOUD_RUN_URL = "https://next-click-predictor-157954281090.asia-south1.run.app"

def test_backend():
    """Test the backend with a sample image"""
    
    # Test data
    user_attributes = {
        "tech_savviness": "medium",
        "age_group": "25-35",
        "device_type": "desktop"
    }
    
    task_description = "I want to login to my account"
    
    # Test image path
    test_image_path = "test_login.png"
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image {test_image_path} not found")
        return False
    
    print(f"Testing backend at: {CLOUD_RUN_URL}")
    print(f"Using test image: {test_image_path}")
    
    try:
        # Prepare files and data
        with open(test_image_path, 'rb') as image_file:
            files = {
                'file': ('test_login.png', image_file, 'image/png')
            }
            data = {
                'user_attributes': json.dumps(user_attributes),
                'task_description': task_description
            }
            
            # Make request
            print("Sending request...")
            response = requests.post(
                f"{CLOUD_RUN_URL}/predict",
                files=files,
                data=data,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"Prediction: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        print("\nTesting health endpoint...")
        response = requests.get(f"{CLOUD_RUN_URL}/health", timeout=10)
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("=== Cloud Run Backend Test ===")
    
    # Update the URL before running
    if "your-service-url" in CLOUD_RUN_URL:
        print("⚠️  Please update CLOUD_RUN_URL with your actual Cloud Run service URL")
        print("You can find it in the Google Cloud Console or by running:")
        print("gcloud run services list")
        exit(1)
    
    # Test health first
    health_ok = test_health_endpoint()
    
    if health_ok:
        # Test prediction
        prediction_ok = test_backend()
        
        if prediction_ok:
            print("\n✅ All tests passed! Backend is working correctly.")
        else:
            print("\n❌ Prediction test failed.")
    else:
        print("\n❌ Health check failed.")