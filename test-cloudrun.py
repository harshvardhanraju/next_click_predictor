#!/usr/bin/env python3
"""
Test script for Google Cloud Run backend
Tests both local and deployed versions
"""

import requests
import json
import sys
import os
from pathlib import Path

def test_backend(base_url="http://localhost:8080", name="Local"):
    """Test backend endpoints"""
    print(f"🧪 Testing {name} backend at {base_url}")
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root: {data['service']} v{data['version']}")
        else:
            print(f"❌ Root failed: {response.status_code}")
            return False
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data['status']}")
        else:
            print(f"❌ Health failed: {response.status_code}")
            return False
        
        # Test predict endpoint if test image exists
        test_image = Path("test_screenshot.png")
        if test_image.exists():
            files = {'file': open(test_image, 'rb')}
            data = {
                'user_attributes': json.dumps({
                    'age_group': '25-34',
                    'tech_savviness': 'medium',
                    'device_type': 'desktop',
                    'browsing_speed': 'medium'
                }),
                'task_description': 'Complete checkout process'
            }
            
            response = requests.post(f"{base_url}/predict", files=files, data=data, timeout=30)
            files['file'].close()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Predict: {result['confidence_score']:.0%} confidence, "
                      f"{result['processing_time']}s processing")
            else:
                print(f"❌ Predict failed: {response.status_code} - {response.text[:100]}")
                return False
        else:
            print("⚠️ No test image found, skipping predict test")
        
        return True
        
    except Exception as e:
        print(f"❌ {name} backend test failed: {e}")
        return False

def main():
    print("🔬 Google Cloud Run Backend Testing")
    print("=" * 50)
    
    # Test local backend
    local_success = test_backend("http://localhost:8080", "Local")
    
    # Test Cloud Run deployment if URL provided
    cloudrun_url = os.environ.get('CLOUDRUN_URL')
    if cloudrun_url:
        cloudrun_success = test_backend(cloudrun_url, "Cloud Run")
    else:
        print("⚠️ CLOUDRUN_URL not set, skipping Cloud Run test")
        cloudrun_success = None
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"   Local Backend:    {'✅ PASS' if local_success else '❌ FAIL'}")
    if cloudrun_success is not None:
        print(f"   Cloud Run Deploy: {'✅ PASS' if cloudrun_success else '❌ FAIL'}")
    
    print("\n💡 Next Steps:")
    if local_success:
        print("   1. Deploy to Cloud Run: ./deploy-cloudrun.sh")
        print("   2. Update Vercel frontend with Cloud Run URL")
        print("   3. Test end-to-end integration")
    else:
        print("   1. Fix local backend issues first")
    
    return local_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)