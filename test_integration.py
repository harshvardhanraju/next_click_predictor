#!/usr/bin/env python3
"""
Integration test script for Next Click Predictor
Tests both backend API and frontend integration
"""

import requests
import json
import sys
import os
import time
from pathlib import Path

def test_backend_api(base_url="http://localhost:8000"):
    """Test backend API endpoints"""
    print(f"🧪 Testing backend API at {base_url}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False
    
    # Test predict endpoint with mock data
    try:
        # Create test image file
        test_image_path = Path("test_screenshot.png")
        if not test_image_path.exists():
            print("⚠️ test_screenshot.png not found, skipping predict test")
            return True
            
        files = {'file': open(test_image_path, 'rb')}
        data = {
            'user_attributes': json.dumps({
                'age_group': '25-34',
                'tech_savviness': 'medium',
                'mood': 'neutral',
                'device_type': 'desktop',
                'browsing_speed': 'medium'
            }),
            'task_description': 'Complete the checkout process'
        }
        
        response = requests.post(f"{base_url}/predict", files=files, data=data)
        print(f"✅ Predict endpoint: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   📊 Confidence: {result.get('confidence_score', 'N/A')}")
            print(f"   🎯 Top prediction: {result.get('top_prediction', {}).get('element_type', 'N/A')}")
            print(f"   ⏱️ Processing time: {result.get('processing_time', 'N/A')}s")
        else:
            print(f"   ❌ Error: {response.text}")
        
        files['file'].close()
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Predict endpoint failed: {e}")
        return False

def test_frontend_build():
    """Test frontend build"""
    print("🏗️ Testing frontend build...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Check if build exists
    build_dir = frontend_dir / ".next"
    if build_dir.exists():
        print("✅ Frontend build directory exists")
        return True
    else:
        print("⚠️ Frontend build directory not found, run 'npm run build' first")
        return False

def test_railway_deployment():
    """Test Railway deployment"""
    print("🚂 Testing Railway deployment...")
    railway_url = "https://nextclickpredictor-production.up.railway.app"
    
    try:
        response = requests.get(railway_url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Railway deployment working: {response.json()}")
            return True
        else:
            print(f"❌ Railway deployment failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Railway deployment error: {e}")
        return False

def test_vercel_deployment():
    """Test Vercel deployment"""
    print("▲ Testing Vercel deployment...")
    vercel_url = "https://next-click-predictor-frontend.vercel.app"
    
    try:
        response = requests.get(vercel_url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Vercel deployment working")
            return True
        else:
            print(f"❌ Vercel deployment failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Vercel deployment error: {e}")
        return False

def main():
    print("🔬 Next Click Predictor Integration Testing")
    print("=" * 50)
    
    # Test local backend
    local_success = test_backend_api("http://localhost:8000")
    
    # Test frontend build
    frontend_success = test_frontend_build()
    
    # Test deployments
    railway_success = test_railway_deployment()
    vercel_success = test_vercel_deployment()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"   Local Backend:  {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"   Frontend Build: {'✅ PASS' if frontend_success else '❌ FAIL'}")
    print(f"   Railway Deploy: {'✅ PASS' if railway_success else '❌ FAIL'}")
    print(f"   Vercel Deploy:  {'✅ PASS' if vercel_success else '❌ FAIL'}")
    
    overall_success = local_success and frontend_success
    print(f"\n🎯 Overall Status: {'✅ READY FOR PRODUCTION' if overall_success else '⚠️ NEEDS ATTENTION'}")
    
    if not railway_success:
        print("\n💡 Recommendation: Railway deployment failing - consider alternative hosting")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)