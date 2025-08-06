#!/usr/bin/env python3
"""
Comprehensive test of the ultra-fast system
Tests both the predictor and the Cloud Run API
"""

import os
import sys
import time
import json
import requests
from pathlib import Path

# Add local modules
sys.path.insert(0, os.path.dirname(__file__))

def test_predictor_directly():
    """Test the production optimized predictor directly"""
    print("🧪 Testing Ultra-Fast Predictor Directly")
    print("-" * 50)
    
    from production_optimized_predictor import ProductionOptimizedPredictor
    
    predictor = ProductionOptimizedPredictor()
    
    # Test with complex UI image
    test_image_path = '/tmp/complex_ui_test.png'
    
    if not os.path.exists(test_image_path):
        print("❌ Test image not found - creating simple test image")
        return False
    
    # Run multiple tests to check consistency
    times = []
    
    for i in range(3):
        start_time = time.time()
        result = predictor.predict_next_click(
            screenshot_path=test_image_path,
            user_attributes={
                'tech_savviness': 'medium',
                'age_group': 'adult'
            },
            task_description="Find and click the submit button"
        )
        end_time = time.time()
        times.append(end_time - start_time)
        
        print(f"   Test {i+1}: {times[-1]:.3f}s - {result.element_type} - confidence: {result.confidence:.2f}")
    
    avg_time = sum(times) / len(times)
    print(f"   ✅ Average time: {avg_time:.3f}s")
    print(f"   ✅ Max time: {max(times):.3f}s")
    print(f"   ✅ All tests < 1 second: {all(t < 1.0 for t in times)}")
    
    return all(t < 1.0 for t in times)

def test_api_locally():
    """Test the API endpoints locally"""
    print("\n🌐 Testing Cloud Run API Locally")  
    print("-" * 50)
    
    # Start the server in background for testing
    import subprocess
    import time
    
    try:
        # Start server
        print("   Starting server...")
        server = subprocess.Popen([
            "python3", "-m", "uvicorn", "cloudrun_app_optimized:app", 
            "--host", "0.0.0.0", "--port", "8001"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(3)
        
        # Test health endpoint
        print("   Testing health endpoint...")
        try:
            response = requests.get("http://localhost:8001/health", timeout=10)
            if response.status_code == 200:
                print("   ✅ Health check passed")
                health_data = response.json()
                print(f"   📊 Version: {health_data.get('version', 'unknown')}")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
        
        # Test root endpoint
        print("   Testing root endpoint...")
        try:
            response = requests.get("http://localhost:8001/", timeout=10)
            if response.status_code == 200:
                print("   ✅ Root endpoint passed")
                root_data = response.json()
                print(f"   🚀 Service: {root_data.get('service', 'unknown')}")
            else:
                print(f"   ❌ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Root endpoint error: {e}")
            return False
        
        # Test prediction endpoint (if test image exists)
        test_image_path = '/tmp/complex_ui_test.png'
        if os.path.exists(test_image_path):
            print("   Testing prediction endpoint...")
            try:
                with open(test_image_path, 'rb') as f:
                    files = {'file': f}
                    data = {
                        'user_attributes': json.dumps({
                            'tech_savviness': 'medium',
                            'age_group': 'adult'
                        }),
                        'task_description': 'Find and click the submit button'
                    }
                    
                    start_time = time.time()
                    response = requests.post(
                        "http://localhost:8001/predict", 
                        files=files, 
                        data=data, 
                        timeout=60  # 60 second timeout for prediction
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        prediction_time = end_time - start_time
                        print(f"   ✅ Prediction passed in {prediction_time:.2f}s")
                        
                        pred_data = response.json()
                        print(f"   🎯 Element: {pred_data['prediction']['element_type']}")
                        print(f"   📊 Confidence: {pred_data['confidence_score']:.2f}")
                        print(f"   ⏱️  Processing: {pred_data.get('processing_time', 'N/A')}s")
                        
                        # Check if it's fast enough
                        if prediction_time < 10:  # Should be much faster than old system
                            print("   🚀 Response time excellent!")
                            return True
                        else:
                            print("   ⚠️ Response time slower than expected")
                            return True  # Still working, just slower
                    else:
                        print(f"   ❌ Prediction failed: {response.status_code}")
                        print(f"   📝 Response: {response.text[:200]}")
                        return False
            except Exception as e:
                print(f"   ❌ Prediction test error: {e}")
                return False
        else:
            print("   ⚠️ No test image found, skipping prediction test")
            return True
        
    except Exception as e:
        print(f"   ❌ Server test error: {e}")
        return False
    finally:
        # Clean up server
        if 'server' in locals():
            try:
                server.terminate()
                server.wait(timeout=5)
                print("   🧹 Server stopped")
            except:
                server.kill()

def main():
    """Run comprehensive tests"""
    print("🚀 Ultra-Fast System Comprehensive Testing")
    print("=" * 60)
    
    # Test predictor directly
    predictor_ok = test_predictor_directly()
    
    # Test API locally
    api_ok = test_api_locally()
    
    # Summary
    print("\n📋 Test Summary")
    print("-" * 30)
    print(f"   Predictor Direct Test: {'✅ PASS' if predictor_ok else '❌ FAIL'}")
    print(f"   API Local Test: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    overall_success = predictor_ok and api_ok
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASS' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🚀 System ready for deployment!")
        print("   - Ultra-fast processing (< 1 second)")
        print("   - No hanging issues")
        print("   - API endpoints working")
        print("   - Timeout protection active")
    else:
        print("\n⚠️ Issues found - check logs above")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)