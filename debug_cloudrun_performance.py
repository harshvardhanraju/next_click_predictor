#!/usr/bin/env python3
"""
Debug Cloud Run performance issues
Check memory usage, cold starts, and response times
"""

import requests
import time
import json
from datetime import datetime

BACKEND_URL = "https://next-click-predictor-157954281090.asia-south1.run.app"

def test_health_multiple_times():
    """Test health endpoint multiple times to check for cold start issues"""
    print("🔍 Testing health endpoint multiple times...")
    
    times = []
    for i in range(5):
        print(f"  Test {i+1}/5...")
        start_time = time.time()
        
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=30)
            end_time = time.time()
            response_time = end_time - start_time
            times.append(response_time)
            
            print(f"    ✅ Response time: {response_time:.2f}s")
            if response.status_code == 200:
                data = response.json()
                print(f"    Memory status: {data.get('memory', 'unknown')}")
            
        except requests.exceptions.Timeout:
            print(f"    ❌ Timeout after 30s")
            times.append(30.0)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            times.append(999)
        
        # Wait between requests
        if i < 4:
            time.sleep(2)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Performance Summary:")
    print(f"    Average: {avg_time:.2f}s")
    print(f"    Min: {min_time:.2f}s") 
    print(f"    Max: {max_time:.2f}s")
    
    if min_time < 2 and max_time > 10:
        print("    🔥 Cold start detected! First request is slow, subsequent requests are fast.")
    elif avg_time > 5:
        print("    ⚠️  Consistently slow responses - likely resource constraint.")
    
    return times

def test_prediction_with_small_image():
    """Test prediction endpoint with minimal payload"""
    print("\n🧪 Testing prediction endpoint with small payload...")
    
    # Create minimal test
    files = {'file': ('test.png', b'fake_png_data', 'image/png')}
    data = {
        'user_attributes': json.dumps({"tech_savviness": "medium"}),
        'task_description': "test task"
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            files=files,
            data=data,
            timeout=60  # Longer timeout for prediction
        )
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"    Response time: {response_time:.2f}s")
        print(f"    Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"    Error response: {response.text[:200]}...")
        
    except requests.exceptions.Timeout:
        print(f"    ❌ Prediction timeout after 60s")
    except Exception as e:
        print(f"    ❌ Prediction error: {e}")

def test_with_real_image():
    """Test with actual test image if available"""
    print("\n📸 Testing with real image...")
    
    import os
    if not os.path.exists('test_login.png'):
        print("    ⚠️  test_login.png not found, skipping real image test")
        return
    
    start_time = time.time()
    try:
        with open('test_login.png', 'rb') as img_file:
            files = {'file': ('test_login.png', img_file, 'image/png')}
            data = {
                'user_attributes': json.dumps({
                    "tech_savviness": "medium",
                    "age_group": "25-35"
                }),
                'task_description': "I want to login"
            }
            
            response = requests.post(
                f"{BACKEND_URL}/predict",
                files=files,
                data=data,
                timeout=120  # Even longer timeout
            )
            
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"    Real image response time: {response_time:.2f}s")
        print(f"    Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"    Processing time: {result.get('processing_time', 'unknown')}s")
            print(f"    ML method: {result.get('ml_method', 'unknown')}")
        else:
            print(f"    Error: {response.text[:200]}...")
            
    except requests.exceptions.Timeout:
        print(f"    ❌ Real image timeout after 120s")
    except Exception as e:
        print(f"    ❌ Real image error: {e}")

def check_google_cloud_run_info():
    """Try to get information about the Cloud Run service"""
    print("\n☁️  Google Cloud Run Service Information:")
    print("    To check memory and CPU usage in Google Cloud Console:")
    print("    1. Go to: https://console.cloud.google.com/run")
    print("    2. Select project: gen-lang-client-0569475118")
    print("    3. Click on 'next-click-predictor' service")
    print("    4. Go to 'Metrics' tab")
    print("    5. Check:")
    print("       - Memory utilization")
    print("       - CPU utilization") 
    print("       - Container instance count")
    print("       - Request latency")
    print("       - Cold starts")
    
    print("\n    Alternative - CLI commands (if gcloud is configured):")
    print("    gcloud run services describe next-click-predictor --region=asia-south1")
    print("    gcloud logging read 'resource.type=cloud_run_revision'")

def main():
    print("=" * 60)
    print("🐛 CLOUD RUN PERFORMANCE DEBUGGING")
    print("=" * 60)
    
    # Test health endpoint multiple times
    health_times = test_health_multiple_times()
    
    # Test prediction endpoints
    test_prediction_with_small_image()
    test_with_real_image()
    
    # Provide Cloud Run monitoring info
    check_google_cloud_run_info()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS")
    print("=" * 60)
    
    avg_health_time = sum(health_times) / len(health_times)
    
    if avg_health_time > 10:
        print("❌ SEVERE PERFORMANCE ISSUE")
        print("   Root cause: Likely insufficient memory or CPU")
        print("   Action needed: Increase Cloud Run resources")
    elif max(health_times) > 10 and min(health_times) < 3:
        print("⚠️  COLD START ISSUE")  
        print("   Root cause: Container cold starts taking too long")
        print("   Action needed: Keep container warm or optimize startup")
    else:
        print("✅ Performance seems acceptable")
    
    print("\n📋 RECOMMENDED ACTIONS:")
    print("1. Check Cloud Run memory/CPU in Google Console")
    print("2. Consider increasing memory to 4Gi and CPU to 2")
    print("3. Check logs for out-of-memory errors")
    print("4. Consider container warm-up strategies")

if __name__ == "__main__":
    main()