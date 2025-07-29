#!/usr/bin/env python3
"""
Backend Functionality Test Script
Tests all API endpoints and identifies "failed to fetch" issues
"""

import requests
import json
import time
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO

def create_test_image():
    """Create a test image for prediction testing"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 245
    
    # Add a login button
    cv2.rectangle(img, (200, 150), (400, 190), (0, 120, 215), -1)
    cv2.putText(img, 'Sign In', (270, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add email field
    cv2.rectangle(img, (200, 100), (400, 130), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 100), (400, 130), (200, 200, 200), 2)
    cv2.putText(img, 'Email', (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return img

def test_endpoint(url, method='GET', data=None, files=None, timeout=30):
    """Test an endpoint with proper error handling"""
    try:
        print(f"🧪 Testing {method} {url}")
        
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, data=data, files=files, timeout=timeout)
        
        print(f"   Status: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)[:500]}...")
        else:
            result = response.text
            print(f"   Response: {result[:200]}...")
        
        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'data': result,
            'error': None
        }
        
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection Error: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {'success': False, 'error': error_msg, 'type': 'connection'}
    
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout Error: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {'success': False, 'error': error_msg, 'type': 'timeout'}
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Request Error: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {'success': False, 'error': error_msg, 'type': 'request'}
    
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {'success': False, 'error': error_msg, 'type': 'unexpected'}

def main():
    """Run comprehensive backend tests"""
    print("🚀 BACKEND FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test different backend URLs
    backend_urls = [
        "http://localhost:8080",
        "https://next-click-predictor-157954281090.asia-south1.run.app"
    ]
    
    for backend_url in backend_urls:
        print(f"\n🔗 Testing Backend: {backend_url}")
        print("-" * 50)
        
        test_results = {}
        
        # Test 1: Health Check
        result = test_endpoint(f"{backend_url}/health")
        test_results['health'] = result
        
        # Test 2: Service Info
        result = test_endpoint(f"{backend_url}/")
        test_results['service_info'] = result
        
        # Test 3: Metrics
        result = test_endpoint(f"{backend_url}/metrics")
        test_results['metrics'] = result
        
        # Test 4: Prediction with image
        print(f"\n🖼️ Testing prediction with image...")
        img = create_test_image()
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, img)
            
            try:
                with open(tmp_file.name, 'rb') as f:
                    files = {'file': ('test.png', f, 'image/png')}
                    data = {
                        'user_attributes': json.dumps({
                            'tech_savviness': 'high',
                            'device_type': 'desktop'
                        }),
                        'task_description': 'Login to the account'
                    }
                    
                    result = test_endpoint(f"{backend_url}/predict", 'POST', data, files, timeout=60)
                    test_results['prediction'] = result
            finally:
                os.unlink(tmp_file.name)
        
        # Test 5: Screenshot Analysis
        print(f"\n🔍 Testing screenshot analysis...")
        img = create_test_image()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, img)
            
            try:
                with open(tmp_file.name, 'rb') as f:
                    files = {'file': ('test.png', f, 'image/png')}
                    
                    result = test_endpoint(f"{backend_url}/analyze-screenshot", 'POST', files=files, timeout=60)
                    test_results['analysis'] = result
            finally:
                os.unlink(tmp_file.name)
        
        # Summary for this backend
        print(f"\n📊 Summary for {backend_url}:")
        print("-" * 50)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results.values() if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Error analysis
        errors = {r['error'] for r in test_results.values() if r['error']}
        if errors:
            print(f"\n🔍 Error Analysis:")
            for error in errors:
                print(f"   - {error}")
        
        # Recommendations
        if failed_tests > 0:
            print(f"\n💡 Recommendations:")
            connection_errors = [r for r in test_results.values() if r.get('type') == 'connection']
            if connection_errors:
                print("   - Check if backend service is running")
                print("   - Verify backend URL is correct")
                print("   - Check network connectivity")
            
            timeout_errors = [r for r in test_results.values() if r.get('type') == 'timeout']
            if timeout_errors:
                print("   - Backend may be slow to respond")
                print("   - Consider increasing timeout values")
                print("   - Check backend performance and resources")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()