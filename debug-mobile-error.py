#!/usr/bin/env python3
"""
Debug specific mobile error scenarios
"""

import requests
import json
import time

API_URL = "https://next-click-predictor-157954281090.asia-south1.run.app"

def test_mobile_browser_sequence():
    """Simulate exact mobile browser behavior"""
    print("📱 Simulating Mobile Browser Request Sequence")
    print("=" * 50)
    
    session = requests.Session()
    
    # Mobile browser headers
    mobile_headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Step 1: Load main page (what happens when you visit the URL)
        print("1️⃣ Loading main page...")
        response = session.get(API_URL, headers=mobile_headers, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
        
        if response.status_code != 200:
            print(f"   ❌ Main page failed: {response.text[:200]}")
            return False
        
        # Step 2: Check if docs page loads (common mobile test)
        print("\n2️⃣ Loading /docs page...")
        docs_response = session.get(f"{API_URL}/docs", headers=mobile_headers, timeout=10)
        print(f"   Status: {docs_response.status_code}")
        
        # Step 3: Try to access OpenAPI spec (mobile swagger might need this)
        print("\n3️⃣ Loading OpenAPI spec...")
        openapi_response = session.get(f"{API_URL}/openapi.json", headers=mobile_headers, timeout=10)
        print(f"   Status: {openapi_response.status_code}")
        
        if openapi_response.status_code == 200:
            spec = openapi_response.json()
            print(f"   API Title: {spec.get('info', {}).get('title', 'Unknown')}")
        
        return True
        
    except requests.exceptions.Timeout:
        print("   ❌ Request timeout - server too slow for mobile")
        return False
    except requests.exceptions.SSLError as e:
        print(f"   ❌ SSL Error (common on mobile): {e}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_slow_mobile_connection():
    """Test with simulated slow mobile connection"""
    print("\n🐌 Testing Slow Mobile Connection")
    print("=" * 50)
    
    try:
        # Very short timeout to simulate poor mobile connection
        start_time = time.time()
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,  # Short timeout
            headers={
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15',
                'Connection': 'close'  # Don't reuse connections
            }
        )
        
        response_time = time.time() - start_time
        print(f"   Response time: {response_time:.2f}s")
        
        if response_time > 5:
            print("   ⚠️ Slow response - might timeout on mobile")
            return False
        
        if response.status_code == 200:
            print("   ✅ Fast enough for mobile")
            return True
        else:
            print(f"   ❌ Bad status: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Timeout on slow connection")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_large_file_upload():
    """Test if file upload might be too large for mobile"""
    print("\n📁 Testing File Upload Size Limits")
    print("=" * 50)
    
    try:
        # Create different sized test payloads
        test_sizes = [
            ("Small (1KB)", b"x" * 1024),
            ("Medium (100KB)", b"x" * 100 * 1024),
            ("Large (1MB)", b"x" * 1024 * 1024),
        ]
        
        for size_name, data in test_sizes:
            print(f"\n   Testing {size_name}...")
            
            files = {'file': ('test.png', data, 'image/png')}
            form_data = {
                'user_attributes': json.dumps({
                    'device_type': 'mobile',
                    'tech_savviness': 'medium'
                }),
                'task_description': 'Mobile test'
            }
            
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    data=form_data,
                    timeout=15,
                    headers={'User-Agent': 'Mobile Safari'}
                )
                
                if response.status_code == 200:
                    print(f"     ✅ {size_name} upload OK")
                elif response.status_code == 413:
                    print(f"     ❌ {size_name} too large (413)")
                    return False
                else:
                    print(f"     ⚠️ {size_name} failed: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"     ❌ {size_name} timeout")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Upload test error: {e}")
        return False

def test_json_parsing():
    """Test if response JSON can be parsed on mobile"""
    print("\n🔍 Testing JSON Response Parsing")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        print(f"   Content-Type: {content_type}")
        
        if 'application/json' not in content_type:
            print("   ⚠️ Not proper JSON content type")
        
        # Try to parse JSON
        try:
            data = response.json()
            print(f"   ✅ JSON parsing successful")
            print(f"   Keys: {list(data.keys())}")
            return True
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing failed: {e}")
            print(f"   Raw content: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Response test error: {e}")
        return False

def main():
    print("🔍 Mobile Error Debugging")
    print("=" * 40)
    
    tests = [
        ("Mobile Browser Sequence", test_mobile_browser_sequence),
        ("Slow Connection", test_slow_mobile_connection),
        ("File Upload Sizes", test_large_file_upload),
        ("JSON Parsing", test_json_parsing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print(f"\n{'='*60}")
    print("📋 MOBILE DEBUG SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:25} {status}")
    
    failed_tests = [name for name, result in results.items() if not result]
    
    if not failed_tests:
        print("\n🤔 All tests passed - the issue might be:")
        print("   • Specific to your Vercel frontend deployment")
        print("   • Network connectivity on your mobile device")
        print("   • Mobile browser caching issues")
        print("   • CORS issues with your specific Vercel domain")
        print("\n💡 Try:")
        print("   1. Clear mobile browser cache")
        print("   2. Try incognito/private browsing")
        print("   3. Check Vercel environment variables")
        print("   4. Test from different mobile network")
    else:
        print(f"\n⚠️ Found issues: {', '.join(failed_tests)}")
    
    return len(failed_tests) == 0

if __name__ == "__main__":
    main()