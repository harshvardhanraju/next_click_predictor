#!/usr/bin/env python3
"""
Test mobile-specific issues with Cloud Run backend
"""

import requests
import json
import sys
from pathlib import Path

API_URL = "https://next-click-predictor-157954281090.asia-south1.run.app"

def test_cors_preflight():
    """Test CORS preflight request (what mobile browsers do)"""
    print("🔍 Testing CORS Preflight (OPTIONS request)...")
    
    try:
        response = requests.options(
            f"{API_URL}/predict",
            headers={
                'Origin': 'https://your-vercel-app.vercel.app',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'content-type',
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15'
            },
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   CORS Headers:")
        for header, value in response.headers.items():
            if 'access-control' in header.lower() or 'cors' in header.lower():
                print(f"     {header}: {value}")
        
        return response.status_code == 200 or response.status_code == 204
        
    except Exception as e:
        print(f"   ❌ CORS preflight failed: {e}")
        return False

def test_mobile_user_agents():
    """Test with different mobile user agents"""
    print("\n📱 Testing Mobile User Agents...")
    
    user_agents = [
        ("iPhone Safari", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"),
        ("Android Chrome", "Mozilla/5.0 (Linux; Android 14; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"),
        ("iPad Safari", "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")
    ]
    
    success_count = 0
    
    for name, ua in user_agents:
        try:
            response = requests.get(
                f"{API_URL}/health",
                headers={'User-Agent': ua},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"   ✅ {name}: OK")
                success_count += 1
            else:
                print(f"   ❌ {name}: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    return success_count == len(user_agents)

def test_file_upload_mobile():
    """Test file upload with mobile headers"""
    print("\n📤 Testing File Upload (Mobile)...")
    
    # Create small test image if needed
    test_image = Path("test_screenshot.png")
    if not test_image.exists():
        print("   Creating small test image...")
        try:
            # Create minimal test image
            with open(test_image, 'wb') as f:
                # Write minimal PNG header (not a real image, just for testing)
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82')
        except:
            print("   ⚠️ Could not create test image, skipping upload test")
            return False
    
    try:
        with open(test_image, 'rb') as img_file:
            files = {'file': ('test.png', img_file, 'image/png')}
            data = {
                'user_attributes': json.dumps({
                    'age_group': '25-34',
                    'tech_savviness': 'medium',
                    'device_type': 'mobile',
                    'browsing_speed': 'medium'
                }),
                'task_description': 'Test mobile upload'
            }
            
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                data=data,
                headers={
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15',
                    'Origin': 'https://your-frontend.vercel.app'
                },
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Upload successful")
            print(f"   Response size: {len(response.content)} bytes")
            print(f"   Confidence: {result.get('confidence_score', 0):.0%}")
            return True
        else:
            print(f"   ❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Upload timeout (>30s)")
        return False
    except Exception as e:
        print(f"   ❌ Upload error: {e}")
        return False

def test_response_size():
    """Test if response is too large for mobile"""
    print("\n📊 Testing Response Size...")
    
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        size_kb = len(response.content) / 1024
        
        print(f"   Response size: {size_kb:.1f} KB")
        
        if size_kb > 100:
            print("   ⚠️ Response might be too large for slow mobile connections")
            return False
        else:
            print("   ✅ Response size acceptable for mobile")
            return True
            
    except Exception as e:
        print(f"   ❌ Size test failed: {e}")
        return False

def test_https_certificate():
    """Test SSL certificate validity"""
    print("\n🔒 Testing HTTPS Certificate...")
    
    try:
        import ssl
        import socket
        
        context = ssl.create_default_context()
        with socket.create_connection(('next-click-predictor-157954281090.asia-south1.run.app', 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname='next-click-predictor-157954281090.asia-south1.run.app') as ssock:
                cert = ssock.getpeercert()
                print(f"   ✅ Certificate valid")
                print(f"   Subject: {cert.get('subject')}")
                print(f"   Issuer: {cert.get('issuer')}")
                return True
                
    except Exception as e:
        print(f"   ❌ Certificate test failed: {e}")
        return False

def test_dns_resolution():
    """Test DNS resolution"""
    print("\n🌐 Testing DNS Resolution...")
    
    try:
        import socket
        ip = socket.gethostbyname('next-click-predictor-157954281090.asia-south1.run.app')
        print(f"   ✅ DNS resolves to: {ip}")
        return True
    except Exception as e:
        print(f"   ❌ DNS resolution failed: {e}")
        return False

def main():
    print("🔍 Mobile Compatibility Testing")
    print("=" * 40)
    
    tests = [
        ("DNS Resolution", test_dns_resolution),
        ("HTTPS Certificate", test_https_certificate),
        ("CORS Preflight", test_cors_preflight),
        ("Mobile User Agents", test_mobile_user_agents),
        ("Response Size", test_response_size),
        ("File Upload Mobile", test_file_upload_mobile),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print(f"\n{'='*50}")
    print("📋 MOBILE TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All mobile tests passed! Backend should work on mobile.")
    else:
        print(f"\n⚠️ {total-passed} tests failed - mobile issues detected.")
        print("\nCommon mobile issues:")
        print("• CORS preflight requests not handled")
        print("• Large response sizes on slow connections")
        print("• SSL/TLS compatibility issues")
        print("• Timeout on file uploads")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)