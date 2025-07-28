#!/usr/bin/env python3
"""
Test frontend integration with Cloud Run backend
Simulates the exact API call that the frontend makes
"""

import requests
import json
import sys
from pathlib import Path

# Your Cloud Run backend URL
API_URL = "https://next-click-predictor-157954281090.asia-south1.run.app"

def test_predict_api():
    """Test the /predict endpoint with realistic frontend data"""
    
    print("🧪 Testing Frontend Integration with Cloud Run Backend")
    print("=" * 60)
    print(f"Backend URL: {API_URL}")
    print()
    
    # Test image file
    test_image_path = Path("test_screenshot.png")
    if not test_image_path.exists():
        print("❌ test_screenshot.png not found!")
        print("Creating a simple test image...")
        
        # Create a simple test image if none exists
        try:
            from PIL import Image, ImageDraw
            # Create a 800x600 test image
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw some UI elements
            draw.rectangle([350, 200, 470, 240], fill='blue', outline='darkblue')  # Button
            draw.rectangle([300, 150, 500, 180], fill='lightgray', outline='gray')  # Input field
            draw.text((360, 210), "Submit", fill='white')
            draw.text((310, 160), "Enter your email...", fill='black')
            
            img.save(test_image_path)
            print("✅ Created test_screenshot.png")
        except ImportError:
            print("⚠️ Pillow not available, using existing image if any")
            return False
    
    # Simulate frontend user input data
    test_data = {
        "user_attributes": {
            "age_group": "25-34",
            "tech_savviness": "medium", 
            "mood": "neutral",
            "device_type": "desktop",
            "browsing_speed": "medium"
        },
        "task_description": "Complete the checkout process and submit my order"
    }
    
    print("📝 Test Data:")
    print(f"   User Profile: {test_data['user_attributes']['age_group']}, "
          f"{test_data['user_attributes']['tech_savviness']} tech level, "
          f"{test_data['user_attributes']['device_type']}")
    print(f"   Task: {test_data['task_description']}")
    print()
    
    try:
        # Prepare the request exactly like the frontend does
        with open(test_image_path, 'rb') as img_file:
            files = {
                'file': (test_image_path.name, img_file, 'image/png')
            }
            
            form_data = {
                'user_attributes': json.dumps(test_data['user_attributes']),
                'task_description': test_data['task_description']
            }
            
            print("🚀 Sending request to Cloud Run backend...")
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                data=form_data,
                timeout=30
            )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ SUCCESS! Backend responded with prediction:")
            print("-" * 50)
            
            # Extract key information like the frontend would
            top_prediction = result.get('top_prediction', {})
            explanation = result.get('explanation', {})
            metadata = result.get('metadata', {})
            
            print(f"🎯 Top Prediction:")
            print(f"   Element Type: {top_prediction.get('element_type', 'Unknown')}")
            print(f"   Element Text: {top_prediction.get('element_text', 'N/A')}")
            print(f"   Click Probability: {top_prediction.get('click_probability', 0):.0%}")
            
            print(f"\n📊 Confidence Metrics:")
            print(f"   Overall Confidence: {result.get('confidence_score', 0):.0%}")
            print(f"   Processing Time: {result.get('processing_time', 0)}s")
            
            print(f"\n🧠 AI Explanation:")
            print(f"   {explanation.get('main_explanation', 'No explanation available')}")
            
            print(f"\n🔍 Key Factors:")
            for factor in explanation.get('key_factors', [])[:3]:
                print(f"   • {factor.get('factor', 'Unknown')}: "
                      f"{factor.get('weight', 0):.0%} weight - "
                      f"{factor.get('description', 'No description')}")
            
            print(f"\n📋 Technical Details:")
            print(f"   UI Elements Detected: {len(result.get('ui_elements', []))}")
            print(f"   Model Version: {metadata.get('model_version', 'Unknown')}")
            print(f"   Platform: {metadata.get('platform', 'Unknown')}")
            
            # Test UI element coordinates (what frontend uses for overlay)
            ui_elements = result.get('ui_elements', [])
            if ui_elements and len(ui_elements) > 0:
                element = ui_elements[0]
                print(f"\n🎨 UI Element Coordinates (for frontend overlay):")
                print(f"   Position: ({element.get('x', 0)}, {element.get('y', 0)})")
                print(f"   Size: {element.get('width', 0)}x{element.get('height', 0)}")
                print(f"   Bounding Box: {element.get('bbox', [])}")
            
            return True
            
        else:
            print(f"\n❌ ERROR: Backend returned {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error Details: {error_data}")
            except:
                print(f"Error Text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - backend may be slow or down")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - check backend URL and internet connection")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_health_check():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Check: {health_data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 Cloud Run Backend Integration Test")
    print("====================================")
    
    # Test health first
    health_ok = test_health_check()
    print()
    
    if not health_ok:
        print("⚠️ Health check failed, but continuing with prediction test...")
        print()
    
    # Test prediction API
    prediction_ok = test_predict_api()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Prediction API: {'✅ PASS' if prediction_ok else '❌ FAIL'}")
    
    if prediction_ok:
        print("\n🎉 SUCCESS! Your Cloud Run backend is working perfectly!")
        print("✅ Frontend integration should work seamlessly")
        print("\n📍 Next Steps:")
        print("   1. Deploy frontend to Vercel with updated environment variable")
        print("   2. Test end-to-end in browser")
        print("   3. Upload real screenshots and verify predictions")
    else:
        print("\n⚠️ Issues found - check backend logs in Cloud Run console")
    
    return prediction_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)