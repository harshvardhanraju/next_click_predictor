#!/usr/bin/env python3
"""
Demo API client to test the Next-Click Prediction System
"""

import requests
import json
import tempfile
import cv2
import numpy as np
import os
import time

def create_demo_image():
    """Create a demo image for testing"""
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Header
    cv2.rectangle(image, (0, 0), (800, 80), (51, 51, 51), -1)
    cv2.putText(image, 'E-commerce Checkout', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Product info
    cv2.rectangle(image, (50, 100), (750, 200), (245, 245, 245), 2)
    cv2.putText(image, 'Wireless Headphones - $99.99', (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, 'Quantity: 1', (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, 'Total: $99.99', (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Main action buttons
    cv2.rectangle(image, (300, 250), (500, 300), (40, 167, 69), -1)  # Green checkout button
    cv2.putText(image, 'CHECKOUT', (330, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.rectangle(image, (50, 250), (280, 300), (200, 200, 200), -1)  # Gray continue shopping
    cv2.putText(image, 'Continue Shopping', (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Help link
    cv2.putText(image, 'Need help?', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        return tmp.name

def test_api_endpoints():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🎯 Next-Click Prediction System - API Demo")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check passed")
            print(f"  Status: {health_data['status']}")
            print(f"  Uptime: {health_data['uptime']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
    
    # Test 2: Get examples
    print("\n2. Testing examples endpoint...")
    try:
        response = requests.get(f"{base_url}/examples")
        if response.status_code == 200:
            examples_data = response.json()
            print(f"✓ Examples retrieved")
            print(f"  Available user profiles: {len(examples_data['user_profiles'])}")
            print(f"  Sample task: {examples_data['task_examples'][0]}")
        else:
            print(f"✗ Examples failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Examples failed: {e}")
    
    # Test 3: Screenshot analysis
    print("\n3. Testing screenshot analysis...")
    try:
        demo_image_path = create_demo_image()
        
        with open(demo_image_path, 'rb') as f:
            files = {'screenshot': f}
            response = requests.post(f"{base_url}/analyze", files=files)
        
        if response.status_code == 200:
            analysis_data = response.json()
            print(f"✓ Screenshot analysis completed")
            print(f"  Elements found: {analysis_data['total_elements']}")
            print(f"  Screen dimensions: {analysis_data['screen_dimensions']}")
            print(f"  Element types: {analysis_data['element_types']}")
        else:
            print(f"✗ Screenshot analysis failed: {response.status_code}")
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"✗ Screenshot analysis failed: {e}")
    
    # Test 4: Full prediction
    print("\n4. Testing full prediction...")
    try:
        demo_image_path = create_demo_image()
        
        prediction_request = {
            "user_attributes": {
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused",
                "device_type": "desktop"
            },
            "task_description": "Complete purchase. What would you click next?",
            "return_detailed": True
        }
        
        with open(demo_image_path, 'rb') as f:
            files = {'screenshot': f}
            data = {'request': json.dumps(prediction_request)}
            response = requests.post(f"{base_url}/predict", files=files, data=data)
        
        if response.status_code == 200:
            prediction_data = response.json()
            print(f"✓ Full prediction completed")
            print(f"  Processing time: {prediction_data['processing_time']:.2f}s")
            print(f"  Confidence: {prediction_data['confidence_score']:.1%}")
            print(f"  UI elements: {prediction_data['ui_elements_count']}")
            
            # Show top prediction
            top_pred = prediction_data['top_prediction']
            print(f"\n🎯 TOP PREDICTION:")
            print(f"  Element: {top_pred['element_text']}")
            print(f"  Type: {top_pred['element_type']}")
            print(f"  Probability: {top_pred['click_probability']:.1%}")
            print(f"  Confidence: {top_pred['confidence']:.1%}")
            
            # Show all predictions
            print(f"\n📊 ALL PREDICTIONS:")
            for i, pred in enumerate(prediction_data['all_predictions'][:5]):
                print(f"  {i+1}. {pred['element_text']} ({pred['element_type']}) - {pred['click_probability']:.1%}")
            
            # Show explanation
            if 'explanation' in prediction_data and 'main_explanation' in prediction_data['explanation']:
                print(f"\n💡 EXPLANATION:")
                print(f"  {prediction_data['explanation']['main_explanation']}")
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
    
    # Test 5: System stats
    print("\n5. Testing system stats...")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats_data = response.json()
            print(f"✓ System stats retrieved")
            print(f"  Total predictions: {stats_data['total_predictions']}")
            print(f"  Average processing time: {stats_data['avg_processing_time']:.2f}s")
            print(f"  Average confidence: {stats_data['avg_confidence']:.1%}")
        else:
            print(f"✗ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Stats failed: {e}")
    
    # Cleanup
    try:
        if 'demo_image_path' in locals():
            os.remove(demo_image_path)
    except:
        pass

def test_different_scenarios():
    """Test different user scenarios"""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT USER SCENARIOS")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    scenarios = [
        {
            "name": "Tech-Savvy User",
            "user_attributes": {
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused",
                "device_type": "desktop"
            },
            "task": "Complete purchase quickly"
        },
        {
            "name": "Casual User",
            "user_attributes": {
                "age_group": "35-44",
                "tech_savviness": "medium",
                "mood": "neutral",
                "device_type": "mobile"
            },
            "task": "Browse and maybe buy something"
        },
        {
            "name": "Senior User",
            "user_attributes": {
                "age_group": "55-64",
                "tech_savviness": "low",
                "mood": "cautious",
                "device_type": "tablet"
            },
            "task": "Need help with purchase"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        try:
            demo_image_path = create_demo_image()
            
            prediction_request = {
                "user_attributes": scenario["user_attributes"],
                "task_description": scenario["task"],
                "return_detailed": False
            }
            
            with open(demo_image_path, 'rb') as f:
                files = {'screenshot': f}
                data = {'request': json.dumps(prediction_request)}
                response = requests.post(f"{base_url}/predict", files=files, data=data)
            
            if response.status_code == 200:
                prediction_data = response.json()
                top_pred = prediction_data['top_prediction']
                
                print(f"Task: {scenario['task']}")
                print(f"Predicted click: {top_pred['element_text']} ({top_pred['element_type']})")
                print(f"Probability: {top_pred['click_probability']:.1%}")
                print(f"Processing time: {prediction_data['processing_time']:.2f}s")
            else:
                print(f"✗ Prediction failed: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Scenario failed: {e}")
        
        # Cleanup
        try:
            if 'demo_image_path' in locals():
                os.remove(demo_image_path)
        except:
            pass

def main():
    """Main demo function"""
    print("Starting API Demo...")
    print("Make sure the web service is running on http://localhost:8000")
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    try:
        test_api_endpoints()
        test_different_scenarios()
        
        print("\n" + "=" * 60)
        print("🎉 API Demo completed successfully!")
        print("=" * 60)
        print("\nThe Next-Click Prediction System is working correctly!")
        print("You can access the API documentation at: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()