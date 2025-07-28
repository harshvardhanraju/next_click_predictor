#!/usr/bin/env python3
"""
Test different user scenarios to verify API flexibility
"""

import requests
import json
from pathlib import Path

API_URL = "https://next-click-predictor-157954281090.asia-south1.run.app"

def test_scenario(scenario_name, user_attrs, task, expected_element=None):
    """Test a specific user scenario"""
    print(f"\n📱 Testing: {scenario_name}")
    print("-" * 40)
    
    test_image = Path("test_screenshot.png")
    if not test_image.exists():
        print("⚠️ No test image, skipping")
        return False
    
    try:
        with open(test_image, 'rb') as img_file:
            files = {'file': (test_image.name, img_file, 'image/png')}
            data = {
                'user_attributes': json.dumps(user_attrs),
                'task_description': task
            }
            
            response = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                top_pred = result.get('top_prediction', {})
                
                print(f"✅ Prediction: {top_pred.get('element_type')} - {top_pred.get('element_text')}")
                print(f"   Confidence: {top_pred.get('click_probability', 0):.0%}")
                print(f"   Reasoning: {', '.join(top_pred.get('reasoning', [])[:2])}")
                
                return True
            else:
                print(f"❌ Failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🎭 Testing Different User Scenarios")
    print("===================================")
    
    scenarios = [
        {
            "name": "Tech-savvy young user shopping",
            "user_attrs": {
                "age_group": "18-24",
                "tech_savviness": "high",
                "device_type": "mobile",
                "browsing_speed": "fast"
            },
            "task": "Buy this product quickly"
        },
        {
            "name": "Senior user browsing cautiously", 
            "user_attrs": {
                "age_group": "55-64",
                "tech_savviness": "low",
                "device_type": "desktop",
                "browsing_speed": "slow"
            },
            "task": "Read more information before deciding"
        },
        {
            "name": "Professional user on tablet",
            "user_attrs": {
                "age_group": "35-44", 
                "tech_savviness": "high",
                "device_type": "tablet",
                "browsing_speed": "medium"
            },
            "task": "Compare prices and features"
        },
        {
            "name": "Student searching for deals",
            "user_attrs": {
                "age_group": "18-24",
                "tech_savviness": "medium", 
                "device_type": "mobile",
                "browsing_speed": "fast"
            },
            "task": "Find the cheapest option available"
        }
    ]
    
    passed = 0
    total = len(scenarios)
    
    for scenario in scenarios:
        success = test_scenario(
            scenario["name"],
            scenario["user_attrs"], 
            scenario["task"]
        )
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} scenarios passed")
    
    if passed == total:
        print("🎉 All scenarios working! Backend handles diverse user profiles correctly.")
    else:
        print("⚠️ Some scenarios failed - check backend logs")
    
    return passed == total

if __name__ == "__main__":
    main()