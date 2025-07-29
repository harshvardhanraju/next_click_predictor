#!/usr/bin/env python3
"""
Debug False Predictions Issue
Comprehensive debugging to identify why model predicts non-existent UI elements
"""

import sys
import os
import cv2
import numpy as np
import json
import tempfile
import logging
from typing import Dict, Any, List

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_real_ui_test_images():
    """Create realistic UI test images with known elements"""
    test_images = {}
    
    # 1. Simple login form
    login_img = np.ones((600, 800, 3), dtype=np.uint8) * 245  # Light gray background
    
    # Header
    cv2.rectangle(login_img, (0, 0), (800, 80), (50, 50, 150), -1)
    cv2.putText(login_img, 'Login to Account', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Email field
    cv2.rectangle(login_img, (200, 150), (600, 190), (255, 255, 255), -1)
    cv2.rectangle(login_img, (200, 150), (600, 190), (200, 200, 200), 2)
    cv2.putText(login_img, 'Email', (210, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    cv2.putText(login_img, 'user@example.com', (210, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    
    # Password field
    cv2.rectangle(login_img, (200, 220), (600, 260), (255, 255, 255), -1)
    cv2.rectangle(login_img, (200, 220), (600, 260), (200, 200, 200), 2)
    cv2.putText(login_img, 'Password', (210, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    cv2.putText(login_img, '********', (210, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    
    # Sign In button (this is what should be detected)
    cv2.rectangle(login_img, (200, 300), (600, 350), (0, 120, 215), -1)  # Blue button
    cv2.putText(login_img, 'Sign In', (370, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Forgot password link
    cv2.putText(login_img, 'Forgot Password?', (320, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 1)
    
    test_images['login_form'] = {
        'image': login_img,
        'expected_elements': [
            {'type': 'form', 'text': 'user@example.com', 'purpose': 'email input'},
            {'type': 'form', 'text': '********', 'purpose': 'password input'},
            {'type': 'button', 'text': 'Sign In', 'purpose': 'login button'},
            {'type': 'link', 'text': 'Forgot Password?', 'purpose': 'forgot password link'}
        ]
    }
    
    # 2. E-commerce product page
    product_img = np.ones((700, 900, 3), dtype=np.uint8) * 250
    
    # Product title
    cv2.putText(product_img, 'Wireless Headphones - $99.99', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    
    # Add to Cart button (this is what should be detected)
    cv2.rectangle(product_img, (50, 400), (300, 450), (255, 140, 0), -1)  # Orange button
    cv2.putText(product_img, 'Add to Cart', (110, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Buy Now button
    cv2.rectangle(product_img, (320, 400), (570, 450), (220, 50, 50), -1)  # Red button
    cv2.putText(product_img, 'Buy Now', (400, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Quantity selector
    cv2.rectangle(product_img, (50, 350), (150, 390), (255, 255, 255), -1)
    cv2.rectangle(product_img, (50, 350), (150, 390), (200, 200, 200), 2)
    cv2.putText(product_img, 'Qty: 1', (70, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    
    test_images['product_page'] = {
        'image': product_img,
        'expected_elements': [
            {'type': 'button', 'text': 'Add to Cart', 'purpose': 'add to cart'},
            {'type': 'button', 'text': 'Buy Now', 'purpose': 'purchase'},
            {'type': 'form', 'text': 'Qty: 1', 'purpose': 'quantity selector'}
        ]
    }
    
    # 3. Settings page with toggle
    settings_img = np.ones((600, 800, 3), dtype=np.uint8) * 248
    
    # Settings title
    cv2.putText(settings_img, 'Settings', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 2)
    
    # Dark Mode toggle
    cv2.putText(settings_img, 'Dark Mode', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    cv2.rectangle(settings_img, (600, 125), (680, 155), (100, 200, 100), -1)  # Toggle background
    cv2.circle(settings_img, (650, 140), 12, (255, 255, 255), -1)  # Toggle knob
    
    # Save Changes button
    cv2.rectangle(settings_img, (50, 500), (200, 550), (50, 150, 50), -1)  # Green button
    cv2.putText(settings_img, 'Save Changes', (70, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    test_images['settings_page'] = {
        'image': settings_img,
        'expected_elements': [
            {'type': 'toggle', 'text': 'Dark Mode', 'purpose': 'dark mode toggle'},
            {'type': 'button', 'text': 'Save Changes', 'purpose': 'save button'}
        ]
    }
    
    return test_images

def debug_screenshot_processor():
    """Debug the screenshot processor with real UI images"""
    print("🔍 DEBUGGING SCREENSHOT PROCESSOR")
    print("=" * 50)
    
    try:
        from screenshot_processor import ScreenshotProcessor
        
        # Test with advanced detector
        processor_advanced = ScreenshotProcessor(use_advanced_detector=True)
        processor_basic = ScreenshotProcessor(use_advanced_detector=False)
        
        test_images = create_real_ui_test_images()
        
        for test_name, test_data in test_images.items():
            print(f"\n🧪 Testing: {test_name}")
            print("-" * 30)
            
            # Save test image
            temp_path = f"/tmp/debug_{test_name}.png"
            cv2.imwrite(temp_path, test_data['image'])
            
            # Expected elements
            expected = test_data['expected_elements']
            print(f"📋 Expected elements: {len(expected)}")
            for exp in expected:
                print(f"   - {exp['type']}: '{exp['text']}'")
            
            # Test advanced processor
            print(f"\n🔬 Advanced Detection Results:")
            result_advanced = processor_advanced.process_screenshot(temp_path)
            detected_advanced = result_advanced['elements']
            
            print(f"   Found: {len(detected_advanced)} elements")
            for i, elem in enumerate(detected_advanced):
                print(f"   {i+1}. {elem['type']}: '{elem['text']}' (confidence: {elem.get('prominence', 0):.2f})")
            
            # Test basic processor
            print(f"\n🔬 Basic Detection Results:")
            result_basic = processor_basic.process_screenshot(temp_path)
            detected_basic = result_basic['elements']
            
            print(f"   Found: {len(detected_basic)} elements")
            for i, elem in enumerate(detected_basic):
                print(f"   {i+1}. {elem['type']}: '{elem['text']}' (confidence: {elem.get('prominence', 0):.2f})")
            
            # Analysis
            print(f"\n📊 Analysis:")
            expected_texts = [exp['text'] for exp in expected]
            detected_texts_adv = [elem['text'] for elem in detected_advanced]
            detected_texts_basic = [elem['text'] for elem in detected_basic]
            
            print(f"   Expected: {expected_texts}")
            print(f"   Advanced: {detected_texts_adv}")
            print(f"   Basic:    {detected_texts_basic}")
            
            # Check accuracy
            matches_adv = sum(1 for text in expected_texts if any(text in det or det in text for det in detected_texts_adv if det))
            matches_basic = sum(1 for text in expected_texts if any(text in det or det in text for det in detected_texts_basic if det))
            
            accuracy_adv = matches_adv / len(expected_texts) if expected_texts else 0
            accuracy_basic = matches_basic / len(expected_texts) if expected_texts else 0
            
            print(f"   Advanced Accuracy: {accuracy_adv:.1%} ({matches_adv}/{len(expected_texts)})")
            print(f"   Basic Accuracy:    {accuracy_basic:.1%} ({matches_basic}/{len(expected_texts)})")
            
            # Clean up
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ Screenshot processor debug failed: {e}")
        import traceback
        traceback.print_exc()

def debug_next_click_predictor():
    """Debug the full prediction pipeline"""
    print("\n🤖 DEBUGGING NEXT CLICK PREDICTOR")
    print("=" * 50)
    
    try:
        from next_click_predictor import NextClickPredictor
        
        predictor = NextClickPredictor()
        test_images = create_real_ui_test_images()
        
        for test_name, test_data in test_images.items():
            print(f"\n🧪 Testing: {test_name}")
            print("-" * 30)
            
            # Save test image
            temp_path = f"/tmp/debug_pred_{test_name}.png"
            cv2.imwrite(temp_path, test_data['image'])
            
            # Expected elements
            expected = test_data['expected_elements']
            print(f"📋 Expected elements:")
            for exp in expected:
                print(f"   - {exp['type']}: '{exp['text']}'")
            
            # Run prediction
            user_attrs = {
                "tech_savviness": "high",
                "device_type": "desktop", 
                "age_group": "25-34",
                "mood": "focused"
            }
            
            if test_name == 'login_form':
                task_desc = "Log into the account. What would you click next?"
            elif test_name == 'product_page':
                task_desc = "Purchase this product. What would you click next?"
            else:
                task_desc = "Complete the settings configuration. What would you click next?"
            
            print(f"📝 Task: {task_desc}")
            
            result = predictor.predict_next_click(
                screenshot_path=temp_path,
                user_attributes=user_attrs,
                task_description=task_desc
            )
            
            print(f"\n🎯 Prediction Results:")
            print(f"   Top prediction: '{result.top_prediction['element_text']}'")
            print(f"   Type: {result.top_prediction['element_type']}")
            print(f"   Probability: {result.top_prediction['click_probability']:.2%}")
            print(f"   Confidence: {result.confidence_score:.2%}")
            
            print(f"\n📊 All detected elements:")
            for i, elem in enumerate(result.ui_elements):
                print(f"   {i+1}. {elem['type']}: '{elem['text']}' (prominence: {elem.get('prominence', 0):.2f})")
            
            # Check if prediction matches actual UI
            predicted_text = result.top_prediction['element_text']
            actual_texts = [exp['text'] for exp in expected]
            
            is_accurate = any(predicted_text in actual or actual in predicted_text 
                            for actual in actual_texts if actual and predicted_text)
            
            print(f"\n✅ Accuracy Check:")
            print(f"   Predicted: '{predicted_text}'")
            print(f"   Actual options: {actual_texts}")
            print(f"   Match found: {'✅ YES' if is_accurate else '❌ NO - FALSE PREDICTION!'}")
            
            if not is_accurate:
                print(f"   🚨 ERROR: Predicted text '{predicted_text}' not found in actual UI!")
                print(f"   🔧 This indicates a serious bug in the prediction pipeline")
            
            # Clean up
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ Next click predictor debug failed: {e}")
        import traceback
        traceback.print_exc()

def debug_ocr_accuracy():
    """Debug OCR text extraction accuracy"""
    print("\n🔤 DEBUGGING OCR TEXT EXTRACTION")
    print("=" * 50)
    
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        
        test_images = create_real_ui_test_images()
        
        for test_name, test_data in test_images.items():
            print(f"\n🧪 Testing OCR on: {test_name}")
            print("-" * 30)
            
            # Save test image
            temp_path = f"/tmp/debug_ocr_{test_name}.png"
            cv2.imwrite(temp_path, test_data['image'])
            
            # Expected texts
            expected_texts = [exp['text'] for exp in test_data['expected_elements']]
            print(f"📋 Expected texts: {expected_texts}")
            
            # Run OCR
            ocr_results = reader.readtext(temp_path)
            
            print(f"🔤 OCR Results:")
            detected_texts = []
            for (bbox, text, confidence) in ocr_results:
                print(f"   Text: '{text}' (confidence: {confidence:.2f})")
                detected_texts.append(text)
            
            # Check accuracy
            matches = 0
            for expected in expected_texts:
                for detected in detected_texts:
                    if expected.lower() in detected.lower() or detected.lower() in expected.lower():
                        matches += 1
                        break
            
            accuracy = matches / len(expected_texts) if expected_texts else 0
            print(f"📊 OCR Accuracy: {accuracy:.1%} ({matches}/{len(expected_texts)} matches)")
            
            # Clean up
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ OCR debug failed: {e}")

def debug_bayesian_network_issue():
    """Debug why Bayesian network might be generating wrong predictions"""
    print("\n🧠 DEBUGGING BAYESIAN NETWORK")
    print("=" * 50)
    
    try:
        from bayesian_network import BayesianNetworkEngine
        from feature_integrator import FeatureIntegrator
        
        # Create mock UI elements based on real detection
        ui_elements = [
            {
                'id': 'elem_1',
                'type': 'button',
                'text': 'Sign In',  # Real text from UI
                'prominence': 0.8,
                'position_features': {'center_distance': 0.3}
            },
            {
                'id': 'elem_2', 
                'type': 'link',
                'text': 'Forgot Password?',  # Real text from UI
                'prominence': 0.4,
                'position_features': {'center_distance': 0.6}
            }
        ]
        
        user_attrs = {"tech_savviness": 0.8, "mood": 0.7, "experience_level": 0.6}
        task_desc = "Log into the account"
        
        # Test feature integration
        integrator = FeatureIntegrator()
        integrated_features = integrator.integrate_features(user_attrs, ui_elements, task_desc)
        
        print(f"🔗 Feature Integration:")
        print(f"   UI Elements: {len(integrated_features.ui_features)}")
        for i, elem in enumerate(integrated_features.ui_features):
            print(f"   {i+1}. {elem.get('type')}: '{elem.get('text')}' (prominence: {elem.get('prominence')})")
        
        # Test Bayesian network
        bayesian_engine = BayesianNetworkEngine()
        network = bayesian_engine.build_network(integrated_features)
        predictions = bayesian_engine.predict_clicks(integrated_features)
        
        print(f"\n🎯 Bayesian Predictions:")
        for i, pred in enumerate(predictions):
            print(f"   {i+1}. Element: '{pred['element_text']}'")
            print(f"       Type: {pred['element_type']}")
            print(f"       Probability: {pred['click_probability']:.2%}")
            print(f"       Confidence: {pred['confidence']:.2%}")
        
        # Check if predictions match input elements
        input_texts = [elem['text'] for elem in ui_elements]
        predicted_texts = [pred['element_text'] for pred in predictions]
        
        print(f"\n📊 Consistency Check:")
        print(f"   Input texts: {input_texts}")
        print(f"   Predicted texts: {predicted_texts}")
        
        # Check for consistency
        for pred_text in predicted_texts:
            if pred_text not in input_texts:
                print(f"   🚨 INCONSISTENCY: Predicted '{pred_text}' not in input elements!")
        
    except Exception as e:
        print(f"❌ Bayesian network debug failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚨 DEBUGGING FALSE PREDICTIONS ISSUE")
    print("=" * 60)
    print("Investigating why model predicts non-existent UI elements")
    print("=" * 60)
    
    # Debug each component
    debug_screenshot_processor()
    debug_ocr_accuracy() 
    debug_next_click_predictor()
    debug_bayesian_network_issue()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUG SUMMARY")
    print("=" * 60)
    print("Check the results above to identify where the pipeline")
    print("starts generating text that doesn't exist in the UI.")
    print("Common issues:")
    print("- OCR hallucinating text")
    print("- Bayesian network using hardcoded fallback text")
    print("- Feature integrator modifying element text")
    print("- Fallback predictions when ML components fail")

if __name__ == "__main__":
    main()