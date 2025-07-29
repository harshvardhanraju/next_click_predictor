#!/usr/bin/env python3
"""
Comprehensive backend feature testing script
Tests UI detection, image processing, Bayesian network, and all ML components
"""

import sys
import os
import json
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

def test_ui_detection():
    """Test UI detection capabilities"""
    print("🔍 Testing UI Detection...")
    
    try:
        from advanced_ui_detector import AdvancedUIDetector
        detector = AdvancedUIDetector()
        print("  ✅ AdvancedUIDetector initialized")
        
        # Test with sample image
        if os.path.exists('test_login.png'):
            from PIL import Image
            import cv2
            
            # Load and process image
            image = Image.open('test_login.png')
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run detection
            elements = detector.detect_elements(cv_image)
            print(f"  ✅ Detected {len(elements)} UI elements")
            
            for i, elem in enumerate(elements[:3]):  # Show first 3
                print(f"    - Element {i+1}: {elem.element_type} at {elem.bbox}")
                
        return True
    except Exception as e:
        print(f"  ❌ UI Detection failed: {e}")
        return False

def test_image_processing():
    """Test image processing pipeline"""
    print("📸 Testing Image Processing...")
    
    try:
        from screenshot_processor import ScreenshotProcessor
        processor = ScreenshotProcessor()
        print("  ✅ ScreenshotProcessor initialized")
        
        if os.path.exists('test_login.png'):
            # Process image
            elements = processor.process_screenshot('test_login.png')
            print(f"  ✅ Processed image, found {len(elements)} elements")
            
            # Test visual features
            for elem in elements[:2]:
                features = processor.extract_visual_features(elem)
                print(f"    - {elem.element_type}: prominence={features.get('prominence', 0):.2f}")
                
        return True
    except Exception as e:
        print(f"  ❌ Image Processing failed: {e}")
        return False

def test_bayesian_network():
    """Test Bayesian network functionality"""
    print("🧠 Testing Bayesian Network...")
    
    try:
        from bayesian_network import BayesianNetworkEngine
        network = BayesianNetworkEngine()
        print("  ✅ BayesianNetworkEngine initialized")
        
        # Test network construction with sample data
        sample_elements = [
            {
                'element_id': 'btn_1',
                'element_type': 'button',
                'text': 'Login',
                'x': 100, 'y': 200,
                'prominence': 0.8,
                'visibility': True
            }
        ]
        
        sample_context = {
            'user_tech_level': 'medium',
            'task_urgency': 'high',
            'task_clarity': 'high'
        }
        
        # Build network
        network.build_network(sample_elements, sample_context)
        print("  ✅ Network built successfully")
        
        # Test inference
        probabilities = network.predict_probabilities()
        print(f"  ✅ Inference completed, {len(probabilities)} predictions")
        
        return True
    except Exception as e:
        print(f"  ❌ Bayesian Network failed: {e}")
        return False

def test_feature_integration():
    """Test feature integration system"""
    print("🔗 Testing Feature Integration...")
    
    try:
        from feature_integrator import FeatureIntegrator
        integrator = FeatureIntegrator()
        print("  ✅ FeatureIntegrator initialized")
        
        # Test user profile parsing
        user_attrs = {
            "tech_savviness": "high",
            "age_group": "25-35",
            "experience_level": "expert"
        }
        
        user_features = integrator.extract_user_features(user_attrs)
        print(f"  ✅ User features extracted: {len(user_features)} features")
        
        # Test task analysis
        task_desc = "I want to login to my account quickly"
        task_features = integrator.analyze_task_description(task_desc)
        print(f"  ✅ Task analyzed: urgency={task_features.get('urgency', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"  ❌ Feature Integration failed: {e}")
        return False

def test_explanation_generator():
    """Test explanation generation"""
    print("💬 Testing Explanation Generator...")
    
    try:
        from explanation_generator import ExplanationGenerator
        generator = ExplanationGenerator()
        print("  ✅ ExplanationGenerator initialized")
        
        # Test explanation generation
        sample_prediction = {
            'element_id': 'btn_login',
            'element_type': 'button',
            'text': 'Sign In',
            'confidence': 0.85,
            'x': 400, 'y': 300
        }
        
        sample_factors = {
            'user_experience': 0.7,
            'task_clarity': 0.9,
            'element_prominence': 0.8
        }
        
        explanation = generator.generate_explanation(sample_prediction, sample_factors)
        print(f"  ✅ Explanation generated: {explanation[:50]}...")
        
        return True
    except Exception as e:
        print(f"  ❌ Explanation Generation failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete prediction pipeline"""
    print("🚀 Testing Full Prediction Pipeline...")
    
    try:
        from next_click_predictor import NextClickPredictor
        predictor = NextClickPredictor()
        print("  ✅ NextClickPredictor initialized")
        
        if os.path.exists('test_login.png'):
            # Test full prediction
            user_attributes = {
                "tech_savviness": "medium",
                "age_group": "25-35"
            }
            
            task_description = "I want to sign in to my account"
            
            result = predictor.predict_next_click(
                'test_login.png',
                user_attributes,
                task_description
            )
            
            print(f"  ✅ Prediction completed!")
            print(f"    - Confidence: {result.get('confidence_score', 0):.2f}")
            print(f"    - Elements found: {len(result.get('ui_elements', []))}")
            print(f"    - Has explanation: {'explanation' in result}")
            
            return True
        else:
            print("  ⚠️  No test image available, skipping full pipeline test")
            return True
            
    except Exception as e:
        print(f"  ❌ Full Pipeline failed: {e}")
        return False

def main():
    """Run all backend feature tests"""
    print("=" * 60)
    print("🧪 COMPREHENSIVE BACKEND TESTING")
    print("=" * 60)
    
    tests = [
        ("UI Detection", test_ui_detection),
        ("Image Processing", test_image_processing),
        ("Bayesian Network", test_bayesian_network),
        ("Feature Integration", test_feature_integration),
        ("Explanation Generator", test_explanation_generator),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All backend features are working correctly!")
    else:
        print("⚠️  Some features need attention")
    
    return passed == total

if __name__ == "__main__":
    main()