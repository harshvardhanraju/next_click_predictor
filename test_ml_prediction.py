#!/usr/bin/env python3
"""
Test script to verify the ML prediction functionality works
"""

import sys
import os
import cv2
import numpy as np
import json
import tempfile

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_ml_components():
    """Test individual ML components"""
    try:
        from screenshot_processor import ScreenshotProcessor
        print("✅ ScreenshotProcessor imported successfully")
        
        # Create test image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (200, 150), (400, 200), (0, 150, 255), -1)  # Orange button
        cv2.putText(img, 'Submit', (260, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, img)
            temp_path = temp_file.name
        
        try:
            # Test screenshot processor
            processor = ScreenshotProcessor(use_advanced_detector=True)
            print("✅ ScreenshotProcessor initialized")
            
            result = processor.process_screenshot(temp_path)
            print(f"✅ Found {result['total_elements']} elements")
            print(f"📊 Detection method: {result['processing_metadata']['detection_method']}")
            
            # Show element details
            for i, elem in enumerate(result['elements'][:3]):
                print(f"  Element {i+1}: {elem['type']} - bbox: {elem['bbox']}")
            
            return True
            
        finally:
            os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ ML component test failed: {e}")
        return False

def test_next_click_predictor():
    """Test the full prediction pipeline"""
    try:
        from next_click_predictor import NextClickPredictor
        print("\n🤖 Testing NextClickPredictor...")
        
        # Create test image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (200, 150), (400, 200), (0, 150, 255), -1)  # Orange button
        cv2.putText(img, 'Submit', (260, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, img)
            temp_path = temp_file.name
        
        try:
            # Initialize predictor
            predictor = NextClickPredictor()
            print("✅ NextClickPredictor initialized")
            
            # Test user attributes
            user_attrs = {
                "tech_savviness": "high",
                "device_type": "desktop",
                "age_group": "25-34",
                "mood": "focused"
            }
            
            task_desc = "Complete form submission. What would you click next?"
            
            # Run prediction
            result = predictor.predict_next_click(
                screenshot_path=temp_path,
                user_attributes=user_attrs,
                task_description=task_desc,
                return_detailed=True
            )
            
            print("✅ Prediction completed")
            print(f"📊 Top prediction: {result.top_prediction['element_text']} ({result.top_prediction['click_probability']:.2%})")
            print(f"🎯 Confidence: {result.confidence_score:.2%}")
            print(f"⏱️ Processing time: {result.processing_time:.2f}s")
            print(f"🔍 UI elements found: {len(result.ui_elements)}")
            
            return True
            
        finally:
            os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ NextClickPredictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing ML Prediction Components")
    print("=" * 50)
    
    success = True
    
    # Test individual components
    if not test_ml_components():
        success = False
    
    # Test full prediction pipeline
    if not test_next_click_predictor():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All ML tests passed!")
    else:
        print("❌ Some tests failed")
    
    return success

if __name__ == "__main__":
    main()