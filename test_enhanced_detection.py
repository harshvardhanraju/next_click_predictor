#!/usr/bin/env python3
"""
Test script for the enhanced UI element detection system.
Tests both the AdvancedUIDetector and the updated ScreenshotProcessor.
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from screenshot_processor import ScreenshotProcessor
    from advanced_ui_detector import AdvancedUIDetector
    print("✅ Successfully imported enhanced detection modules")
except ImportError as e:
    print(f"❌ Failed to import detection modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DetectionTester:
    """Test suite for the enhanced UI detection system"""
    
    def __init__(self):
        self.results = []
        
    def create_test_image(self, test_name: str) -> np.ndarray:
        """Create synthetic test images with known UI elements"""
        
        if test_name == "simple_button":
            # Create image with a simple button
            image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
            
            # Draw a blue button
            cv2.rectangle(image, (200, 150), (400, 200), (255, 100, 0), -1)  # Blue filled
            cv2.rectangle(image, (200, 150), (400, 200), (200, 80, 0), 2)   # Darker border
            
            # Add text (simulated)
            cv2.putText(image, "Click Me", (240, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return image
            
        elif test_name == "form_inputs":
            # Create image with form inputs
            image = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Draw input fields
            cv2.rectangle(image, (100, 100), (500, 130), (255, 255, 255), -1)  # White input
            cv2.rectangle(image, (100, 100), (500, 130), (128, 128, 128), 2)   # Gray border
            
            cv2.rectangle(image, (100, 150), (500, 180), (255, 255, 255), -1)  # White input
            cv2.rectangle(image, (100, 150), (500, 180), (128, 128, 128), 2)   # Gray border
            
            # Submit button
            cv2.rectangle(image, (200, 220), (300, 250), (0, 150, 0), -1)  # Green button
            cv2.putText(image, "Submit", (215, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return image
            
        elif test_name == "multiple_elements":
            # Create complex UI with multiple element types
            image = np.ones((500, 700, 3), dtype=np.uint8) * 250  # Light background
            
            # Header area
            cv2.rectangle(image, (0, 0), (700, 50), (50, 50, 150), -1)  # Dark blue header
            
            # Navigation buttons
            cv2.rectangle(image, (50, 70), (150, 100), (100, 200, 100), -1)   # Green
            cv2.rectangle(image, (160, 70), (260, 100), (200, 100, 100), -1)  # Red
            cv2.rectangle(image, (270, 70), (370, 100), (100, 100, 200), -1)  # Blue
            
            # Form area
            cv2.rectangle(image, (50, 130), (400, 160), (255, 255, 255), -1)  # Input field
            cv2.rectangle(image, (50, 130), (400, 160), (128, 128, 128), 1)
            
            # Action buttons
            cv2.rectangle(image, (50, 200), (150, 230), (255, 150, 0), -1)   # Orange
            cv2.rectangle(image, (160, 200), (260, 230), (0, 180, 0), -1)    # Green
            
            # Text elements (simulated links)
            cv2.putText(image, "Learn more", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.line(image, (50, 285), (140, 285), (0, 0, 255), 1)  # Underline
            
            return image
        
        else:
            # Default: simple white image
            return np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    def test_advanced_detector_direct(self):
        """Test the AdvancedUIDetector directly"""
        print("\n🔍 Testing AdvancedUIDetector directly...")
        
        try:
            detector = AdvancedUIDetector()
            test_cases = ["simple_button", "form_inputs", "multiple_elements"]
            
            for test_case in test_cases:
                print(f"\n  Testing: {test_case}")
                image = self.create_test_image(test_case)
                
                start_time = time.time()
                elements = detector.detect_elements(image)
                detection_time = time.time() - start_time
                
                print(f"    ✅ Found {len(elements)} elements in {detection_time:.3f}s")
                
                for i, elem in enumerate(elements[:3]):  # Show first 3 elements
                    print(f"      {i+1}. {elem.element_type} - confidence: {elem.confidence:.2f} - method: {elem.detection_method}")
                
                self.results.append({
                    'test': f'advanced_direct_{test_case}',
                    'elements_found': len(elements),
                    'detection_time': detection_time,
                    'success': True
                })
                
        except Exception as e:
            print(f"    ❌ Advanced detector test failed: {e}")
            self.results.append({
                'test': 'advanced_direct',
                'success': False,
                'error': str(e)
            })
    
    def test_screenshot_processor_enhanced(self):
        """Test the enhanced ScreenshotProcessor"""
        print("\n🔍 Testing enhanced ScreenshotProcessor...")
        
        try:
            # Test with advanced detection enabled
            processor_advanced = ScreenshotProcessor(use_advanced_detector=True)
            print("  ✅ Advanced processor initialized")
            
            # Test with basic detection (fallback)
            processor_basic = ScreenshotProcessor(use_advanced_detector=False)
            print("  ✅ Basic processor initialized")
            
            test_cases = ["simple_button", "form_inputs", "multiple_elements"]
            
            for test_case in test_cases:
                print(f"\n  Testing: {test_case}")
                image = self.create_test_image(test_case)
                
                # Save test image temporarily
                temp_path = f"/tmp/test_{test_case}.png"
                cv2.imwrite(temp_path, image)
                
                # Test advanced processor
                start_time = time.time()
                result_advanced = processor_advanced.process_screenshot(temp_path)
                time_advanced = time.time() - start_time
                
                # Test basic processor
                start_time = time.time()
                result_basic = processor_basic.process_screenshot(temp_path)
                time_basic = time.time() - start_time
                
                print(f"    Advanced: {len(result_advanced['elements'])} elements in {time_advanced:.3f}s")
                print(f"    Basic:    {len(result_basic['elements'])} elements in {time_basic:.3f}s")
                print(f"    Detection method: {result_advanced['processing_metadata']['detection_method']}")
                
                # Compare results
                improvement_ratio = len(result_advanced['elements']) / max(1, len(result_basic['elements']))
                print(f"    Improvement ratio: {improvement_ratio:.2f}x")
                
                self.results.append({
                    'test': f'processor_{test_case}',
                    'advanced_elements': len(result_advanced['elements']),
                    'basic_elements': len(result_basic['elements']),
                    'improvement_ratio': improvement_ratio,
                    'advanced_time': time_advanced,
                    'basic_time': time_basic,
                    'success': True
                })
                
                # Clean up
                os.remove(temp_path)
                
        except Exception as e:
            print(f"    ❌ ScreenshotProcessor test failed: {e}")
            self.results.append({
                'test': 'processor_enhanced',
                'success': False,
                'error': str(e)
            })
    
    def test_detection_accuracy(self):
        """Test detection accuracy on known elements"""
        print("\n🎯 Testing detection accuracy...")
        
        try:
            processor = ScreenshotProcessor(use_advanced_detector=True)
            
            # Create test image with known elements
            image = self.create_test_image("multiple_elements")
            temp_path = "/tmp/test_accuracy.png"
            cv2.imwrite(temp_path, image)
            
            result = processor.process_screenshot(temp_path)
            elements = result['elements']
            
            # Count detected element types
            type_counts = {}
            for elem in elements:
                elem_type = elem['type']
                type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
            
            print(f"    Element type distribution:")
            for elem_type, count in type_counts.items():
                print(f"      {elem_type}: {count}")
            
            # Check if we detected buttons (we created 5 button-like elements)
            buttons_detected = type_counts.get('button', 0)
            forms_detected = type_counts.get('form', 0)
            links_detected = type_counts.get('link', 0)
            
            accuracy_score = 0.0
            if buttons_detected > 0:
                accuracy_score += 0.4  # 40% for detecting buttons
            if forms_detected > 0:
                accuracy_score += 0.3  # 30% for detecting forms
            if links_detected > 0:
                accuracy_score += 0.3  # 30% for detecting links
            
            print(f"    Accuracy score: {accuracy_score:.2f} ({accuracy_score*100:.0f}%)")
            
            self.results.append({
                'test': 'accuracy',
                'buttons_detected': buttons_detected,
                'forms_detected': forms_detected,
                'links_detected': links_detected,
                'accuracy_score': accuracy_score,
                'success': True
            })
            
            os.remove(temp_path)
            
        except Exception as e:
            print(f"    ❌ Accuracy test failed: {e}")
            self.results.append({
                'test': 'accuracy',
                'success': False,
                'error': str(e)
            })
    
    def test_bounding_box_accuracy(self):
        """Test bounding box positioning accuracy"""
        print("\n📐 Testing bounding box accuracy...")
        
        try:
            processor = ScreenshotProcessor(use_advanced_detector=True)
            
            # Create test image with precisely positioned button
            image = np.ones((300, 400, 3), dtype=np.uint8) * 255
            true_bbox = (100, 100, 250, 150)  # Known button position
            cv2.rectangle(image, (true_bbox[0], true_bbox[1]), (true_bbox[2], true_bbox[3]), (0, 150, 255), -1)
            
            temp_path = "/tmp/test_bbox.png"
            cv2.imwrite(temp_path, image)
            
            result = processor.process_screenshot(temp_path)
            elements = result['elements']
            
            if elements:
                # Find the element closest to our known button
                best_match = None
                best_iou = 0.0
                
                for elem in elements:
                    detected_bbox = elem['bbox']  # (x1, y1, x2, y2)
                    iou = self.calculate_iou(true_bbox, detected_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = elem
                
                if best_match:
                    print(f"    Best match IoU: {best_iou:.3f}")
                    print(f"    True bbox:     {true_bbox}")
                    print(f"    Detected bbox: {best_match['bbox']}")
                    
                    # Good bbox accuracy is IoU > 0.5
                    bbox_accuracy = "Good" if best_iou > 0.5 else "Poor"
                    print(f"    Bbox accuracy: {bbox_accuracy}")
                    
                    self.results.append({
                        'test': 'bbox_accuracy',
                        'iou': best_iou,
                        'bbox_accuracy': bbox_accuracy,
                        'success': True
                    })
                else:
                    print("    ❌ No elements detected")
                    self.results.append({
                        'test': 'bbox_accuracy',
                        'success': False,
                        'error': 'No elements detected'
                    })
            else:
                print("    ❌ No elements detected")
                self.results.append({
                    'test': 'bbox_accuracy',
                    'success': False,
                    'error': 'No elements detected'
                })
            
            os.remove(temp_path)
            
        except Exception as e:
            print(f"    ❌ Bounding box test failed: {e}")
            self.results.append({
                'test': 'bbox_accuracy',
                'success': False,
                'error': str(e)
            })
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting Enhanced UI Detection Tests")
        print("=" * 50)
        
        self.test_advanced_detector_direct()
        self.test_screenshot_processor_enhanced()
        self.test_detection_accuracy()
        self.test_bounding_box_accuracy()
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        successful_tests = [r for r in self.results if r.get('success', False)]
        failed_tests = [r for r in self.results if not r.get('success', False)]
        
        print(f"✅ Successful tests: {len(successful_tests)}")
        print(f"❌ Failed tests: {len(failed_tests)}")
        
        if successful_tests:
            print("\n🎯 Key Results:")
            for result in successful_tests:
                test_name = result['test']
                if 'improvement_ratio' in result:
                    print(f"  {test_name}: {result['improvement_ratio']:.2f}x improvement")
                elif 'accuracy_score' in result:
                    print(f"  {test_name}: {result['accuracy_score']*100:.0f}% accuracy")
                elif 'iou' in result:
                    print(f"  {test_name}: {result['iou']:.3f} IoU")
                elif 'elements_found' in result:
                    print(f"  {test_name}: {result['elements_found']} elements detected")
        
        if failed_tests:
            print("\n❌ Failed Tests:")
            for result in failed_tests:
                print(f"  {result['test']}: {result.get('error', 'Unknown error')}")
        
        print("\n🏁 Testing completed!")


def main():
    """Main test function"""
    tester = DetectionTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()