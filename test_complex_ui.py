#!/usr/bin/env python3
"""
Test script for complex UI with 300+ visual elements
This will help identify performance bottlenecks and timeout issues
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_complex_ui_image():
    """Create a complex UI image with 300+ visual elements"""
    print("🎨 Creating complex UI image with 300+ elements...")
    
    # Create a large canvas (1920x1080 - typical screen size)
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 245  # Light gray background
    
    element_count = 0
    
    # Header section with navigation (20 elements)
    cv2.rectangle(img, (0, 0), (1920, 80), (50, 50, 50), -1)  # Dark header
    for i in range(10):
        x = 100 + i * 150
        cv2.rectangle(img, (x, 20), (x + 120, 60), (100, 100, 100), -1)  # Nav buttons
        cv2.rectangle(img, (x + 5, 25), (x + 115, 55), (255, 255, 255), 1)  # Button borders
        element_count += 2
    
    # Sidebar with menu items (50 elements)
    cv2.rectangle(img, (0, 80), (250, 1080), (220, 220, 220), -1)  # Sidebar background
    for i in range(25):
        y = 100 + i * 35
        cv2.rectangle(img, (10, y), (240, y + 30), (200, 200, 200), -1)  # Menu items
        cv2.rectangle(img, (10, y), (240, y + 30), (150, 150, 150), 1)  # Menu borders
        element_count += 2
    
    # Main content area with cards/buttons (150+ elements)
    start_x, start_y = 270, 100
    card_width, card_height = 180, 120
    spacing = 20
    
    rows = 15
    cols = 8
    
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * (card_width + spacing)
            y = start_y + row * (card_height + spacing)
            
            if x + card_width > 1920 or y + card_height > 1080:
                continue
                
            # Card background
            cv2.rectangle(img, (x, y), (x + card_width, y + card_height), (255, 255, 255), -1)
            cv2.rectangle(img, (x, y), (x + card_width, y + card_height), (200, 200, 200), 2)
            element_count += 2
            
            # Card header
            cv2.rectangle(img, (x + 10, y + 10), (x + card_width - 10, y + 35), (100, 150, 200), -1)
            element_count += 1
            
            # Card buttons
            btn_y = y + card_height - 35
            cv2.rectangle(img, (x + 10, btn_y), (x + 80, btn_y + 25), (50, 150, 50), -1)  # Green button
            cv2.rectangle(img, (x + 90, btn_y), (x + 160, btn_y + 25), (200, 50, 50), -1)  # Red button
            element_count += 2
            
            # Card image placeholder
            cv2.rectangle(img, (x + 20, y + 45), (x + card_width - 20, y + 85), (180, 180, 180), -1)
            element_count += 1
    
    # Footer with many small elements (50+ elements)
    footer_y = 1000
    cv2.rectangle(img, (0, footer_y), (1920, 1080), (60, 60, 60), -1)  # Footer background
    
    # Footer links/buttons
    for i in range(25):
        x = 50 + i * 70
        if x + 60 > 1920:
            break
        cv2.rectangle(img, (x, footer_y + 20), (x + 60, footer_y + 45), (100, 100, 100), -1)
        cv2.rectangle(img, (x, footer_y + 20), (x + 60, footer_y + 45), (150, 150, 150), 1)
        element_count += 2
    
    # Add some form elements (30+ elements)
    form_x, form_y = 270, 950
    for i in range(6):
        x = form_x + i * 200
        if x + 180 > 1920:
            break
        # Input field
        cv2.rectangle(img, (x, form_y), (x + 180, form_y + 35), (255, 255, 255), -1)
        cv2.rectangle(img, (x, form_y), (x + 180, form_y + 35), (100, 100, 100), 2)
        # Label above
        cv2.rectangle(img, (x, form_y - 20), (x + 100, form_y - 5), (240, 240, 240), -1)
        element_count += 3
    
    # Add some overlapping elements to test complex detection
    overlay_elements = 20
    for i in range(overlay_elements):
        x = np.random.randint(300, 1600)
        y = np.random.randint(200, 800)
        w = np.random.randint(40, 120)
        h = np.random.randint(25, 60)
        
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
        element_count += 2
    
    print(f"✅ Created complex UI with approximately {element_count} visual elements")
    return img, element_count

def test_performance_with_timeouts():
    """Test the system with various timeout scenarios"""
    print("\n🚀 Starting performance test with complex UI...")
    
    # Create test image
    complex_img, expected_elements = create_complex_ui_image()
    test_image_path = '/tmp/complex_ui_test.png'
    cv2.imwrite(test_image_path, complex_img)
    print(f"💾 Saved test image to: {test_image_path}")
    
    try:
        from improved_next_click_predictor import ImprovedNextClickPredictor
        
        # Test with different configurations
        test_configs = [
            {
                'name': 'Default Config',
                'config': {
                    'log_level': 'INFO',
                    'enable_evaluation': False,
                    'ensemble_config': {'ensemble_method': 'adaptive'}
                },
                'timeout': 60  # 1 minute
            },
            {
                'name': 'Fast Config (Limited Elements)',
                'config': {
                    'log_level': 'WARNING',
                    'enable_evaluation': False,
                    'max_elements_to_process': 50,  # Limit processing
                    'ensemble_config': {'ensemble_method': 'adaptive'}
                },
                'timeout': 30
            },
            {
                'name': 'Ultra Fast Config',
                'config': {
                    'log_level': 'WARNING', 
                    'enable_evaluation': False,
                    'max_elements_to_process': 20,
                    'skip_ocr': True,  # Skip OCR for speed
                    'ensemble_config': {'ensemble_method': 'weighted_average'}
                },
                'timeout': 15
            }
        ]
        
        results = []
        
        for test_config in test_configs:
            print(f"\n📊 Testing: {test_config['name']}")
            print(f"   Timeout: {test_config['timeout']} seconds")
            
            start_time = time.time()
            
            try:
                # Create predictor with test config
                predictor = ImprovedNextClickPredictor(test_config['config'])
                init_success = predictor.initialize()
                
                if not init_success:
                    print("   ⚠️ ML initialization failed, using fallback mode")
                
                # Run prediction with timeout simulation
                print("   🔄 Running prediction...")
                prediction_start = time.time()
                
                result = predictor.predict_next_click(
                    screenshot_path=test_image_path,
                    user_attributes={
                        'tech_savviness': 'medium',
                        'age_group': 'adult',
                        'device_type': 'desktop'
                    },
                    task_description="Find and click the submit button",
                    return_detailed=True
                )
                
                prediction_time = time.time() - prediction_start
                total_time = time.time() - start_time
                
                # Analyze results
                ui_elements_detected = getattr(result, 'ui_elements_detected', 0)
                processing_time = getattr(result, 'processing_time', prediction_time)
                
                test_result = {
                    'config_name': test_config['name'],
                    'total_time': total_time,
                    'prediction_time': prediction_time,
                    'processing_time': processing_time,
                    'ui_elements_detected': ui_elements_detected,
                    'expected_elements': expected_elements,
                    'success': True,
                    'timeout_exceeded': total_time > test_config['timeout'],
                    'top_prediction': result.top_prediction if hasattr(result, 'top_prediction') else None
                }
                
                results.append(test_result)
                
                print(f"   ✅ Success!")
                print(f"   ⏱️  Total time: {total_time:.2f}s")
                print(f"   🔍 Elements detected: {ui_elements_detected}")
                print(f"   ⚡ Processing time: {processing_time:.2f}s")
                
                if test_result['timeout_exceeded']:
                    print(f"   ⚠️  TIMEOUT EXCEEDED (limit: {test_config['timeout']}s)")
                
            except Exception as e:
                error_time = time.time() - start_time
                print(f"   ❌ Error after {error_time:.2f}s: {str(e)}")
                
                results.append({
                    'config_name': test_config['name'],
                    'total_time': error_time,
                    'success': False,
                    'error': str(e),
                    'timeout_exceeded': error_time > test_config['timeout']
                })
                
                # Print short traceback for debugging
                print("   📋 Error details:")
                traceback.print_exc()
        
        # Summary report
        print("\n" + "="*60)
        print("📊 PERFORMANCE TEST SUMMARY")
        print("="*60)
        
        for result in results:
            print(f"\n🔹 {result['config_name']}")
            if result['success']:
                print(f"   Status: ✅ Success")
                print(f"   Time: {result['total_time']:.2f}s")
                print(f"   Elements: {result.get('ui_elements_detected', 'N/A')}")
                print(f"   Timeout: {'❌ Yes' if result['timeout_exceeded'] else '✅ No'}")
            else:
                print(f"   Status: ❌ Failed")
                print(f"   Time: {result['total_time']:.2f}s")
                print(f"   Error: {result.get('error', 'Unknown')}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        successful_fast = [r for r in results if r['success'] and not r['timeout_exceeded']]
        
        if successful_fast:
            fastest = min(successful_fast, key=lambda x: x['total_time'])
            print(f"   🚀 Fastest successful config: {fastest['config_name']}")
            print(f"      Time: {fastest['total_time']:.2f}s")
            print(f"      Elements detected: {fastest.get('ui_elements_detected', 'N/A')}")
        else:
            print("   ⚠️  All configs exceeded timeout or failed")
            print("   🔧 Consider:")
            print("      - Reducing max_elements_to_process further")
            print("      - Implementing element pre-filtering")
            print("      - Adding early termination conditions")
            print("      - Using simpler detection methods only")
        
        return results
        
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return None
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🧪 Complex UI Performance Test")
    print("Testing system with 300+ visual elements")
    print("-" * 50)
    
    results = test_performance_with_timeouts()
    
    if results:
        print(f"\n✅ Test completed. Check results above.")
        
        # Suggest optimizations based on results
        print(f"\n🔧 OPTIMIZATION SUGGESTIONS:")
        print(f"   1. Add max_elements_to_process limit (20-50)")
        print(f"   2. Implement element pre-filtering by size/position")
        print(f"   3. Add processing timeout with partial results")
        print(f"   4. Use faster detection methods for complex UIs")
        print(f"   5. Consider element sampling instead of processing all")
    else:
        print(f"\n❌ Test failed to complete")

if __name__ == "__main__":
    main()