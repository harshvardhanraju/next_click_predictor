import unittest
import tempfile
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import logging
from unittest.mock import patch, MagicMock
import sys
import traceback

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from screenshot_processor import ScreenshotProcessor, UIElement


class TestScreenshotProcessorDetailed(unittest.TestCase):
    """Comprehensive test suite for ScreenshotProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = ScreenshotProcessor()
        self.test_images = {}
        self.test_results = []
        self.errors = []
        
        # Create various test images
        self.test_images = {
            'simple_button': self._create_simple_button_image(),
            'complex_ui': self._create_complex_ui_image(),
            'text_heavy': self._create_text_heavy_image(),
            'minimal_ui': self._create_minimal_ui_image(),
            'empty_image': self._create_empty_image(),
            'high_contrast': self._create_high_contrast_image(),
            'low_contrast': self._create_low_contrast_image(),
            'overlapping_elements': self._create_overlapping_elements(),
            'small_elements': self._create_small_elements(),
            'large_elements': self._create_large_elements()
        }
        
        logger.info(f"Created {len(self.test_images)} test images")
    
    def tearDown(self):
        """Clean up test fixtures"""
        for image_path in self.test_images.values():
            if os.path.exists(image_path):
                os.remove(image_path)
        
        # Log test results
        logger.info(f"Test completed with {len(self.test_results)} results and {len(self.errors)} errors")
    
    def _create_simple_button_image(self) -> str:
        """Create image with single button"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (200, 150), (400, 200), (52, 152, 219), -1)
        cv2.putText(image, 'Click Me', (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_complex_ui_image(self) -> str:
        """Create image with complex UI elements"""
        image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        
        # Header
        cv2.rectangle(image, (0, 0), (1200, 80), (51, 51, 51), -1)
        cv2.putText(image, 'Complex UI', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Navigation buttons
        buttons = [
            ('Home', 200, 100, 150, 30, (46, 204, 113)),
            ('Products', 370, 100, 150, 30, (52, 152, 219)),
            ('About', 540, 100, 150, 30, (155, 89, 182)),
            ('Contact', 710, 100, 150, 30, (231, 76, 60))
        ]
        
        for text, x, y, w, h, color in buttons:
            cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
            cv2.putText(image, text, (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Form elements
        cv2.rectangle(image, (100, 200), (500, 240), (255, 255, 255), -1)
        cv2.rectangle(image, (100, 200), (500, 240), (200, 200, 200), 2)
        cv2.putText(image, 'Enter text here...', (110, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Links
        cv2.putText(image, 'Learn more', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(image, 'Help Center', (100, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Large CTA button
        cv2.rectangle(image, (400, 400), (700, 460), (230, 126, 34), -1)
        cv2.putText(image, 'GET STARTED', (450, 435), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_text_heavy_image(self) -> str:
        """Create image with lots of text"""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        texts = [
            'Welcome to our platform',
            'This is a sample paragraph with multiple lines',
            'of text that should be detected by OCR',
            'Click here to continue',
            'Learn more about our services',
            'Contact us for support',
            'Privacy Policy',
            'Terms of Service'
        ]
        
        for i, text in enumerate(texts):
            y = 50 + i * 40
            cv2.putText(image, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add some clickable elements
        cv2.rectangle(image, (50, 400), (200, 440), (52, 152, 219), -1)
        cv2.putText(image, 'Click Here', (70, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_minimal_ui_image(self) -> str:
        """Create image with minimal UI elements"""
        image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Single small button
        cv2.rectangle(image, (150, 120), (250, 160), (52, 152, 219), -1)
        cv2.putText(image, 'OK', (180, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_empty_image(self) -> str:
        """Create empty white image"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_high_contrast_image(self) -> str:
        """Create high contrast image"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Black button on white background
        cv2.rectangle(image, (200, 150), (400, 200), (0, 0, 0), -1)
        cv2.putText(image, 'HIGH CONTRAST', (210, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_low_contrast_image(self) -> str:
        """Create low contrast image"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 240
        
        # Light gray button on light background
        cv2.rectangle(image, (200, 150), (400, 200), (220, 220, 220), -1)
        cv2.putText(image, 'Low Contrast', (210, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_overlapping_elements(self) -> str:
        """Create image with overlapping elements"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Overlapping rectangles
        cv2.rectangle(image, (100, 100), (300, 200), (255, 0, 0), -1)
        cv2.rectangle(image, (200, 150), (400, 250), (0, 255, 0), -1)
        cv2.rectangle(image, (150, 200), (350, 300), (0, 0, 255), -1)
        
        cv2.putText(image, 'Button 1', (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, 'Button 2', (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, 'Button 3', (170, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_small_elements(self) -> str:
        """Create image with very small elements"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Small buttons
        small_buttons = [
            (100, 100, 20, 15), (130, 100, 20, 15), (160, 100, 20, 15),
            (100, 130, 20, 15), (130, 130, 20, 15), (160, 130, 20, 15)
        ]
        
        for i, (x, y, w, h) in enumerate(small_buttons):
            cv2.rectangle(image, (x, y), (x+w, y+h), (52, 152, 219), -1)
            cv2.putText(image, str(i+1), (x+5, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def _create_large_elements(self) -> str:
        """Create image with very large elements"""
        image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        
        # Large button covering most of the screen
        cv2.rectangle(image, (50, 50), (1150, 750), (52, 152, 219), -1)
        cv2.putText(image, 'LARGE BUTTON', (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 3)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def test_basic_functionality(self):
        """Test basic screenshot processing functionality"""
        logger.info("Testing basic functionality...")
        
        try:
            result = self.processor.process_screenshot(self.test_images['simple_button'])
            
            self.assertIsInstance(result, dict)
            self.assertIn('screen_dimensions', result)
            self.assertIn('elements', result)
            self.assertIn('total_elements', result)
            self.assertGreater(result['total_elements'], 0)
            
            # Check element structure
            if result['elements']:
                element = result['elements'][0]
                required_fields = ['id', 'type', 'text', 'bbox', 'center', 'size', 'prominence', 'visibility']
                for field in required_fields:
                    self.assertIn(field, element)
            
            self.test_results.append({
                'test': 'basic_functionality',
                'status': 'PASS',
                'elements_found': result['total_elements'],
                'screen_size': result['screen_dimensions']
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'basic_functionality',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_complex_ui_processing(self):
        """Test processing of complex UI"""
        logger.info("Testing complex UI processing...")
        
        try:
            result = self.processor.process_screenshot(self.test_images['complex_ui'])
            
            self.assertGreater(result['total_elements'], 5)  # Should find multiple elements
            
            # Check for different element types
            element_types = [elem['type'] for elem in result['elements']]
            self.assertIn('button', element_types)
            
            self.test_results.append({
                'test': 'complex_ui_processing',
                'status': 'PASS',
                'elements_found': result['total_elements'],
                'element_types': list(set(element_types))
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'complex_ui_processing',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_text_detection(self):
        """Test OCR text detection"""
        logger.info("Testing text detection...")
        
        try:
            result = self.processor.process_screenshot(self.test_images['text_heavy'])
            
            # Should find text elements
            text_elements = [elem for elem in result['elements'] if elem['text']]
            self.assertGreater(len(text_elements), 0)
            
            # Check text content
            all_text = ' '.join([elem['text'] for elem in text_elements])
            self.assertIn('Click', all_text.lower())
            
            self.test_results.append({
                'test': 'text_detection',
                'status': 'PASS',
                'text_elements': len(text_elements),
                'sample_text': text_elements[0]['text'] if text_elements else None
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'text_detection',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_empty_image_handling(self):
        """Test handling of empty images"""
        logger.info("Testing empty image handling...")
        
        try:
            result = self.processor.process_screenshot(self.test_images['empty_image'])
            
            # Should handle empty images gracefully
            self.assertIsInstance(result, dict)
            self.assertEqual(result['total_elements'], 0)
            
            self.test_results.append({
                'test': 'empty_image_handling',
                'status': 'PASS',
                'elements_found': result['total_elements']
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'empty_image_handling',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_contrast_analysis(self):
        """Test contrast analysis"""
        logger.info("Testing contrast analysis...")
        
        try:
            high_contrast_result = self.processor.process_screenshot(self.test_images['high_contrast'])
            low_contrast_result = self.processor.process_screenshot(self.test_images['low_contrast'])
            
            # High contrast should have higher prominence
            if high_contrast_result['elements'] and low_contrast_result['elements']:
                high_prominence = high_contrast_result['elements'][0]['prominence']
                low_prominence = low_contrast_result['elements'][0]['prominence']
                
                self.assertGreater(high_prominence, low_prominence)
            
            self.test_results.append({
                'test': 'contrast_analysis',
                'status': 'PASS',
                'high_contrast_elements': high_contrast_result['total_elements'],
                'low_contrast_elements': low_contrast_result['total_elements']
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'contrast_analysis',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_element_deduplication(self):
        """Test duplicate element removal"""
        logger.info("Testing element deduplication...")
        
        try:
            result = self.processor.process_screenshot(self.test_images['overlapping_elements'])
            
            # Should remove overlapping duplicates
            self.assertLess(result['total_elements'], 6)  # Less than 3 overlapping elements
            
            self.test_results.append({
                'test': 'element_deduplication',
                'status': 'PASS',
                'elements_after_deduplication': result['total_elements']
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'element_deduplication',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_size_filtering(self):
        """Test element size filtering"""
        logger.info("Testing size filtering...")
        
        try:
            small_result = self.processor.process_screenshot(self.test_images['small_elements'])
            large_result = self.processor.process_screenshot(self.test_images['large_elements'])
            
            # Small elements might be filtered out
            self.assertLessEqual(small_result['total_elements'], 6)
            
            # Large element should be detected
            self.assertGreater(large_result['total_elements'], 0)
            
            self.test_results.append({
                'test': 'size_filtering',
                'status': 'PASS',
                'small_elements': small_result['total_elements'],
                'large_elements': large_result['total_elements']
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'size_filtering',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_invalid_image_path(self):
        """Test handling of invalid image paths"""
        logger.info("Testing invalid image path handling...")
        
        try:
            with self.assertRaises(ValueError):
                self.processor.process_screenshot('/nonexistent/path.png')
            
            self.test_results.append({
                'test': 'invalid_image_path',
                'status': 'PASS',
                'note': 'Correctly raised ValueError for invalid path'
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'invalid_image_path',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_element_classification(self):
        """Test element type classification"""
        logger.info("Testing element classification...")
        
        try:
            result = self.processor.process_screenshot(self.test_images['complex_ui'])
            
            # Check element type classification
            element_types = [elem['type'] for elem in result['elements']]
            valid_types = ['button', 'link', 'text', 'form', 'menu', 'image']
            
            for elem_type in element_types:
                self.assertIn(elem_type, valid_types)
            
            self.test_results.append({
                'test': 'element_classification',
                'status': 'PASS',
                'types_found': list(set(element_types))
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'element_classification',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_prominence_calculation(self):
        """Test prominence score calculation"""
        logger.info("Testing prominence calculation...")
        
        try:
            result = self.processor.process_screenshot(self.test_images['complex_ui'])
            
            for element in result['elements']:
                prominence = element['prominence']
                self.assertGreaterEqual(prominence, 0.0)
                self.assertLessEqual(prominence, 1.0)
            
            self.test_results.append({
                'test': 'prominence_calculation',
                'status': 'PASS',
                'prominence_range': [
                    min([e['prominence'] for e in result['elements']]),
                    max([e['prominence'] for e in result['elements']])
                ] if result['elements'] else [0, 0]
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'prominence_calculation',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    @patch('easyocr.Reader')
    def test_ocr_failure_handling(self, mock_ocr):
        """Test handling of OCR failures"""
        logger.info("Testing OCR failure handling...")
        
        try:
            # Mock OCR failure
            mock_ocr.return_value.readtext.side_effect = Exception("OCR failed")
            
            processor = ScreenshotProcessor()
            result = processor.process_screenshot(self.test_images['simple_button'])
            
            # Should handle OCR failure gracefully
            self.assertIsInstance(result, dict)
            
            self.test_results.append({
                'test': 'ocr_failure_handling',
                'status': 'PASS',
                'note': 'Handled OCR failure gracefully'
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'ocr_failure_handling',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def test_performance_metrics(self):
        """Test performance metrics"""
        logger.info("Testing performance metrics...")
        
        try:
            import time
            
            start_time = time.time()
            result = self.processor.process_screenshot(self.test_images['complex_ui'])
            processing_time = time.time() - start_time
            
            # Should process within reasonable time
            self.assertLess(processing_time, 10.0)  # 10 seconds max
            
            self.test_results.append({
                'test': 'performance_metrics',
                'status': 'PASS',
                'processing_time': processing_time,
                'elements_per_second': result['total_elements'] / processing_time if processing_time > 0 else 0
            })
            
        except Exception as e:
            self.errors.append({
                'test': 'performance_metrics',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'module': 'ScreenshotProcessor',
            'timestamp': os.popen('date').read().strip(),
            'total_tests': len(self.test_results) + len(self.errors),
            'passed_tests': len(self.test_results),
            'failed_tests': len(self.errors),
            'success_rate': len(self.test_results) / (len(self.test_results) + len(self.errors)) * 100,
            'test_results': self.test_results,
            'errors': self.errors,
            'test_images': list(self.test_images.keys())
        }
        
        return report


def run_screenshot_processor_tests():
    """Run all screenshot processor tests and generate report"""
    logger.info("Starting Screenshot Processor comprehensive testing...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_class = TestScreenshotProcessorDetailed
    
    # Add all test methods
    test_methods = [
        'test_basic_functionality',
        'test_complex_ui_processing',
        'test_text_detection',
        'test_empty_image_handling',
        'test_contrast_analysis',
        'test_element_deduplication',
        'test_size_filtering',
        'test_invalid_image_path',
        'test_element_classification',
        'test_prominence_calculation',
        'test_ocr_failure_handling',
        'test_performance_metrics'
    ]
    
    for method in test_methods:
        test_suite.addTest(test_class(method))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate report
    if hasattr(result, 'testsRun'):
        test_instance = test_class()
        test_instance.setUp()
        
        # Run tests to collect results
        for method in test_methods:
            try:
                getattr(test_instance, method)()
            except Exception as e:
                logger.error(f"Test {method} failed: {str(e)}")
        
        report = test_instance.generate_test_report()
        test_instance.tearDown()
        
        return report
    
    return None


if __name__ == '__main__':
    # Run tests and generate report
    report = run_screenshot_processor_tests()
    
    if report:
        print(f"\n{'='*60}")
        print("SCREENSHOT PROCESSOR TEST REPORT")
        print(f"{'='*60}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        
        if report['errors']:
            print(f"\nErrors:")
            for error in report['errors']:
                print(f"  - {error['test']}: {error['error']}")
    
    unittest.main()