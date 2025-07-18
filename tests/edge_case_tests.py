#!/usr/bin/env python3
"""
Comprehensive edge case testing suite for the Next-Click Prediction System
"""

import sys
import os
import traceback
import tempfile
import cv2
import numpy as np
import json
import time
from datetime import datetime
import logging
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeCaseTestSuite:
    """Comprehensive edge case testing suite"""
    
    def __init__(self):
        self.test_results = []
        self.errors = []
        self.performance_metrics = {}
        self.temp_files = []
    
    def cleanup(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def create_edge_case_image(self, case_type: str) -> str:
        """Create edge case images for testing"""
        if case_type == "very_large":
            # 4K resolution image
            image = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
            # Add some elements
            cv2.rectangle(image, (1000, 1000), (2000, 1200), (52, 152, 219), -1)
            cv2.putText(image, 'Large Screen Button', (1100, 1100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            
        elif case_type == "very_small":
            # Very small image
            image = np.ones((50, 80, 3), dtype=np.uint8) * 255
            cv2.rectangle(image, (10, 10), (70, 40), (52, 152, 219), -1)
            
        elif case_type == "monochrome":
            # Black and white image
            image = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.rectangle(image, (200, 150), (400, 200), (0, 0, 0), -1)
            cv2.putText(image, 'Mono', (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        elif case_type == "noisy":
            # Noisy image
            image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
            cv2.rectangle(image, (200, 150), (400, 200), (52, 152, 219), -1)
            cv2.putText(image, 'Noisy', (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        elif case_type == "corrupted":
            # Partially corrupted image
            image = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.rectangle(image, (200, 150), (400, 200), (52, 152, 219), -1)
            # Add corruption
            image[100:300, 100:300] = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            
        elif case_type == "rotated":
            # Rotated image
            image = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.rectangle(image, (200, 150), (400, 200), (52, 152, 219), -1)
            # Rotate 45 degrees
            center = (300, 200)
            rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (600, 400))
            
        elif case_type == "gradient":
            # Gradient background
            image = np.zeros((400, 600, 3), dtype=np.uint8)
            for i in range(400):
                for j in range(600):
                    image[i, j] = [i//2, j//3, (i+j)//4]
            cv2.rectangle(image, (200, 150), (400, 200), (255, 255, 255), -1)
            cv2.putText(image, 'Gradient', (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
        else:
            # Default case
            image = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.rectangle(image, (200, 150), (400, 200), (52, 152, 219), -1)
        
        # Save image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            self.temp_files.append(tmp.name)
            return tmp.name
    
    def test_screenshot_processor_edge_cases(self):
        """Test Screenshot Processor with edge cases"""
        logger.info("Testing Screenshot Processor edge cases...")
        
        try:
            from screenshot_processor import ScreenshotProcessor
            processor = ScreenshotProcessor()
            
            edge_cases = [
                ("very_large", "4K resolution image"),
                ("very_small", "Very small image"),
                ("monochrome", "Black and white image"),
                ("noisy", "Noisy image"),
                ("corrupted", "Partially corrupted image"),
                ("rotated", "Rotated image"),
                ("gradient", "Gradient background")
            ]
            
            for case_type, description in edge_cases:
                logger.info(f"Testing {description}...")
                
                start_time = time.time()
                try:
                    test_image = self.create_edge_case_image(case_type)
                    result = processor.process_screenshot(test_image)
                    
                    processing_time = time.time() - start_time
                    
                    self.test_results.append({
                        'module': 'screenshot_processor',
                        'test': f'edge_case_{case_type}',
                        'status': 'PASS',
                        'processing_time': processing_time,
                        'elements_found': result['total_elements'],
                        'description': description
                    })
                    
                    logger.info(f"✓ {description} passed: {result['total_elements']} elements in {processing_time:.2f}s")
                    
                except Exception as e:
                    self.errors.append({
                        'module': 'screenshot_processor',
                        'test': f'edge_case_{case_type}',
                        'error': str(e),
                        'description': description,
                        'traceback': traceback.format_exc()
                    })
                    logger.error(f"✗ {description} failed: {e}")
                    
        except ImportError as e:
            self.errors.append({
                'module': 'screenshot_processor',
                'test': 'edge_case_import',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    def test_feature_integrator_edge_cases(self):
        """Test Feature Integrator with edge cases"""
        logger.info("Testing Feature Integrator edge cases...")
        
        try:
            from feature_integrator import FeatureIntegrator
            integrator = FeatureIntegrator()
            
            edge_cases = [
                ({}, [], ""),  # All empty
                ({"invalid_key": "value"}, [], ""),  # Invalid user attributes
                ({"age_group": "999"}, [], ""),  # Invalid age group
                ({"tech_savviness": "ultra_expert"}, [], ""),  # Invalid tech level
                ({"mood": "extremely_frustrated"}, [], ""),  # Invalid mood
                ({"age_group": "25-34"}, [{"id": "test"}], ""),  # Minimal UI element
                ({"age_group": "25-34"}, [{"id": "test", "type": "unknown_type"}], ""),  # Unknown element type
                ({"age_group": "25-34"}, [], "A" * 1000),  # Very long task description
                ({"age_group": "25-34"}, [], "Task with special chars: !@#$%^&*()"),  # Special characters
                ({"age_group": "25-34"}, [{"id": f"elem_{i}", "type": "button"} for i in range(100)], "Test"),  # Many elements
            ]
            
            for i, (user_attrs, ui_elements, task) in enumerate(edge_cases):
                logger.info(f"Testing feature integration case {i+1}...")
                
                start_time = time.time()
                try:
                    result = integrator.integrate_features(user_attrs, ui_elements, task)
                    processing_time = time.time() - start_time
                    
                    self.test_results.append({
                        'module': 'feature_integrator',
                        'test': f'edge_case_{i+1}',
                        'status': 'PASS',
                        'processing_time': processing_time,
                        'description': f"Case {i+1}: {len(ui_elements)} elements, {len(task)} char task"
                    })
                    
                    logger.info(f"✓ Feature integration case {i+1} passed in {processing_time:.2f}s")
                    
                except Exception as e:
                    self.errors.append({
                        'module': 'feature_integrator',
                        'test': f'edge_case_{i+1}',
                        'error': str(e),
                        'description': f"Case {i+1}",
                        'traceback': traceback.format_exc()
                    })
                    logger.error(f"✗ Feature integration case {i+1} failed: {e}")
                    
        except ImportError as e:
            self.errors.append({
                'module': 'feature_integrator',
                'test': 'edge_case_import',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    def test_bayesian_network_edge_cases(self):
        """Test Bayesian Network with edge cases"""
        logger.info("Testing Bayesian Network edge cases...")
        
        try:
            from bayesian_network import BayesianNetworkEngine
            from feature_integrator import IntegratedFeatures
            
            engine = BayesianNetworkEngine()
            
            edge_cases = [
                # Empty features
                IntegratedFeatures(
                    user_features={},
                    ui_features=[],
                    task_features={},
                    interaction_features={},
                    feature_weights={}
                ),
                # Single element
                IntegratedFeatures(
                    user_features={'tech_savviness': 0.5},
                    ui_features=[{'id': 'test', 'type': 'button'}],
                    task_features={'task_type': 0.5},
                    interaction_features={},
                    feature_weights={'user_features': 1.0}
                ),
                # Many elements
                IntegratedFeatures(
                    user_features={'tech_savviness': 0.8},
                    ui_features=[{'id': f'elem_{i}', 'type': 'button'} for i in range(20)],
                    task_features={'task_type': 0.8},
                    interaction_features={},
                    feature_weights={'user_features': 0.5, 'ui_features': 0.5}
                ),
                # Extreme values
                IntegratedFeatures(
                    user_features={'tech_savviness': 1.0, 'mood': 0.0},
                    ui_features=[{'id': 'test', 'type': 'button', 'prominence': 1.0}],
                    task_features={'task_type': 1.0, 'urgency_level': 1.0},
                    interaction_features={'user_task_compatibility': 1.0},
                    feature_weights={'user_features': 1.0}
                )
            ]
            
            for i, features in enumerate(edge_cases):
                logger.info(f"Testing Bayesian network case {i+1}...")
                
                start_time = time.time()
                try:
                    network = engine.build_network(features)
                    predictions = engine.predict_clicks(features)
                    processing_time = time.time() - start_time
                    
                    self.test_results.append({
                        'module': 'bayesian_network',
                        'test': f'edge_case_{i+1}',
                        'status': 'PASS',
                        'processing_time': processing_time,
                        'predictions_count': len(predictions),
                        'description': f"Case {i+1}: {len(features.ui_features)} elements"
                    })
                    
                    logger.info(f"✓ Bayesian network case {i+1} passed: {len(predictions)} predictions in {processing_time:.2f}s")
                    
                except Exception as e:
                    self.errors.append({
                        'module': 'bayesian_network',
                        'test': f'edge_case_{i+1}',
                        'error': str(e),
                        'description': f"Case {i+1}",
                        'traceback': traceback.format_exc()
                    })
                    logger.error(f"✗ Bayesian network case {i+1} failed: {e}")
                    
        except ImportError as e:
            self.errors.append({
                'module': 'bayesian_network',
                'test': 'edge_case_import',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        logger.info("Running performance benchmarks...")
        
        try:
            from next_click_predictor import NextClickPredictor
            
            predictor = NextClickPredictor()
            
            # Test different image sizes
            image_sizes = [
                (400, 600, "Small"),
                (800, 1200, "Medium"),
                (1080, 1920, "Large"),
                (2160, 3840, "4K")
            ]
            
            for height, width, size_name in image_sizes:
                logger.info(f"Testing {size_name} image performance...")
                
                # Create test image
                image = np.ones((height, width, 3), dtype=np.uint8) * 255
                for i in range(5):  # Add 5 elements
                    x = (i + 1) * width // 6
                    y = height // 2
                    cv2.rectangle(image, (x-50, y-25), (x+50, y+25), (52, 152, 219), -1)
                    cv2.putText(image, f'Btn{i+1}', (x-20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cv2.imwrite(tmp.name, image)
                    self.temp_files.append(tmp.name)
                    
                    # Test analysis only
                    start_time = time.time()
                    try:
                        result = predictor.analyze_screenshot_only(tmp.name)
                        analysis_time = time.time() - start_time
                        
                        self.performance_metrics[f'analysis_{size_name.lower()}'] = {
                            'time': analysis_time,
                            'elements': result.get('total_elements', 0),
                            'image_size': f"{width}x{height}"
                        }
                        
                        logger.info(f"✓ {size_name} analysis: {analysis_time:.2f}s, {result.get('total_elements', 0)} elements")
                        
                    except Exception as e:
                        logger.error(f"✗ {size_name} analysis failed: {e}")
                        
        except ImportError as e:
            logger.error(f"Performance test import failed: {e}")
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        logger.info("Testing memory usage...")
        
        try:
            import psutil
            import os
            
            from next_click_predictor import NextClickPredictor
            
            predictor = NextClickPredictor()
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test with increasing number of elements
            for num_elements in [1, 10, 50, 100]:
                logger.info(f"Testing memory with {num_elements} elements...")
                
                # Create complex image
                image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
                for i in range(num_elements):
                    x = (i % 10) * 120
                    y = (i // 10) * 80 + 50
                    cv2.rectangle(image, (x, y), (x+100, y+50), (52, 152, 219), -1)
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cv2.imwrite(tmp.name, image)
                    self.temp_files.append(tmp.name)
                    
                    try:
                        result = predictor.analyze_screenshot_only(tmp.name)
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        
                        self.performance_metrics[f'memory_{num_elements}_elements'] = {
                            'memory_mb': current_memory,
                            'memory_increase': current_memory - initial_memory,
                            'elements_found': result.get('total_elements', 0)
                        }
                        
                        logger.info(f"✓ {num_elements} elements: {current_memory:.1f}MB (+{current_memory-initial_memory:.1f}MB)")
                        
                    except Exception as e:
                        logger.error(f"✗ Memory test with {num_elements} elements failed: {e}")
                        
        except ImportError:
            logger.warning("psutil not available, skipping memory tests")
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        logger.info("Testing concurrent request handling...")
        
        try:
            from next_click_predictor import NextClickPredictor
            import threading
            import queue
            
            predictor = NextClickPredictor()
            
            # Create test image
            test_image = self.create_edge_case_image("default")
            
            def worker(result_queue, worker_id):
                """Worker function for concurrent testing"""
                try:
                    start_time = time.time()
                    result = predictor.analyze_screenshot_only(test_image)
                    processing_time = time.time() - start_time
                    
                    result_queue.put({
                        'worker_id': worker_id,
                        'success': True,
                        'processing_time': processing_time,
                        'elements': result.get('total_elements', 0)
                    })
                except Exception as e:
                    result_queue.put({
                        'worker_id': worker_id,
                        'success': False,
                        'error': str(e)
                    })
            
            # Test with different numbers of concurrent threads
            for num_threads in [1, 3, 5, 10]:
                logger.info(f"Testing with {num_threads} concurrent threads...")
                
                result_queue = queue.Queue()
                threads = []
                
                start_time = time.time()
                
                # Start threads
                for i in range(num_threads):
                    thread = threading.Thread(target=worker, args=(result_queue, i))
                    thread.start()
                    threads.append(thread)
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                total_time = time.time() - start_time
                
                # Collect results
                results = []
                while not result_queue.empty():
                    results.append(result_queue.get())
                
                successful = len([r for r in results if r['success']])
                failed = len([r for r in results if not r['success']])
                
                self.performance_metrics[f'concurrent_{num_threads}_threads'] = {
                    'total_time': total_time,
                    'successful': successful,
                    'failed': failed,
                    'avg_time_per_request': total_time / num_threads if num_threads > 0 else 0
                }
                
                logger.info(f"✓ {num_threads} threads: {successful} successful, {failed} failed in {total_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Concurrent test failed: {e}")
    
    def run_all_edge_case_tests(self):
        """Run all edge case tests"""
        logger.info("Starting comprehensive edge case testing...")
        
        start_time = time.time()
        
        # Run all test categories
        self.test_screenshot_processor_edge_cases()
        self.test_feature_integrator_edge_cases()
        self.test_bayesian_network_edge_cases()
        self.test_performance_benchmarks()
        self.test_memory_usage()
        self.test_concurrent_requests()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = {
            'edge_case_testing': {
                'timestamp': datetime.now().isoformat(),
                'total_runtime': total_time,
                'test_results': self.test_results,
                'errors': self.errors,
                'performance_metrics': self.performance_metrics,
                'summary': {
                    'total_tests': len(self.test_results) + len(self.errors),
                    'passed_tests': len(self.test_results),
                    'failed_tests': len(self.errors),
                    'success_rate': len(self.test_results) / (len(self.test_results) + len(self.errors)) * 100 if (len(self.test_results) + len(self.errors)) > 0 else 0
                }
            }
        }
        
        return report


def main():
    """Main function to run edge case tests"""
    logger.info("Starting Edge Case Testing Suite...")
    
    test_suite = EdgeCaseTestSuite()
    
    try:
        report = test_suite.run_all_edge_case_tests()
        
        # Save report
        with open('edge_case_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report['edge_case_testing']['summary']
        logger.info(f"\n{'='*60}")
        logger.info("EDGE CASE TESTING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total Runtime: {report['edge_case_testing']['total_runtime']:.2f}s")
        logger.info(f"Report saved to: edge_case_test_report.json")
        
        # Show performance metrics
        if report['edge_case_testing']['performance_metrics']:
            logger.info(f"\n{'='*40}")
            logger.info("PERFORMANCE METRICS")
            logger.info(f"{'='*40}")
            for metric, data in report['edge_case_testing']['performance_metrics'].items():
                logger.info(f"{metric}: {data}")
        
        # Show errors
        if report['edge_case_testing']['errors']:
            logger.info(f"\n{'='*40}")
            logger.info("ERRORS")
            logger.info(f"{'='*40}")
            for error in report['edge_case_testing']['errors']:
                logger.error(f"{error['module']}.{error['test']}: {error['error']}")
        
    except Exception as e:
        logger.error(f"Edge case testing failed: {e}")
        traceback.print_exc()
    
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    main()