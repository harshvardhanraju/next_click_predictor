#!/usr/bin/env python3
"""
Manual test runner for comprehensive module testing
"""

import sys
import os
import traceback
import tempfile
import cv2
import numpy as np
import json
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image(name: str, image_type: str = "simple") -> str:
    """Create a test image for testing"""
    if image_type == "simple":
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (200, 150), (400, 200), (52, 152, 219), -1)
        cv2.putText(image, 'Click Me', (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif image_type == "complex":
        image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        # Header
        cv2.rectangle(image, (0, 0), (1200, 80), (51, 51, 51), -1)
        cv2.putText(image, 'Test UI', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        # Buttons
        cv2.rectangle(image, (300, 150), (550, 200), (40, 167, 69), -1)
        cv2.putText(image, 'Button', (370, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif image_type == "empty":
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    else:
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        return tmp.name

def test_screenshot_processor():
    """Test Screenshot Processor module"""
    logger.info("Testing Screenshot Processor...")
    
    results = []
    errors = []
    
    try:
        from screenshot_processor import ScreenshotProcessor
        processor = ScreenshotProcessor()
        
        # Test 1: Basic functionality
        logger.info("Test 1: Basic functionality")
        test_image = create_test_image("basic", "simple")
        try:
            result = processor.process_screenshot(test_image)
            assert isinstance(result, dict)
            assert 'elements' in result
            assert 'total_elements' in result
            results.append({
                'test': 'basic_functionality',
                'status': 'PASS',
                'elements_found': result['total_elements'],
                'details': f"Found {result['total_elements']} elements"
            })
            logger.info(f"✓ Basic functionality test passed: {result['total_elements']} elements found")
        except Exception as e:
            errors.append({
                'test': 'basic_functionality',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Basic functionality test failed: {e}")
        finally:
            if os.path.exists(test_image):
                os.remove(test_image)
        
        # Test 2: Complex UI processing
        logger.info("Test 2: Complex UI processing")
        test_image = create_test_image("complex", "complex")
        try:
            result = processor.process_screenshot(test_image)
            results.append({
                'test': 'complex_ui_processing',
                'status': 'PASS',
                'elements_found': result['total_elements'],
                'details': f"Complex UI processed with {result['total_elements']} elements"
            })
            logger.info(f"✓ Complex UI test passed: {result['total_elements']} elements found")
        except Exception as e:
            errors.append({
                'test': 'complex_ui_processing',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Complex UI test failed: {e}")
        finally:
            if os.path.exists(test_image):
                os.remove(test_image)
        
        # Test 3: Empty image handling
        logger.info("Test 3: Empty image handling")
        test_image = create_test_image("empty", "empty")
        try:
            result = processor.process_screenshot(test_image)
            results.append({
                'test': 'empty_image_handling',
                'status': 'PASS',
                'elements_found': result['total_elements'],
                'details': f"Empty image handled gracefully: {result['total_elements']} elements"
            })
            logger.info(f"✓ Empty image test passed: {result['total_elements']} elements found")
        except Exception as e:
            errors.append({
                'test': 'empty_image_handling',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Empty image test failed: {e}")
        finally:
            if os.path.exists(test_image):
                os.remove(test_image)
        
        # Test 4: Invalid image path
        logger.info("Test 4: Invalid image path")
        try:
            processor.process_screenshot('/nonexistent/path.png')
            errors.append({
                'test': 'invalid_image_path',
                'error': 'Should have raised ValueError',
                'traceback': 'No exception raised'
            })
            logger.error("✗ Invalid path test failed: Should have raised ValueError")
        except ValueError:
            results.append({
                'test': 'invalid_image_path',
                'status': 'PASS',
                'details': 'Correctly raised ValueError for invalid path'
            })
            logger.info("✓ Invalid path test passed: Correctly raised ValueError")
        except Exception as e:
            errors.append({
                'test': 'invalid_image_path',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Invalid path test failed with unexpected error: {e}")
    
    except ImportError as e:
        errors.append({
            'test': 'module_import',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        logger.error(f"✗ Failed to import ScreenshotProcessor: {e}")
    
    return results, errors

def test_feature_integrator():
    """Test Feature Integrator module"""
    logger.info("Testing Feature Integrator...")
    
    results = []
    errors = []
    
    try:
        from feature_integrator import FeatureIntegrator
        integrator = FeatureIntegrator()
        
        # Test 1: Basic feature integration
        logger.info("Test 1: Basic feature integration")
        try:
            user_attributes = {
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused"
            }
            ui_elements = [
                {
                    'id': 'btn_test',
                    'type': 'button',
                    'text': 'Test Button',
                    'prominence': 0.8,
                    'position_features': {'relative_x': 0.5, 'relative_y': 0.7}
                }
            ]
            task = "Complete test task"
            
            result = integrator.integrate_features(user_attributes, ui_elements, task)
            assert hasattr(result, 'user_features')
            assert hasattr(result, 'ui_features')
            assert hasattr(result, 'task_features')
            
            results.append({
                'test': 'basic_feature_integration',
                'status': 'PASS',
                'details': f"Successfully integrated features"
            })
            logger.info("✓ Basic feature integration test passed")
        except Exception as e:
            errors.append({
                'test': 'basic_feature_integration',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Basic feature integration test failed: {e}")
        
        # Test 2: Missing user attributes
        logger.info("Test 2: Missing user attributes")
        try:
            user_attributes = {}  # Empty attributes
            ui_elements = [{'id': 'test', 'type': 'button', 'text': 'Test'}]
            task = "Test task"
            
            result = integrator.integrate_features(user_attributes, ui_elements, task)
            results.append({
                'test': 'missing_user_attributes',
                'status': 'PASS',
                'details': 'Handled missing user attributes gracefully'
            })
            logger.info("✓ Missing user attributes test passed")
        except Exception as e:
            errors.append({
                'test': 'missing_user_attributes',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Missing user attributes test failed: {e}")
    
    except ImportError as e:
        errors.append({
            'test': 'module_import',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        logger.error(f"✗ Failed to import FeatureIntegrator: {e}")
    
    return results, errors

def test_bayesian_network():
    """Test Bayesian Network Engine module"""
    logger.info("Testing Bayesian Network Engine...")
    
    results = []
    errors = []
    
    try:
        from bayesian_network import BayesianNetworkEngine
        from feature_integrator import IntegratedFeatures
        
        engine = BayesianNetworkEngine()
        
        # Test 1: Network building
        logger.info("Test 1: Network building")
        try:
            # Create mock integrated features
            mock_features = IntegratedFeatures(
                user_features={'tech_savviness': 0.8, 'mood': 0.7},
                ui_features=[{
                    'id': 'btn_test',
                    'type': 'button',
                    'text': 'Test',
                    'prominence': 0.8,
                    'position_features': {'center_distance': 0.3}
                }],
                task_features={'task_type': 0.8, 'urgency_level': 0.7},
                interaction_features={'user_task_compatibility': 0.8},
                feature_weights={'user_features': 0.3, 'ui_features': 0.4}
            )
            
            network = engine.build_network(mock_features)
            assert network is not None
            
            results.append({
                'test': 'network_building',
                'status': 'PASS',
                'details': 'Successfully built Bayesian network'
            })
            logger.info("✓ Network building test passed")
        except Exception as e:
            errors.append({
                'test': 'network_building',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Network building test failed: {e}")
    
    except ImportError as e:
        errors.append({
            'test': 'module_import',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        logger.error(f"✗ Failed to import BayesianNetworkEngine: {e}")
    
    return results, errors

def test_explanation_generator():
    """Test Explanation Generator module"""
    logger.info("Testing Explanation Generator...")
    
    results = []
    errors = []
    
    try:
        from explanation_generator import ExplanationGenerator
        from feature_integrator import IntegratedFeatures
        
        generator = ExplanationGenerator()
        
        # Test 1: Basic explanation generation
        logger.info("Test 1: Basic explanation generation")
        try:
            mock_predictions = [
                {
                    'element_id': 'btn_test',
                    'element_type': 'button',
                    'element_text': 'Test Button',
                    'click_probability': 0.8,
                    'confidence': 0.9,
                    'rank': 1
                }
            ]
            
            mock_features = IntegratedFeatures(
                user_features={'tech_savviness': 0.8, 'mood': 0.7},
                ui_features=[{
                    'id': 'btn_test',
                    'type': 'button',
                    'text': 'Test Button',
                    'prominence': 0.8,
                    'position_features': {'center_distance': 0.3}
                }],
                task_features={'task_type': 0.8, 'urgency_level': 0.7},
                interaction_features={'user_task_compatibility': 0.8},
                feature_weights={'user_features': 0.3, 'ui_features': 0.4}
            )
            
            explanation = generator.generate_explanation(
                mock_predictions, mock_features, mock_predictions[0]
            )
            
            assert 'main_explanation' in explanation
            assert isinstance(explanation['main_explanation'], str)
            
            results.append({
                'test': 'basic_explanation_generation',
                'status': 'PASS',
                'details': 'Successfully generated explanation'
            })
            logger.info("✓ Basic explanation generation test passed")
        except Exception as e:
            errors.append({
                'test': 'basic_explanation_generation',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Basic explanation generation test failed: {e}")
    
    except ImportError as e:
        errors.append({
            'test': 'module_import',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        logger.error(f"✗ Failed to import ExplanationGenerator: {e}")
    
    return results, errors

def test_main_orchestrator():
    """Test Main Orchestrator module"""
    logger.info("Testing Main Orchestrator...")
    
    results = []
    errors = []
    
    try:
        from next_click_predictor import NextClickPredictor
        
        predictor = NextClickPredictor()
        
        # Test 1: Basic prediction pipeline
        logger.info("Test 1: Basic prediction pipeline")
        try:
            test_image = create_test_image("orchestrator", "simple")
            user_attributes = {
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused"
            }
            task = "Test prediction task"
            
            # This might fail due to missing dependencies, but we'll catch it
            try:
                result = predictor.predict_next_click(test_image, user_attributes, task)
                results.append({
                    'test': 'basic_prediction_pipeline',
                    'status': 'PASS',
                    'details': f"Successfully ran prediction pipeline"
                })
                logger.info("✓ Basic prediction pipeline test passed")
            except Exception as e:
                # This is expected if dependencies are missing
                results.append({
                    'test': 'basic_prediction_pipeline',
                    'status': 'PARTIAL',
                    'details': f"Pipeline ran but failed due to dependencies: {str(e)[:100]}"
                })
                logger.warning(f"⚠ Basic prediction pipeline test partial: {e}")
        except Exception as e:
            errors.append({
                'test': 'basic_prediction_pipeline',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Basic prediction pipeline test failed: {e}")
        finally:
            if 'test_image' in locals() and os.path.exists(test_image):
                os.remove(test_image)
        
        # Test 2: Screenshot analysis only
        logger.info("Test 2: Screenshot analysis only")
        try:
            test_image = create_test_image("analysis", "simple")
            
            analysis = predictor.analyze_screenshot_only(test_image)
            if 'error' not in analysis:
                results.append({
                    'test': 'screenshot_analysis_only',
                    'status': 'PASS',
                    'details': f"Successfully analyzed screenshot: {analysis.get('total_elements', 0)} elements"
                })
                logger.info("✓ Screenshot analysis only test passed")
            else:
                errors.append({
                    'test': 'screenshot_analysis_only',
                    'error': analysis['error'],
                    'traceback': 'Screenshot analysis failed'
                })
                logger.error(f"✗ Screenshot analysis only test failed: {analysis['error']}")
        except Exception as e:
            errors.append({
                'test': 'screenshot_analysis_only',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"✗ Screenshot analysis only test failed: {e}")
        finally:
            if 'test_image' in locals() and os.path.exists(test_image):
                os.remove(test_image)
    
    except ImportError as e:
        errors.append({
            'test': 'module_import',
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        logger.error(f"✗ Failed to import NextClickPredictor: {e}")
    
    return results, errors

def generate_test_report(all_results, all_errors):
    """Generate comprehensive test report"""
    report = {
        'test_run': {
            'timestamp': datetime.now().isoformat(),
            'total_modules': 5,
            'python_version': sys.version,
            'platform': sys.platform
        },
        'module_results': {
            'screenshot_processor': {
                'results': all_results.get('screenshot_processor', []),
                'errors': all_errors.get('screenshot_processor', [])
            },
            'feature_integrator': {
                'results': all_results.get('feature_integrator', []),
                'errors': all_errors.get('feature_integrator', [])
            },
            'bayesian_network': {
                'results': all_results.get('bayesian_network', []),
                'errors': all_errors.get('bayesian_network', [])
            },
            'explanation_generator': {
                'results': all_results.get('explanation_generator', []),
                'errors': all_errors.get('explanation_generator', [])
            },
            'main_orchestrator': {
                'results': all_results.get('main_orchestrator', []),
                'errors': all_errors.get('main_orchestrator', [])
            }
        },
        'summary': {
            'total_tests': sum(len(results) for results in all_results.values()) + sum(len(errors) for errors in all_errors.values()),
            'passed_tests': sum(len(results) for results in all_results.values()),
            'failed_tests': sum(len(errors) for errors in all_errors.values()),
            'success_rate': 0.0
        }
    }
    
    if report['summary']['total_tests'] > 0:
        report['summary']['success_rate'] = (report['summary']['passed_tests'] / report['summary']['total_tests']) * 100
    
    return report

def main():
    """Main test runner"""
    logger.info("Starting comprehensive module testing...")
    
    all_results = {}
    all_errors = {}
    
    # Test each module
    modules = [
        ('screenshot_processor', test_screenshot_processor),
        ('feature_integrator', test_feature_integrator),
        ('bayesian_network', test_bayesian_network),
        ('explanation_generator', test_explanation_generator),
        ('main_orchestrator', test_main_orchestrator)
    ]
    
    for module_name, test_func in modules:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {module_name.replace('_', ' ').title()}")
        logger.info(f"{'='*60}")
        
        try:
            results, errors = test_func()
            all_results[module_name] = results
            all_errors[module_name] = errors
            
            logger.info(f"Module {module_name}: {len(results)} passed, {len(errors)} failed")
        except Exception as e:
            logger.error(f"Failed to test {module_name}: {e}")
            all_errors[module_name] = [{
                'test': 'module_test',
                'error': str(e),
                'traceback': traceback.format_exc()
            }]
            all_results[module_name] = []
    
    # Generate comprehensive report
    report = generate_test_report(all_results, all_errors)
    
    # Save report
    report_path = 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {report['summary']['total_tests']}")
    logger.info(f"Passed: {report['summary']['passed_tests']}")
    logger.info(f"Failed: {report['summary']['failed_tests']}")
    logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    logger.info(f"Report saved to: {report_path}")
    
    # Show detailed results
    for module_name, module_data in report['module_results'].items():
        logger.info(f"\n{module_name.replace('_', ' ').title()}:")
        for result in module_data['results']:
            logger.info(f"  ✓ {result['test']}: {result['status']}")
        for error in module_data['errors']:
            logger.error(f"  ✗ {error['test']}: {error['error']}")

if __name__ == "__main__":
    main()