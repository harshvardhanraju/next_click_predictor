import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import cv2

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from next_click_predictor import NextClickPredictor
from screenshot_processor import ScreenshotProcessor
from feature_integrator import FeatureIntegrator
from bayesian_network import BayesianNetworkEngine
from explanation_generator import ExplanationGenerator


class TestNextClickPredictor(unittest.TestCase):
    """Test suite for NextClickPredictor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = NextClickPredictor()
        self.test_user_attributes = {
            "age_group": "25-34",
            "tech_savviness": "high",
            "mood": "focused",
            "device_type": "desktop"
        }
        self.test_task = "Complete purchase. What would you click next?"
        
        # Create a temporary test image
        self.test_image_path = self._create_test_image()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def _create_test_image(self) -> str:
        """Create a test PNG image"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create a simple test image with some shapes
            image = Image.new('RGB', (800, 600), color='white')
            image.save(tmp.name)
            return tmp.name
    
    def test_predictor_initialization(self):
        """Test that predictor initializes correctly"""
        self.assertIsInstance(self.predictor.screenshot_processor, ScreenshotProcessor)
        self.assertIsInstance(self.predictor.feature_integrator, FeatureIntegrator)
        self.assertIsInstance(self.predictor.bayesian_engine, BayesianNetworkEngine)
        self.assertIsInstance(self.predictor.explanation_generator, ExplanationGenerator)
    
    @patch('src.screenshot_processor.ScreenshotProcessor.process_screenshot')
    def test_predict_next_click_with_mock_data(self, mock_process_screenshot):
        """Test prediction with mocked screenshot data"""
        # Mock screenshot processing result
        mock_process_screenshot.return_value = {
            'screen_dimensions': [800, 600],
            'total_elements': 2,
            'elements': [
                {
                    'id': 'btn_checkout',
                    'type': 'button',
                    'text': 'Checkout',
                    'bbox': [300, 400, 500, 450],
                    'center': [400, 425],
                    'size': [200, 50],
                    'prominence': 0.8,
                    'visibility': True,
                    'color_features': {'dominant_color': '#FF6B35', 'contrast': 0.9},
                    'position_features': {'relative_x': 0.5, 'relative_y': 0.7, 'quadrant': 'bottom_right'}
                },
                {
                    'id': 'btn_continue',
                    'type': 'button',
                    'text': 'Continue Shopping',
                    'bbox': [100, 400, 250, 450],
                    'center': [175, 425],
                    'size': [150, 50],
                    'prominence': 0.5,
                    'visibility': True,
                    'color_features': {'dominant_color': '#CCCCCC', 'contrast': 0.5},
                    'position_features': {'relative_x': 0.2, 'relative_y': 0.7, 'quadrant': 'bottom_left'}
                }
            ]
        }
        
        # Run prediction
        result = self.predictor.predict_next_click(
            self.test_image_path,
            self.test_user_attributes,
            self.test_task
        )
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIn('element_id', result.top_prediction)
        self.assertIn('click_probability', result.top_prediction)
        self.assertGreater(len(result.all_predictions), 0)
        self.assertIn('main_explanation', result.explanation)
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        with patch('src.screenshot_processor.ScreenshotProcessor.process_screenshot') as mock_process:
            mock_process.return_value = {
                'screen_dimensions': [800, 600],
                'total_elements': 1,
                'elements': [{
                    'id': 'test_element',
                    'type': 'button',
                    'text': 'Test Button',
                    'bbox': [100, 100, 200, 150],
                    'center': [150, 125],
                    'size': [100, 50],
                    'prominence': 0.7,
                    'visibility': True,
                    'color_features': {},
                    'position_features': {}
                }]
            }
            
            results = self.predictor.predict_batch(
                [self.test_image_path, self.test_image_path],
                self.test_user_attributes,
                [self.test_task, "Browse products"]
            )
            
            self.assertEqual(len(results), 2)
            self.assertIsNotNone(results[0])
            self.assertIsNotNone(results[1])
    
    def test_screenshot_analysis_only(self):
        """Test screenshot analysis without prediction"""
        with patch('src.screenshot_processor.ScreenshotProcessor.process_screenshot') as mock_process:
            mock_process.return_value = {
                'screen_dimensions': [800, 600],
                'total_elements': 1,
                'elements': [{
                    'id': 'test_element',
                    'type': 'button',
                    'text': 'Test Button',
                    'bbox': [100, 100, 200, 150],
                    'center': [150, 125],
                    'size': [100, 50],
                    'prominence': 0.7,
                    'visibility': True,
                    'color_features': {},
                    'position_features': {}
                }]
            }
            
            result = self.predictor.analyze_screenshot_only(self.test_image_path)
            
            self.assertIn('ui_elements', result)
            self.assertIn('screen_dimensions', result)
            self.assertIn('total_elements', result)
    
    def test_system_stats(self):
        """Test system statistics collection"""
        # Initially no stats
        stats = self.predictor.get_system_stats()
        self.assertIn('error', stats)
        
        # Add some mock history
        self.predictor.prediction_history = [
            {'processing_time': 1.0, 'confidence': 0.8, 'timestamp': 1234567890},
            {'processing_time': 1.5, 'confidence': 0.7, 'timestamp': 1234567891}
        ]
        
        stats = self.predictor.get_system_stats()
        self.assertIn('total_predictions', stats)
        self.assertIn('avg_processing_time', stats)
        self.assertIn('avg_confidence', stats)
    
    def test_save_and_load_prediction_result(self):
        """Test saving and loading prediction results"""
        # Create a mock result
        from next_click_predictor import PredictionResult
        
        mock_result = PredictionResult(
            top_prediction={'element_id': 'test', 'click_probability': 0.8},
            all_predictions=[{'element_id': 'test', 'click_probability': 0.8}],
            explanation={'main_explanation': 'Test explanation'},
            ui_elements=[],
            processing_time=1.0,
            confidence_score=0.8,
            metadata={'test': True}
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            self.predictor.save_prediction_result(mock_result, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Load back
            loaded_result = self.predictor.load_prediction_result(tmp_path)
            self.assertIsNotNone(loaded_result)
            self.assertEqual(loaded_result.top_prediction['element_id'], 'test')
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestScreenshotProcessor(unittest.TestCase):
    """Test suite for ScreenshotProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = ScreenshotProcessor()
        self.test_image_path = self._create_test_image()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def _create_test_image(self) -> str:
        """Create a test image with some UI elements"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create image with rectangles simulating UI elements
            image = np.ones((600, 800, 3), dtype=np.uint8) * 255
            
            # Draw a button-like rectangle
            cv2.rectangle(image, (300, 400), (500, 450), (255, 107, 53), -1)
            cv2.putText(image, 'Checkout', (320, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw another rectangle
            cv2.rectangle(image, (100, 400), (250, 450), (200, 200, 200), -1)
            cv2.putText(image, 'Continue', (110, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.imwrite(tmp.name, image)
            return tmp.name
    
    def test_process_screenshot_basic(self):
        """Test basic screenshot processing"""
        result = self.processor.process_screenshot(self.test_image_path)
        
        self.assertIn('screen_dimensions', result)
        self.assertIn('elements', result)
        self.assertIn('total_elements', result)
        self.assertGreater(result['total_elements'], 0)
    
    def test_element_detection(self):
        """Test UI element detection"""
        result = self.processor.process_screenshot(self.test_image_path)
        elements = result['elements']
        
        # Should detect at least one element
        self.assertGreater(len(elements), 0)
        
        # Check element structure
        for element in elements:
            self.assertIn('id', element)
            self.assertIn('type', element)
            self.assertIn('bbox', element)
            self.assertIn('prominence', element)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image paths"""
        with self.assertRaises(ValueError):
            self.processor.process_screenshot('/nonexistent/path.png')


class TestFeatureIntegrator(unittest.TestCase):
    """Test suite for FeatureIntegrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.integrator = FeatureIntegrator()
        self.test_user_attributes = {
            "age_group": "25-34",
            "tech_savviness": "high",
            "mood": "focused"
        }
        self.test_ui_elements = [
            {
                'id': 'btn_test',
                'type': 'button',
                'text': 'Test Button',
                'prominence': 0.8,
                'position_features': {'relative_x': 0.5, 'relative_y': 0.7}
            }
        ]
        self.test_task = "Complete purchase"
    
    def test_integrate_features_basic(self):
        """Test basic feature integration"""
        result = self.integrator.integrate_features(
            self.test_user_attributes,
            self.test_ui_elements,
            self.test_task
        )
        
        self.assertIsNotNone(result)
        self.assertIn('user_features', result.__dict__)
        self.assertIn('ui_features', result.__dict__)
        self.assertIn('task_features', result.__dict__)
    
    def test_user_feature_extraction(self):
        """Test user feature extraction"""
        result = self.integrator.integrate_features(
            self.test_user_attributes,
            self.test_ui_elements,
            self.test_task
        )
        
        user_features = result.user_features
        self.assertIn('tech_savviness', user_features)
        self.assertIn('mood', user_features)
        self.assertIsInstance(user_features['tech_savviness'], (int, float))
    
    def test_task_parsing(self):
        """Test task description parsing"""
        result = self.integrator.integrate_features(
            self.test_user_attributes,
            self.test_ui_elements,
            "Complete purchase. What would you click next?"
        )
        
        task_features = result.task_features
        self.assertIn('task_type', task_features)
        self.assertIn('urgency_level', task_features)


class TestBayesianNetworkEngine(unittest.TestCase):
    """Test suite for BayesianNetworkEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BayesianNetworkEngine()
        
        # Create mock integrated features
        from feature_integrator import IntegratedFeatures
        self.mock_features = IntegratedFeatures(
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
    
    def test_network_building(self):
        """Test Bayesian network construction"""
        network = self.engine.build_network(self.mock_features)
        
        self.assertIsNotNone(network)
        self.assertIsNotNone(self.engine.model)
        self.assertIsNotNone(self.engine.inference_engine)
    
    def test_prediction_generation(self):
        """Test click prediction generation"""
        # Build network first
        self.engine.build_network(self.mock_features)
        
        # Generate predictions
        predictions = self.engine.predict_clicks(self.mock_features)
        
        self.assertIsInstance(predictions, list)
        if predictions:  # May be empty due to mock data
            self.assertIn('element_id', predictions[0])
            self.assertIn('click_probability', predictions[0])


class TestExplanationGenerator(unittest.TestCase):
    """Test suite for ExplanationGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ExplanationGenerator()
        
        # Mock prediction data
        self.mock_predictions = [
            {
                'element_id': 'btn_checkout',
                'element_type': 'button',
                'element_text': 'Checkout',
                'click_probability': 0.8,
                'confidence': 0.9,
                'rank': 1
            }
        ]
        
        # Mock integrated features
        from feature_integrator import IntegratedFeatures
        self.mock_features = IntegratedFeatures(
            user_features={'tech_savviness': 0.8, 'mood': 0.7},
            ui_features=[{
                'id': 'btn_checkout',
                'type': 'button',
                'text': 'Checkout',
                'prominence': 0.8,
                'position_features': {'center_distance': 0.3}
            }],
            task_features={'task_type': 0.8, 'urgency_level': 0.7},
            interaction_features={'user_task_compatibility': 0.8},
            feature_weights={'user_features': 0.3, 'ui_features': 0.4}
        )
    
    def test_explanation_generation(self):
        """Test explanation generation"""
        explanation = self.generator.generate_explanation(
            self.mock_predictions,
            self.mock_features,
            self.mock_predictions[0]
        )
        
        self.assertIn('main_explanation', explanation)
        self.assertIn('reasoning_chain', explanation)
        self.assertIn('key_factors', explanation)
        self.assertIsInstance(explanation['main_explanation'], str)
    
    def test_simple_explanation(self):
        """Test simple explanation generation"""
        explanation = self.generator.generate_simple_explanation(self.mock_predictions[0])
        
        self.assertIsInstance(explanation, str)
        self.assertIn('Checkout', explanation)


def run_integration_test():
    """Run a simple integration test"""
    print("Running integration test...")
    
    # Create test image
    test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (300, 400), (500, 450), (255, 107, 53), -1)
    cv2.putText(test_image, 'Checkout', (320, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, test_image)
        test_image_path = tmp.name
    
    try:
        # Initialize predictor
        predictor = NextClickPredictor()
        
        # Run prediction
        result = predictor.predict_next_click(
            test_image_path,
            {"age_group": "25-34", "tech_savviness": "high", "mood": "focused"},
            "Complete purchase"
        )
        
        print(f"Integration test successful!")
        print(f"Top prediction: {result.top_prediction.get('element_text', 'N/A')}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
    except Exception as e:
        print(f"Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(exit=False, verbosity=2)
    
    # Run integration test
    print("\n" + "="*50)
    run_integration_test()