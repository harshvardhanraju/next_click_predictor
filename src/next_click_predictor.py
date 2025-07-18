import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

from screenshot_processor import ScreenshotProcessor
from feature_integrator import FeatureIntegrator
from bayesian_network import BayesianNetworkEngine
from explanation_generator import ExplanationGenerator


@dataclass
class PredictionResult:
    """Complete prediction result with all components"""
    top_prediction: Dict[str, Any]
    all_predictions: List[Dict[str, Any]]
    explanation: Dict[str, Any]
    ui_elements: List[Dict[str, Any]]
    processing_time: float
    confidence_score: float
    metadata: Dict[str, Any]


class NextClickPredictor:
    """
    Main orchestrator for the next-click prediction system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prediction system
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.screenshot_processor = ScreenshotProcessor()
        self.feature_integrator = FeatureIntegrator()
        self.bayesian_engine = BayesianNetworkEngine()
        self.explanation_generator = ExplanationGenerator()
        
        # Setup logging
        self._setup_logging()
        
        # Performance tracking
        self.prediction_history = []
        
        self.logger.info("NextClickPredictor initialized successfully")
    
    def predict_next_click(self, 
                          screenshot_path: str,
                          user_attributes: Dict[str, Any],
                          task_description: str,
                          return_detailed: bool = True) -> PredictionResult:
        """
        Main prediction pipeline
        
        Args:
            screenshot_path: Path to PNG screenshot
            user_attributes: User profile data
            task_description: Task description string
            return_detailed: Whether to return detailed explanation
            
        Returns:
            PredictionResult with prediction and explanation
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting prediction for screenshot: {screenshot_path}")
            
            # Step 1: Process screenshot
            self.logger.info("Processing screenshot...")
            screenshot_data = self.screenshot_processor.process_screenshot(screenshot_path)
            ui_elements = screenshot_data['elements']
            
            if not ui_elements:
                return self._handle_no_elements_found(screenshot_path, start_time)
            
            self.logger.info(f"Found {len(ui_elements)} UI elements")
            
            # Step 2: Integrate features
            self.logger.info("Integrating features...")
            integrated_features = self.feature_integrator.integrate_features(
                user_attributes, ui_elements, task_description
            )
            
            # Step 3: Build Bayesian network
            self.logger.info("Building Bayesian network...")
            network = self.bayesian_engine.build_network(integrated_features)
            
            # Step 4: Run inference
            self.logger.info("Running inference...")
            predictions = self.bayesian_engine.predict_clicks(integrated_features)
            
            if not predictions:
                return self._handle_no_predictions(screenshot_path, ui_elements, start_time)
            
            # Step 5: Generate explanations
            top_prediction = predictions[0]
            explanation = {}
            
            if return_detailed:
                self.logger.info("Generating explanations...")
                explanation = self.explanation_generator.generate_explanation(
                    predictions, integrated_features, top_prediction
                )
            else:
                explanation = {
                    'simple_explanation': self.explanation_generator.generate_simple_explanation(top_prediction)
                }
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(predictions, integrated_features)
            
            # Create result
            processing_time = time.time() - start_time
            result = PredictionResult(
                top_prediction=top_prediction,
                all_predictions=predictions,
                explanation=explanation,
                ui_elements=ui_elements,
                processing_time=processing_time,
                confidence_score=confidence_score,
                metadata=self._create_metadata(screenshot_path, user_attributes, task_description)
            )
            
            # Log result
            self.logger.info(f"Prediction completed in {processing_time:.2f}s")
            self.logger.info(f"Top prediction: {top_prediction['element_text']} ({top_prediction['click_probability']:.2%})")
            
            # Store in history
            self.prediction_history.append({
                'timestamp': time.time(),
                'screenshot_path': screenshot_path,
                'top_prediction': top_prediction,
                'processing_time': processing_time,
                'confidence': confidence_score
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return self._handle_prediction_error(e, screenshot_path, start_time)
    
    def predict_batch(self, 
                     screenshot_paths: List[str],
                     user_attributes: Dict[str, Any],
                     task_descriptions: List[str]) -> List[PredictionResult]:
        """
        Predict clicks for multiple screenshots
        
        Args:
            screenshot_paths: List of screenshot paths
            user_attributes: User profile data
            task_descriptions: List of task descriptions
            
        Returns:
            List of PredictionResult objects
        """
        if len(screenshot_paths) != len(task_descriptions):
            raise ValueError("Number of screenshots must match number of task descriptions")
        
        results = []
        for screenshot_path, task_description in zip(screenshot_paths, task_descriptions):
            try:
                result = self.predict_next_click(
                    screenshot_path, user_attributes, task_description
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch prediction failed for {screenshot_path}: {str(e)}")
                results.append(None)
        
        return results
    
    def analyze_screenshot_only(self, screenshot_path: str) -> Dict[str, Any]:
        """
        Analyze screenshot without prediction (for debugging)
        
        Args:
            screenshot_path: Path to PNG screenshot
            
        Returns:
            Screenshot analysis results
        """
        try:
            screenshot_data = self.screenshot_processor.process_screenshot(screenshot_path)
            return {
                'ui_elements': screenshot_data['elements'],
                'screen_dimensions': screenshot_data['screen_dimensions'],
                'total_elements': screenshot_data['total_elements'],
                'element_types': self._analyze_element_types(screenshot_data['elements']),
                'prominence_distribution': self._analyze_prominence_distribution(screenshot_data['elements'])
            }
        except Exception as e:
            self.logger.error(f"Screenshot analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def get_prediction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent prediction history"""
        return self.prediction_history[-limit:]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        if not self.prediction_history:
            return {'error': 'No predictions made yet'}
        
        processing_times = [p['processing_time'] for p in self.prediction_history]
        confidences = [p['confidence'] for p in self.prediction_history]
        
        return {
            'total_predictions': len(self.prediction_history),
            'avg_processing_time': sum(processing_times) / len(processing_times),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_processing_time': min(processing_times),
            'max_processing_time': max(processing_times),
            'recent_predictions': len([p for p in self.prediction_history if time.time() - p['timestamp'] < 3600])
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'log_level': 'INFO',
            'max_elements': 50,
            'min_confidence_threshold': 0.1,
            'explanation_detail_level': 'detailed',
            'cache_networks': True,
            'performance_tracking': True
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('next_click_predictor.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _handle_no_elements_found(self, screenshot_path: str, start_time: float) -> PredictionResult:
        """Handle case when no UI elements are found"""
        self.logger.warning(f"No UI elements found in screenshot: {screenshot_path}")
        
        return PredictionResult(
            top_prediction={
                'element_id': 'none',
                'element_type': 'none',
                'element_text': 'No elements found',
                'click_probability': 0.0,
                'confidence': 0.0,
                'rank': 0
            },
            all_predictions=[],
            explanation={
                'main_explanation': 'No clickable elements were detected in the screenshot.',
                'error': 'No UI elements found'
            },
            ui_elements=[],
            processing_time=time.time() - start_time,
            confidence_score=0.0,
            metadata={'error': 'No UI elements detected'}
        )
    
    def _handle_no_predictions(self, screenshot_path: str, ui_elements: List[Dict], start_time: float) -> PredictionResult:
        """Handle case when no predictions are generated"""
        self.logger.warning(f"No predictions generated for screenshot: {screenshot_path}")
        
        # Use fallback heuristic prediction
        fallback_prediction = self._generate_fallback_prediction(ui_elements)
        
        return PredictionResult(
            top_prediction=fallback_prediction,
            all_predictions=[fallback_prediction],
            explanation={
                'main_explanation': 'Using fallback heuristic prediction due to inference failure.',
                'warning': 'Bayesian inference failed'
            },
            ui_elements=ui_elements,
            processing_time=time.time() - start_time,
            confidence_score=0.3,
            metadata={'fallback_used': True}
        )
    
    def _handle_prediction_error(self, error: Exception, screenshot_path: str, start_time: float) -> PredictionResult:
        """Handle prediction errors"""
        self.logger.error(f"Prediction error for {screenshot_path}: {str(error)}")
        
        return PredictionResult(
            top_prediction={
                'element_id': 'error',
                'element_type': 'error',
                'element_text': 'Prediction failed',
                'click_probability': 0.0,
                'confidence': 0.0,
                'rank': 0
            },
            all_predictions=[],
            explanation={
                'error': str(error),
                'main_explanation': 'Prediction failed due to system error.'
            },
            ui_elements=[],
            processing_time=time.time() - start_time,
            confidence_score=0.0,
            metadata={'error': str(error)}
        )
    
    def _generate_fallback_prediction(self, ui_elements: List[Dict]) -> Dict[str, Any]:
        """Generate fallback prediction based on simple heuristics"""
        if not ui_elements:
            return {
                'element_id': 'none',
                'element_type': 'none',
                'element_text': 'No elements',
                'click_probability': 0.0,
                'confidence': 0.0,
                'rank': 0
            }
        
        # Sort by prominence and select highest
        sorted_elements = sorted(ui_elements, key=lambda x: x.get('prominence', 0), reverse=True)
        best_element = sorted_elements[0]
        
        return {
            'element_id': best_element['id'],
            'element_type': best_element['type'],
            'element_text': best_element['text'],
            'click_probability': best_element.get('prominence', 0.5),
            'confidence': 0.5,
            'rank': 1,
            'fallback_used': True
        }
    
    def _calculate_overall_confidence(self, predictions: List[Dict], integrated_features) -> float:
        """Calculate overall confidence score"""
        if not predictions:
            return 0.0
        
        top_prediction = predictions[0]
        
        # Base confidence from top prediction
        base_confidence = top_prediction.get('confidence', 0.5)
        
        # Adjust based on prediction spread
        if len(predictions) > 1:
            prob_spread = top_prediction['click_probability'] - predictions[1]['click_probability']
            spread_factor = min(1.0, prob_spread * 2)  # Higher spread = higher confidence
            base_confidence *= (0.8 + 0.2 * spread_factor)
        
        # Adjust based on feature quality
        feature_quality = integrated_features.interaction_features.get('overall_alignment', 0.5)
        base_confidence *= (0.7 + 0.3 * feature_quality)
        
        return min(1.0, base_confidence)
    
    def _create_metadata(self, screenshot_path: str, user_attributes: Dict, task_description: str) -> Dict[str, Any]:
        """Create metadata for the prediction"""
        return {
            'screenshot_path': screenshot_path,
            'screenshot_exists': os.path.exists(screenshot_path),
            'user_attributes': user_attributes,
            'task_description': task_description,
            'timestamp': time.time(),
            'system_version': '1.0.0',
            'components': {
                'screenshot_processor': 'ScreenshotProcessor',
                'feature_integrator': 'FeatureIntegrator',
                'bayesian_engine': 'BayesianNetworkEngine',
                'explanation_generator': 'ExplanationGenerator'
            }
        }
    
    def _analyze_element_types(self, ui_elements: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of element types"""
        type_counts = {}
        for element in ui_elements:
            element_type = element.get('type', 'unknown')
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
        return type_counts
    
    def _analyze_prominence_distribution(self, ui_elements: List[Dict]) -> Dict[str, float]:
        """Analyze prominence distribution"""
        if not ui_elements:
            return {}
        
        prominences = [element.get('prominence', 0) for element in ui_elements]
        
        return {
            'mean': sum(prominences) / len(prominences),
            'min': min(prominences),
            'max': max(prominences),
            'high_prominence_count': len([p for p in prominences if p > 0.7])
        }
    
    def save_prediction_result(self, result: PredictionResult, output_path: str):
        """Save prediction result to JSON file"""
        try:
            # Convert result to serializable format
            result_dict = {
                'top_prediction': result.top_prediction,
                'all_predictions': result.all_predictions,
                'explanation': result.explanation,
                'ui_elements': result.ui_elements,
                'processing_time': result.processing_time,
                'confidence_score': result.confidence_score,
                'metadata': result.metadata
            }
            
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            self.logger.info(f"Prediction result saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save prediction result: {str(e)}")
    
    def load_prediction_result(self, input_path: str) -> Optional[PredictionResult]:
        """Load prediction result from JSON file"""
        try:
            with open(input_path, 'r') as f:
                result_dict = json.load(f)
            
            return PredictionResult(
                top_prediction=result_dict['top_prediction'],
                all_predictions=result_dict['all_predictions'],
                explanation=result_dict['explanation'],
                ui_elements=result_dict['ui_elements'],
                processing_time=result_dict['processing_time'],
                confidence_score=result_dict['confidence_score'],
                metadata=result_dict['metadata']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load prediction result: {str(e)}")
            return None


def main():
    """Example usage of the NextClickPredictor"""
    
    # Initialize predictor
    predictor = NextClickPredictor()
    
    # Example prediction
    try:
        result = predictor.predict_next_click(
            screenshot_path="/path/to/screenshot.png",
            user_attributes={
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused",
                "device_type": "desktop"
            },
            task_description="Complete purchase. What would you click next?"
        )
        
        print(f"Top prediction: {result.top_prediction['element_text']}")
        print(f"Probability: {result.top_prediction['click_probability']:.2%}")
        print(f"Confidence: {result.confidence_score:.2%}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Explanation: {result.explanation.get('main_explanation', 'No explanation')}")
        
    except Exception as e:
        print(f"Example failed: {str(e)}")


if __name__ == "__main__":
    main()