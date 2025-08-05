import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# Import improved components
from improved_ui_detector import ImprovedUIDetector, DetectedElement
from clean_feature_integration import CleanFeatureIntegrator, CleanFeatures
from ensemble_predictor import EnsemblePredictor, EnsemblePrediction
from evaluation_framework import EvaluationFramework, PredictionResult


@dataclass
class ImprovedPredictionResult:
    """Enhanced prediction result with comprehensive information"""
    # Core prediction
    top_prediction: Dict[str, Any]
    all_predictions: List[Dict[str, Any]]
    
    # Processing information
    ui_elements_detected: int
    processing_time: float
    detection_method: str
    
    # Ensemble details
    ensemble_prediction: EnsemblePrediction
    
    # Explanations
    explanation: Dict[str, Any]
    confidence_breakdown: Dict[str, float]
    
    # Quality metrics
    prediction_quality: Dict[str, float]
    feature_quality: Dict[str, float]
    
    # Metadata
    metadata: Dict[str, Any]


class ImprovedNextClickPredictor:
    """
    Improved next-click prediction system with enhanced accuracy and explainability
    
    Key improvements:
    - Simplified, robust UI detection
    - Clean feature integration with validation
    - Ensemble prediction (Bayesian + Gradient Boosting)
    - Comprehensive evaluation framework
    - Better error handling and fallbacks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the improved prediction system
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize core components
        self.ui_detector = ImprovedUIDetector()
        self.feature_integrator = CleanFeatureIntegrator()
        self.ensemble_predictor = EnsemblePredictor(self.config.get('ensemble_config'))
        
        # Initialize evaluation framework
        if self.config.get('enable_evaluation', True):
            self.evaluator = EvaluationFramework(
                output_dir=self.config.get('evaluation_output_dir', 'evaluation_results')
            )
        else:
            self.evaluator = None
        
        # Setup logging
        self._setup_logging()
        
        # System state
        self.is_initialized = False
        self.prediction_history = []
        self.performance_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0
        }
        
        self.logger.info("ImprovedNextClickPredictor created")
    
    def initialize(self, training_data: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Initialize the prediction system
        
        Args:
            training_data: Optional training data for the ensemble models
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing improved prediction system...")
            
            # Initialize ensemble predictor
            ensemble_success = self.ensemble_predictor.initialize(training_data)
            
            if not ensemble_success:
                self.logger.warning("Ensemble initialization failed, using fallback mode")
            
            self.is_initialized = True
            self.logger.info("Improved prediction system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def predict_next_click(self, 
                          screenshot_path: str,
                          user_attributes: Dict[str, Any],
                          task_description: str,
                          return_detailed: bool = True) -> ImprovedPredictionResult:
        """
        Main prediction pipeline with comprehensive analysis
        
        Args:
            screenshot_path: Path to PNG screenshot
            user_attributes: User profile data
            task_description: Task description string
            return_detailed: Whether to return detailed explanations
            
        Returns:
            ImprovedPredictionResult with comprehensive prediction information
        """
        start_time = time.time()
        
        if not self.is_initialized:
            self.logger.warning("System not initialized, attempting auto-initialization")
            if not self.initialize():
                return self._create_error_result("System initialization failed", start_time)
        
        try:
            self.logger.info(f"Starting improved prediction for: {screenshot_path}")
            
            # Step 1: Enhanced UI element detection
            self.logger.debug("Step 1: UI element detection")
            ui_elements = self._detect_ui_elements(screenshot_path)
            
            if not ui_elements:
                return self._handle_no_elements(screenshot_path, start_time)
            
            self.logger.info(f"Detected {len(ui_elements)} UI elements")
            
            # Step 2: Clean feature integration with validation
            self.logger.debug("Step 2: Feature integration")
            element_predictions = []
            
            for element in ui_elements:
                try:
                    # Convert DetectedElement to dictionary format
                    element_features = self._convert_detected_element(element)
                    
                    # Create clean features
                    clean_features = self.feature_integrator.integrate_features(
                        element_features, user_attributes, {'task_description': task_description}
                    )
                    
                    # Validate features
                    is_valid, issues = self.feature_integrator.validate_features(clean_features)
                    
                    if not is_valid:
                        self.logger.debug(f"Feature validation issues for {element.element_id}: {issues}")
                    
                    # Get ensemble prediction
                    ensemble_pred = self.ensemble_predictor.predict(
                        element_features, user_attributes, {'task_description': task_description}
                    )
                    
                    # Create comprehensive prediction entry
                    prediction_entry = {
                        'element_id': element.element_id,
                        'element_type': element.element_type,
                        'element_text': element.text,
                        'click_probability': ensemble_pred.final_click_probability,
                        'confidence': ensemble_pred.final_confidence,
                        'bbox': element.bbox,
                        'center': element.center,
                        'size': element.size,
                        'clean_features': clean_features,
                        'ensemble_details': ensemble_pred,
                        'feature_issues': issues,
                        'feature_valid': is_valid
                    }
                    
                    element_predictions.append(prediction_entry)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process element {element.element_id}: {e}")
                    continue
            
            if not element_predictions:
                return self._handle_no_predictions(screenshot_path, ui_elements, start_time)
            
            # Step 3: Rank and select top predictions
            element_predictions.sort(key=lambda x: x['click_probability'], reverse=True)
            
            # Step 4: Generate comprehensive explanations
            self.logger.debug("Step 4: Explanation generation")
            top_prediction = element_predictions[0]
            
            explanation = self._generate_comprehensive_explanation(
                top_prediction, element_predictions, user_attributes, task_description
            ) if return_detailed else {'simple_explanation': 'Detailed explanation disabled'}
            
            # Step 5: Calculate quality metrics
            prediction_quality = self._calculate_prediction_quality(element_predictions)
            feature_quality = self._calculate_feature_quality(element_predictions)
            
            # Step 6: Create result
            processing_time = time.time() - start_time
            
            result = ImprovedPredictionResult(
                top_prediction={
                    'element_id': top_prediction['element_id'],
                    'element_type': top_prediction['element_type'],
                    'element_text': top_prediction['element_text'],
                    'click_probability': top_prediction['click_probability'],
                    'confidence': top_prediction['confidence'],
                    'bbox': top_prediction['bbox'],
                    'center': top_prediction['center']
                },
                all_predictions=[
                    {
                        'element_id': p['element_id'],
                        'element_type': p['element_type'],
                        'element_text': p['element_text'],
                        'click_probability': p['click_probability'],
                        'confidence': p['confidence'],
                        'rank': i + 1
                    }
                    for i, p in enumerate(element_predictions[:10])  # Top 10
                ],
                ui_elements_detected=len(ui_elements),
                processing_time=processing_time,
                detection_method='improved_multi_method',
                ensemble_prediction=top_prediction['ensemble_details'],
                explanation=explanation,
                confidence_breakdown=self._calculate_confidence_breakdown(top_prediction),
                prediction_quality=prediction_quality,
                feature_quality=feature_quality,
                metadata=self._create_metadata(screenshot_path, user_attributes, task_description)
            )
            
            # Step 7: Update performance stats and history
            self._update_performance_stats(result)
            self.prediction_history.append(result)
            
            # Step 8: Add to evaluation framework if enabled
            if self.evaluator:
                self._add_to_evaluation(result, element_predictions)
            
            self.logger.info(f"Prediction completed in {processing_time:.3f}s")
            self.logger.info(f"Top prediction: {top_prediction['element_text']} "
                           f"({top_prediction['click_probability']:.1%})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction pipeline failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    def _detect_ui_elements(self, screenshot_path: str) -> List[DetectedElement]:
        """Detect UI elements using improved detector"""
        
        if not os.path.exists(screenshot_path):
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
        
        try:
            # Load image
            import cv2
            image = cv2.imread(screenshot_path)
            
            if image is None:
                raise ValueError(f"Could not load image from {screenshot_path}")
            
            # Run improved detection
            elements = self.ui_detector.detect_elements(image)
            
            self.logger.debug(f"UI detector found {len(elements)} elements")
            return elements
            
        except Exception as e:
            self.logger.error(f"UI detection failed: {e}")
            return []
    
    def _convert_detected_element(self, element: DetectedElement) -> Dict[str, Any]:
        """Convert DetectedElement to dictionary format for feature integration"""
        
        return {
            'id': element.element_id,
            'element_type': element.element_type,
            'text': element.text,
            'bbox': element.bbox,
            'center': element.center,
            'size': element.size,
            'prominence': element.confidence,  # Use detection confidence as prominence
            'confidence': element.confidence,
            'position_features': {
                'relative_x': element.center[0] / 1920 if element.center[0] < 5000 else 0.5,  # Rough normalization
                'relative_y': element.center[1] / 1080 if element.center[1] < 5000 else 0.5,
                'center_distance': min(1.0, 
                    ((element.center[0] - 960)**2 + (element.center[1] - 540)**2)**0.5 / 1000)
            },
            'color_features': element.visual_features.get('color_features', {}),
            'visual_features': element.visual_features
        }
    
    def _generate_comprehensive_explanation(self, top_prediction: Dict[str, Any],
                                          all_predictions: List[Dict[str, Any]],
                                          user_attributes: Dict[str, Any],
                                          task_description: str) -> Dict[str, Any]:
        """Generate comprehensive explanation combining all information sources"""
        
        ensemble_pred = top_prediction['ensemble_details']
        clean_features = top_prediction['clean_features']
        
        explanation = {
            # Primary explanation from ensemble
            'primary_reasoning': ensemble_pred.explanation_summary.get('primary_reasoning', []),
            
            # Element-specific factors
            'element_analysis': {
                'type': f"This is a {top_prediction['element_type']} element",
                'prominence': f"Visual prominence: {clean_features.prominence:.1%}",
                'position': f"Position: ({clean_features.relative_x:.2f}, {clean_features.relative_y:.2f})",
                'text_content': f"Text: '{top_prediction['element_text']}'" if top_prediction['element_text'] else "No text content",
                'size': f"Size category: {self._categorize_size(clean_features.area_normalized)}"
            },
            
            # User context explanation
            'user_context_analysis': {
                'tech_expertise': f"User tech expertise: {self._describe_tech_level(clean_features.tech_savviness_score)}",
                'intent': f"User intent: {'goal-directed' if clean_features.intent_score > 0.6 else 'exploring'}",
                'compatibility': f"User-element compatibility: {clean_features.user_element_compatibility:.1%}"
            },
            
            # Task context explanation
            'task_context_analysis': {
                'urgency': f"Task urgency: {self._describe_urgency(clean_features.task_urgency)}",
                'task_type': f"Task type score: {clean_features.task_type_score:.1%}",
                'alignment': f"Task-element alignment: {clean_features.task_element_alignment:.1%}"
            },
            
            # Model agreement and confidence
            'model_analysis': ensemble_pred.explanation_summary,
            
            # Alternative predictions
            'alternatives': [
                {
                    'element': pred['element_text'] or pred['element_type'],
                    'probability': pred['click_probability'],
                    'why_lower': self._explain_lower_ranking(pred, top_prediction)
                }
                for pred in all_predictions[1:4]  # Top 3 alternatives
            ],
            
            # Quality assessment
            'prediction_confidence': {
                'overall_confidence': top_prediction['confidence'],
                'data_quality': clean_features.data_quality_score,
                'feature_completeness': clean_features.feature_completeness,
                'assessment': self._assess_prediction_confidence(top_prediction, clean_features)
            }
        }
        
        return explanation
    
    def _categorize_size(self, area_normalized: float) -> str:
        """Categorize element size"""
        if area_normalized < 0.2:
            return "small"
        elif area_normalized < 0.6:
            return "medium"
        else:
            return "large"
    
    def _describe_tech_level(self, tech_score: float) -> str:
        """Describe user tech expertise level"""
        if tech_score < 0.3:
            return "beginner"
        elif tech_score < 0.7:
            return "intermediate"
        else:
            return "advanced"
    
    def _describe_urgency(self, urgency_score: float) -> str:
        """Describe task urgency level"""
        if urgency_score < 0.4:
            return "low"
        elif urgency_score < 0.7:
            return "medium"
        else:
            return "high"
    
    def _explain_lower_ranking(self, lower_pred: Dict[str, Any], 
                              top_pred: Dict[str, Any]) -> str:
        """Explain why a prediction ranked lower"""
        
        prob_diff = top_pred['click_probability'] - lower_pred['click_probability']
        
        if prob_diff > 0.3:
            return "Significantly lower click probability"
        elif prob_diff > 0.1:
            return "Moderately lower click probability"
        else:
            return "Slightly lower click probability"
    
    def _assess_prediction_confidence(self, prediction: Dict[str, Any], 
                                    features: CleanFeatures) -> str:
        """Assess overall prediction confidence"""
        
        confidence = prediction['confidence']
        data_quality = features.data_quality_score
        
        if confidence > 0.8 and data_quality > 0.7:
            return "High confidence - strong prediction with good data quality"
        elif confidence > 0.6:
            return "Moderate confidence - reasonable prediction"
        elif data_quality < 0.5:
            return "Lower confidence due to data quality issues"
        else:
            return "Lower confidence - uncertain prediction"
    
    def _calculate_confidence_breakdown(self, prediction: Dict[str, Any]) -> Dict[str, float]:
        """Calculate breakdown of confidence factors"""
        
        ensemble_details = prediction['ensemble_details']
        clean_features = prediction['clean_features']
        
        return {
            'detection_confidence': prediction['confidence'],
            'ensemble_confidence': ensemble_details.final_confidence,
            'data_quality': clean_features.data_quality_score,
            'feature_completeness': clean_features.feature_completeness,
            'model_agreement': ensemble_details.confidence_factors.get('model_agreement', 0.5)
        }
    
    def _calculate_prediction_quality(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall prediction quality metrics"""
        
        if not predictions:
            return {'error': 'no_predictions'}
        
        # Analyze prediction distribution
        probabilities = [p['click_probability'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        return {
            'top_probability': max(probabilities),
            'probability_spread': max(probabilities) - min(probabilities),
            'avg_confidence': sum(confidences) / len(confidences),
            'confident_predictions': len([c for c in confidences if c > 0.7]),
            'prediction_diversity': len(set(p['element_type'] for p in predictions[:5])) / 5.0
        }
    
    def _calculate_feature_quality(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate feature quality metrics"""
        
        if not predictions:
            return {'error': 'no_predictions'}
        
        valid_features = [p for p in predictions if p['feature_valid']]
        
        data_qualities = [p['clean_features'].data_quality_score for p in predictions]
        completeness_scores = [p['clean_features'].feature_completeness for p in predictions]
        
        return {
            'avg_data_quality': sum(data_qualities) / len(data_qualities),
            'avg_completeness': sum(completeness_scores) / len(completeness_scores),
            'valid_feature_ratio': len(valid_features) / len(predictions),
            'high_quality_features': len([q for q in data_qualities if q > 0.7])
        }
    
    def _update_performance_stats(self, result: ImprovedPredictionResult):
        """Update system performance statistics"""
        
        self.performance_stats['total_predictions'] += 1
        
        if result.top_prediction['click_probability'] > 0:
            self.performance_stats['successful_predictions'] += 1
        
        # Update running averages
        n = self.performance_stats['total_predictions']
        old_avg_time = self.performance_stats['avg_processing_time']
        old_avg_conf = self.performance_stats['avg_confidence']
        
        self.performance_stats['avg_processing_time'] = (
            (old_avg_time * (n - 1) + result.processing_time) / n
        )
        
        self.performance_stats['avg_confidence'] = (
            (old_avg_conf * (n - 1) + result.ensemble_prediction.final_confidence) / n
        )
    
    def _add_to_evaluation(self, result: ImprovedPredictionResult, 
                          element_predictions: List[Dict[str, Any]]):
        """Add prediction results to evaluation framework"""
        
        if not self.evaluator:
            return
        
        # Convert to evaluation format (we don't have ground truth here)
        eval_predictions = []
        
        for pred in element_predictions:
            eval_pred = PredictionResult(
                element_id=pred['element_id'],
                predicted_probability=pred['click_probability'],
                predicted_class=pred['click_probability'] > 0.5,
                actual_class=False,  # Unknown without ground truth
                confidence=pred['confidence'],
                prediction_time=result.processing_time / len(element_predictions),
                element_features=self.feature_integrator.to_dict(pred['clean_features']),
                explanation_quality=pred['clean_features'].data_quality_score
            )
            eval_predictions.append(eval_pred)
        
        # Add to evaluator for future analysis
        self.evaluator.add_predictions('improved_ensemble', eval_predictions)
    
    def _handle_no_elements(self, screenshot_path: str, start_time: float) -> ImprovedPredictionResult:
        """Handle case when no UI elements are detected"""
        
        self.logger.warning(f"No UI elements detected in: {screenshot_path}")
        
        return ImprovedPredictionResult(
            top_prediction={
                'element_id': 'none',
                'element_type': 'none',
                'element_text': 'No elements detected',
                'click_probability': 0.0,
                'confidence': 0.0,
                'bbox': (0, 0, 0, 0),
                'center': (0, 0)
            },
            all_predictions=[],
            ui_elements_detected=0,
            processing_time=time.time() - start_time,
            detection_method='none',
            ensemble_prediction=None,
            explanation={'error': 'No UI elements were detected in the screenshot'},
            confidence_breakdown={'detection_confidence': 0.0},
            prediction_quality={'error': 'no_elements'},
            feature_quality={'error': 'no_elements'},
            metadata={'error': 'no_ui_elements_detected'}
        )
    
    def _handle_no_predictions(self, screenshot_path: str, ui_elements: List[DetectedElement], 
                              start_time: float) -> ImprovedPredictionResult:
        """Handle case when UI elements are detected but no predictions generated"""
        
        self.logger.warning(f"No predictions generated despite detecting {len(ui_elements)} elements")
        
        return ImprovedPredictionResult(
            top_prediction={
                'element_id': 'failed',
                'element_type': 'unknown',
                'element_text': 'Prediction failed',
                'click_probability': 0.0,
                'confidence': 0.0,
                'bbox': (0, 0, 0, 0),
                'center': (0, 0)
            },
            all_predictions=[],
            ui_elements_detected=len(ui_elements),
            processing_time=time.time() - start_time,
            detection_method='improved_multi_method',
            ensemble_prediction=None,
            explanation={'error': 'Failed to generate predictions for detected elements'},
            confidence_breakdown={'detection_confidence': 0.0},
            prediction_quality={'error': 'prediction_failed'},
            feature_quality={'error': 'prediction_failed'},
            metadata={'error': 'prediction_generation_failed', 'elements_detected': len(ui_elements)}
        )
    
    def _create_error_result(self, error_message: str, start_time: float) -> ImprovedPredictionResult:
        """Create error result when prediction pipeline fails"""
        
        return ImprovedPredictionResult(
            top_prediction={
                'element_id': 'error',
                'element_type': 'error',
                'element_text': 'System error',
                'click_probability': 0.0,
                'confidence': 0.0,
                'bbox': (0, 0, 0, 0),
                'center': (0, 0)
            },
            all_predictions=[],
            ui_elements_detected=0,
            processing_time=time.time() - start_time,
            detection_method='error',
            ensemble_prediction=None,
            explanation={'error': error_message},
            confidence_breakdown={'system_error': 1.0},
            prediction_quality={'error': 'system_error'},
            feature_quality={'error': 'system_error'},
            metadata={'error': error_message}
        )
    
    def _create_metadata(self, screenshot_path: str, user_attributes: Dict[str, Any], 
                        task_description: str) -> Dict[str, Any]:
        """Create metadata for the prediction"""
        
        return {
            'screenshot_path': screenshot_path,
            'screenshot_exists': os.path.exists(screenshot_path),
            'user_attributes': user_attributes,
            'task_description': task_description,
            'timestamp': time.time(),
            'system_version': '2.0.0-improved',
            'components': {
                'ui_detector': 'ImprovedUIDetector',
                'feature_integrator': 'CleanFeatureIntegrator',
                'predictor': 'EnsemblePredictor'
            },
            'config': self.config
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = self.performance_stats.copy()
        
        # Add ensemble stats if available
        if self.ensemble_predictor:
            ensemble_stats = self.ensemble_predictor.get_ensemble_stats()
            if 'error' not in ensemble_stats:
                stats['ensemble'] = ensemble_stats
        
        # Add recent prediction analysis
        if self.prediction_history:
            recent = self.prediction_history[-10:]  # Last 10 predictions
            
            stats['recent_analysis'] = {
                'predictions_count': len(recent),
                'avg_elements_detected': sum(r.ui_elements_detected for r in recent) / len(recent),
                'avg_top_probability': sum(r.top_prediction['click_probability'] for r in recent) / len(recent),
                'avg_processing_time': sum(r.processing_time for r in recent) / len(recent)
            }
        
        return stats
    
    def export_results(self, output_path: str) -> bool:
        """Export comprehensive results and analysis"""
        
        try:
            export_data = {
                'system_info': {
                    'version': '2.0.0-improved',
                    'config': self.config,
                    'is_initialized': self.is_initialized
                },
                'performance_stats': self.performance_stats,
                'prediction_history': [
                    {
                        'top_prediction': r.top_prediction,
                        'processing_time': r.processing_time,
                        'ui_elements_detected': r.ui_elements_detected,
                        'prediction_quality': r.prediction_quality,
                        'feature_quality': r.feature_quality
                    }
                    for r in self.prediction_history
                ],
                'system_stats': self.get_system_stats()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Results exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        log_file = self.config.get('log_file', 'improved_next_click_predictor.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        
        return {
            'log_level': 'INFO',
            'log_file': 'improved_next_click_predictor.log',
            'enable_evaluation': True,
            'evaluation_output_dir': 'evaluation_results',
            'max_predictions_history': 1000,
            'ensemble_config': {
                'ensemble_method': 'adaptive',
                'min_confidence_threshold': 0.3
            },
            'ui_detection_config': {
                'max_elements': 50,
                'min_confidence': 0.3
            },
            'feature_integration_config': {
                'validate_features': True,
                'require_min_quality': 0.3
            }
        }


def main():
    """Example usage of the ImprovedNextClickPredictor"""
    
    # Initialize predictor
    predictor = ImprovedNextClickPredictor()
    
    # Initialize system
    success = predictor.initialize()
    if not success:
        print("Failed to initialize prediction system")
        return
    
    # Example prediction
    try:
        result = predictor.predict_next_click(
            screenshot_path="/path/to/screenshot.png",  # Replace with actual path
            user_attributes={
                "age_group": "25-34",
                "tech_savviness": "high",
                "mood": "focused",
                "device_type": "desktop"
            },
            task_description="Complete purchase checkout process"
        )
        
        print(f"\n=== Prediction Results ===")
        print(f"Top prediction: {result.top_prediction['element_text']} ({result.top_prediction['element_type']})")
        print(f"Click probability: {result.top_prediction['click_probability']:.1%}")
        print(f"Confidence: {result.ensemble_prediction.final_confidence:.1%}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"UI elements detected: {result.ui_elements_detected}")
        
        print(f"\n=== Quality Metrics ===")
        for metric, value in result.prediction_quality.items():
            print(f"{metric}: {value}")
        
        print(f"\n=== Primary Reasoning ===")
        reasoning = result.explanation.get('primary_reasoning', [])
        for i, reason in enumerate(reasoning[:3], 1):
            print(f"{i}. {reason}")
        
        # Export results
        predictor.export_results("improved_prediction_results.json")
        
    except Exception as e:
        print(f"Example prediction failed: {e}")


if __name__ == "__main__":
    main()