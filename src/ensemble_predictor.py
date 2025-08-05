import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import json
from pathlib import Path

# Import our custom models
from explainable_bayesian_network import SimpleBayesianNetwork, BayesianPrediction
from gradient_boosting_predictor import GradientBoostingPredictor, GradientBoostingPrediction


@dataclass
class EnsemblePrediction:
    """Combined prediction result from ensemble of models"""
    element_id: str
    final_click_probability: float
    final_confidence: float
    
    # Individual model predictions
    bayesian_prediction: BayesianPrediction
    gradient_boosting_prediction: GradientBoostingPrediction
    
    # Ensemble details
    model_weights: Dict[str, float]
    ensemble_method: str
    
    # Combined explanations
    explanation_summary: Dict[str, Any]
    confidence_factors: Dict[str, float]


class EnsemblePredictor:
    """
    Ensemble predictor combining explainable Bayesian network with gradient boosting
    Provides both high accuracy and explainability
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize component models
        self.bayesian_network = SimpleBayesianNetwork()
        self.gradient_boosting = GradientBoostingPredictor()
        
        # Model weights (can be learned or manually set)
        self.model_weights = {
            'bayesian': 0.4,      # Lower weight but provides explanations
            'gradient_boosting': 0.6  # Higher weight for accuracy
        }
        
        # Ensemble method
        self.ensemble_method = self.config.get('ensemble_method', 'weighted_average')
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.prediction_history = []
        self.is_initialized = False
    
    def initialize(self, training_data: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Initialize the ensemble system
        
        Args:
            training_data: Optional training data for gradient boosting
            
        Returns:
            True if initialization successful
        """
        try:
            # Initialize Bayesian network
            bayesian_success = self.bayesian_network.build_network(training_data)
            if not bayesian_success:
                self.logger.warning("Bayesian network initialization failed")
            
            # Initialize gradient boosting if training data available
            gb_success = True
            if training_data and len(training_data) > 10:
                metrics = self.gradient_boosting.train(training_data)
                if 'error' in metrics:
                    self.logger.warning(f"Gradient boosting training failed: {metrics['error']}")
                    gb_success = False
                else:
                    self.logger.info(f"Gradient boosting trained with accuracy: {metrics.get('accuracy', 'N/A')}")
            else:
                # Try to load pre-trained model
                gb_success = self.gradient_boosting.load_model()
                if not gb_success:
                    self.logger.info("No pre-trained gradient boosting model found, using Bayesian only")
            
            # Adjust weights based on available models
            if bayesian_success and gb_success:
                self.model_weights = {'bayesian': 0.4, 'gradient_boosting': 0.6}
            elif bayesian_success:
                self.model_weights = {'bayesian': 1.0, 'gradient_boosting': 0.0}
            elif gb_success:
                self.model_weights = {'bayesian': 0.0, 'gradient_boosting': 1.0}
            else:
                self.logger.error("Both models failed to initialize")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Ensemble initialized with weights: {self.model_weights}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ensemble initialization failed: {e}")
            return False
    
    def predict(self, element_features: Dict[str, Any], 
               user_context: Dict[str, Any],
               task_context: Dict[str, Any]) -> EnsemblePrediction:
        """
        Make ensemble prediction combining multiple models
        
        Args:
            element_features: Features of the UI element
            user_context: User characteristics and context
            task_context: Task-related context
            
        Returns:
            EnsemblePrediction with combined results and explanations
        """
        if not self.is_initialized:
            self.logger.warning("Ensemble not initialized, using fallback")
            return self._create_fallback_prediction(element_features)
        
        try:
            # Get predictions from individual models
            bayesian_pred = None
            gb_pred = None
            
            if self.model_weights.get('bayesian', 0) > 0:
                bayesian_pred = self.bayesian_network.predict_click(
                    element_features, user_context, task_context
                )
            
            if self.model_weights.get('gradient_boosting', 0) > 0:
                gb_pred = self.gradient_boosting.predict(
                    element_features, user_context, task_context
                )
            
            # Combine predictions
            ensemble_result = self._combine_predictions(
                bayesian_pred, gb_pred, element_features
            )
            
            # Track prediction
            self.prediction_history.append({
                'element_id': element_features.get('id', 'unknown'),
                'final_probability': ensemble_result.final_click_probability,
                'bayesian_prob': bayesian_pred.click_probability if bayesian_pred else None,
                'gb_prob': gb_pred.click_probability if gb_pred else None,
                'weights': self.model_weights.copy()
            })
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return self._create_fallback_prediction(element_features)
    
    def _combine_predictions(self, bayesian_pred: Optional[BayesianPrediction],
                           gb_pred: Optional[GradientBoostingPrediction],
                           element_features: Dict[str, Any]) -> EnsemblePrediction:
        """Combine predictions from individual models"""
        
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average_combination(bayesian_pred, gb_pred, element_features)
        elif self.ensemble_method == 'confidence_weighted':
            return self._confidence_weighted_combination(bayesian_pred, gb_pred, element_features)
        elif self.ensemble_method == 'adaptive':
            return self._adaptive_combination(bayesian_pred, gb_pred, element_features)
        else:
            return self._weighted_average_combination(bayesian_pred, gb_pred, element_features)
    
    def _weighted_average_combination(self, bayesian_pred: Optional[BayesianPrediction],
                                    gb_pred: Optional[GradientBoostingPrediction],
                                    element_features: Dict[str, Any]) -> EnsemblePrediction:
        """Combine using fixed weighted average"""
        
        probabilities = []
        confidences = []
        
        if bayesian_pred and self.model_weights.get('bayesian', 0) > 0:
            probabilities.append(bayesian_pred.click_probability * self.model_weights['bayesian'])
            confidences.append(bayesian_pred.confidence * self.model_weights['bayesian'])
        
        if gb_pred and self.model_weights.get('gradient_boosting', 0) > 0:
            probabilities.append(gb_pred.click_probability * self.model_weights['gradient_boosting'])
            confidences.append(gb_pred.confidence * self.model_weights['gradient_boosting'])
        
        final_probability = sum(probabilities) if probabilities else 0.5
        final_confidence = sum(confidences) if confidences else 0.3
        
        # Generate combined explanation
        explanation_summary = self._generate_explanation_summary(bayesian_pred, gb_pred)
        confidence_factors = self._analyze_confidence_factors(bayesian_pred, gb_pred)
        
        return EnsemblePrediction(
            element_id=element_features.get('id', 'unknown'),
            final_click_probability=final_probability,
            final_confidence=final_confidence,
            bayesian_prediction=bayesian_pred,
            gradient_boosting_prediction=gb_pred,
            model_weights=self.model_weights.copy(),
            ensemble_method='weighted_average',
            explanation_summary=explanation_summary,
            confidence_factors=confidence_factors
        )
    
    def _confidence_weighted_combination(self, bayesian_pred: Optional[BayesianPrediction],
                                       gb_pred: Optional[GradientBoostingPrediction],
                                       element_features: Dict[str, Any]) -> EnsemblePrediction:
        """Combine using confidence-based weighting"""
        
        total_confidence = 0
        weighted_probability = 0
        
        if bayesian_pred:
            total_confidence += bayesian_pred.confidence
            weighted_probability += bayesian_pred.click_probability * bayesian_pred.confidence
        
        if gb_pred:
            total_confidence += gb_pred.confidence
            weighted_probability += gb_pred.click_probability * gb_pred.confidence
        
        final_probability = weighted_probability / total_confidence if total_confidence > 0 else 0.5
        final_confidence = total_confidence / 2 if total_confidence > 0 else 0.3  # Average confidence
        
        # Dynamic weights based on confidence
        dynamic_weights = {}
        if bayesian_pred and gb_pred:
            bayesian_weight = bayesian_pred.confidence / total_confidence
            gb_weight = gb_pred.confidence / total_confidence
            dynamic_weights = {'bayesian': bayesian_weight, 'gradient_boosting': gb_weight}
        elif bayesian_pred:
            dynamic_weights = {'bayesian': 1.0, 'gradient_boosting': 0.0}
        elif gb_pred:
            dynamic_weights = {'bayesian': 0.0, 'gradient_boosting': 1.0}
        
        explanation_summary = self._generate_explanation_summary(bayesian_pred, gb_pred)
        confidence_factors = self._analyze_confidence_factors(bayesian_pred, gb_pred)
        
        return EnsemblePrediction(
            element_id=element_features.get('id', 'unknown'),
            final_click_probability=final_probability,
            final_confidence=final_confidence,
            bayesian_prediction=bayesian_pred,
            gradient_boosting_prediction=gb_pred,
            model_weights=dynamic_weights,
            ensemble_method='confidence_weighted',
            explanation_summary=explanation_summary,
            confidence_factors=confidence_factors
        )
    
    def _adaptive_combination(self, bayesian_pred: Optional[BayesianPrediction],
                            gb_pred: Optional[GradientBoostingPrediction],
                            element_features: Dict[str, Any]) -> EnsemblePrediction:
        """Adaptive combination based on element and context characteristics"""
        
        # Start with base weights
        adaptive_weights = self.model_weights.copy()
        
        # Adjust based on element type (Bayesian is better for reasoning about user behavior)
        element_type = element_features.get('element_type', 'text')
        if element_type in ['button', 'link']:  # Interactive elements
            adaptive_weights['bayesian'] = min(0.6, adaptive_weights['bayesian'] + 0.1)
            adaptive_weights['gradient_boosting'] = 1.0 - adaptive_weights['bayesian']
        
        # Adjust based on text presence (GB is better with text features)
        text = element_features.get('text', '')
        text_str = str(text) if text is not None else ''
        if len(text_str) > 5:  # Has meaningful text
            adaptive_weights['gradient_boosting'] = min(0.8, adaptive_weights['gradient_boosting'] + 0.1)
            adaptive_weights['bayesian'] = 1.0 - adaptive_weights['gradient_boosting']
        
        # Adjust based on prominence (Bayesian reasoning about visual attention)
        prominence = element_features.get('prominence', 0.5)
        try:
            prominence_float = float(prominence) if prominence is not None else 0.5
        except (ValueError, TypeError):
            prominence_float = 0.5
        if prominence_float > 0.7:  # High prominence
            adaptive_weights['bayesian'] = min(0.7, adaptive_weights['bayesian'] + 0.1)
            adaptive_weights['gradient_boosting'] = 1.0 - adaptive_weights['bayesian']
        
        # Combine with adaptive weights
        final_probability = 0.5
        final_confidence = 0.3
        
        if bayesian_pred and gb_pred:
            final_probability = (bayesian_pred.click_probability * adaptive_weights['bayesian'] +
                               gb_pred.click_probability * adaptive_weights['gradient_boosting'])
            final_confidence = (bayesian_pred.confidence * adaptive_weights['bayesian'] +
                              gb_pred.confidence * adaptive_weights['gradient_boosting'])
        elif bayesian_pred:
            final_probability = bayesian_pred.click_probability
            final_confidence = bayesian_pred.confidence
        elif gb_pred:
            final_probability = gb_pred.click_probability
            final_confidence = gb_pred.confidence
        
        explanation_summary = self._generate_explanation_summary(bayesian_pred, gb_pred)
        confidence_factors = self._analyze_confidence_factors(bayesian_pred, gb_pred)
        
        return EnsemblePrediction(
            element_id=element_features.get('id', 'unknown'),
            final_click_probability=final_probability,
            final_confidence=final_confidence,
            bayesian_prediction=bayesian_pred,
            gradient_boosting_prediction=gb_pred,
            model_weights=adaptive_weights,
            ensemble_method='adaptive',
            explanation_summary=explanation_summary,
            confidence_factors=confidence_factors
        )
    
    def _generate_explanation_summary(self, bayesian_pred: Optional[BayesianPrediction],
                                    gb_pred: Optional[GradientBoostingPrediction]) -> Dict[str, Any]:
        """Generate combined explanation from both models"""
        
        explanation = {
            'primary_reasoning': [],
            'supporting_factors': [],
            'confidence_assessment': '',
            'model_agreement': 'unknown'
        }
        
        # Extract Bayesian reasoning (primary for explainability)
        if bayesian_pred:
            explanation['primary_reasoning'] = bayesian_pred.reasoning_chain[:3]  # Top 3 reasons
            
            # Add key explanatory factors
            for comp in bayesian_pred.explanation_components[:3]:
                if abs(comp.influence_score) > 0.1:  # Only significant factors
                    explanation['supporting_factors'].append({
                        'factor': comp.factor_name,
                        'influence': comp.influence_score,
                        'explanation': comp.explanation_text
                    })
        
        # Add gradient boosting insights (feature importance)
        if gb_pred and gb_pred.top_features:
            gb_insights = []
            for feature_name, importance in gb_pred.top_features[:3]:
                if importance > 0.1:  # Only important features
                    gb_insights.append({
                        'feature': feature_name,
                        'importance': importance,
                        'explanation': self._interpret_gb_feature(feature_name, importance)
                    })
            
            if gb_insights:
                explanation['supporting_factors'].extend([
                    {'factor': 'data_driven_analysis', 'insights': gb_insights}
                ])
        
        # Assess model agreement
        if bayesian_pred and gb_pred:
            prob_diff = abs(bayesian_pred.click_probability - gb_pred.click_probability)
            if prob_diff < 0.2:
                explanation['model_agreement'] = 'high'
                explanation['confidence_assessment'] = 'Both reasoning and data analysis agree'
            elif prob_diff < 0.4:
                explanation['model_agreement'] = 'moderate'
                explanation['confidence_assessment'] = 'Models show some disagreement'
            else:
                explanation['model_agreement'] = 'low'
                explanation['confidence_assessment'] = 'Significant disagreement between models'
        elif bayesian_pred:
            explanation['confidence_assessment'] = 'Based on behavioral reasoning only'
        elif gb_pred:
            explanation['confidence_assessment'] = 'Based on data patterns only'
        
        return explanation
    
    def _interpret_gb_feature(self, feature_name: str, importance: float) -> str:
        """Interpret gradient boosting feature importance"""
        
        interpretations = {
            'is_button': 'Element type (button) strongly influences click likelihood',
            'is_link': 'Element type (link) affects click probability',
            'prominence': 'Visual prominence is a key factor',
            'center_distance': 'Distance from screen center affects attention',
            'has_action_words': 'Presence of action words influences user behavior',
            'text_length': 'Amount of text affects user engagement',
            'aspect_ratio': 'Element shape influences interaction likelihood',
            'tech_savviness_encoded': 'User technical expertise affects behavior',
            'urgency_level': 'Task urgency influences click patterns'
        }
        
        base_interpretation = interpretations.get(feature_name, f'{feature_name} influences the prediction')
        
        if importance > 0.3:
            return f"{base_interpretation} (strong effect)"
        elif importance > 0.1:
            return f"{base_interpretation} (moderate effect)"
        else:
            return f"{base_interpretation} (minor effect)"
    
    def _analyze_confidence_factors(self, bayesian_pred: Optional[BayesianPrediction],
                                  gb_pred: Optional[GradientBoostingPrediction]) -> Dict[str, float]:
        """Analyze factors that contribute to prediction confidence"""
        
        factors = {}
        
        if bayesian_pred:
            factors['reasoning_clarity'] = min(1.0, len(bayesian_pred.reasoning_chain) / 5.0)
            factors['factor_significance'] = np.mean([abs(comp.influence_score) 
                                                    for comp in bayesian_pred.explanation_components[:3]])
            factors['bayesian_confidence'] = bayesian_pred.confidence
        
        if gb_pred:
            factors['feature_importance'] = max([imp for _, imp in gb_pred.top_features[:3]]) if gb_pred.top_features else 0.0
            factors['gb_confidence'] = gb_pred.confidence
        
        # Model agreement factor
        if bayesian_pred and gb_pred:
            prob_agreement = 1.0 - abs(bayesian_pred.click_probability - gb_pred.click_probability)
            factors['model_agreement'] = prob_agreement
        
        return factors
    
    def _create_fallback_prediction(self, element_features: Dict[str, Any]) -> EnsemblePrediction:
        """Create fallback prediction when ensemble fails"""
        
        fallback_bayesian = BayesianPrediction(
            element_id=element_features.get('id', 'unknown'),
            click_probability=0.5,
            confidence=0.2,
            explanation_components=[],
            reasoning_chain=['Fallback prediction due to system error']
        )
        
        fallback_gb = GradientBoostingPrediction(
            element_id=element_features.get('id', 'unknown'),
            click_probability=0.5,
            confidence=0.2,
            feature_importance={},
            top_features=[]
        )
        
        return EnsemblePrediction(
            element_id=element_features.get('id', 'unknown'),
            final_click_probability=0.5,
            final_confidence=0.2,
            bayesian_prediction=fallback_bayesian,
            gradient_boosting_prediction=fallback_gb,
            model_weights={'bayesian': 0.5, 'gradient_boosting': 0.5},
            ensemble_method='fallback',
            explanation_summary={'error': 'Ensemble system failed'},
            confidence_factors={'system_error': 1.0}
        )
    
    def update_weights(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update ensemble weights based on performance feedback"""
        
        if not feedback_data:
            return self.model_weights
        
        try:
            bayesian_errors = []
            gb_errors = []
            
            for feedback in feedback_data:
                actual_clicked = feedback.get('actual_clicked', False)
                bayesian_prob = feedback.get('bayesian_probability')
                gb_prob = feedback.get('gb_probability')
                
                if bayesian_prob is not None:
                    bayesian_error = abs(bayesian_prob - (1.0 if actual_clicked else 0.0))
                    bayesian_errors.append(bayesian_error)
                
                if gb_prob is not None:
                    gb_error = abs(gb_prob - (1.0 if actual_clicked else 0.0))
                    gb_errors.append(gb_error)
            
            # Calculate average errors
            if bayesian_errors and gb_errors:
                avg_bayesian_error = np.mean(bayesian_errors)
                avg_gb_error = np.mean(gb_errors)
                
                # Assign weights inversely proportional to error
                total_inverse_error = (1.0 / (avg_bayesian_error + 0.1)) + (1.0 / (avg_gb_error + 0.1))
                
                new_bayesian_weight = (1.0 / (avg_bayesian_error + 0.1)) / total_inverse_error
                new_gb_weight = (1.0 / (avg_gb_error + 0.1)) / total_inverse_error
                
                # Smooth the weight update (don't change too drastically)
                self.model_weights['bayesian'] = (0.7 * self.model_weights['bayesian'] + 
                                                0.3 * new_bayesian_weight)
                self.model_weights['gradient_boosting'] = (0.7 * self.model_weights['gradient_boosting'] + 
                                                         0.3 * new_gb_weight)
                
                self.logger.info(f"Updated ensemble weights based on {len(feedback_data)} feedback samples")
            
            return self.model_weights
            
        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
            return self.model_weights
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble performance statistics"""
        
        if not self.prediction_history:
            return {'error': 'No predictions made yet'}
        
        recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
        
        stats = {
            'total_predictions': len(self.prediction_history),
            'recent_predictions': len(recent_predictions),
            'current_weights': self.model_weights.copy(),
            'ensemble_method': self.ensemble_method
        }
        
        # Analyze recent predictions
        final_probs = [p['final_probability'] for p in recent_predictions]
        stats['avg_probability'] = np.mean(final_probs)
        stats['probability_std'] = np.std(final_probs)
        
        # Model agreement analysis
        agreements = []
        for pred in recent_predictions:
            if pred['bayesian_prob'] is not None and pred['gb_prob'] is not None:
                agreement = 1.0 - abs(pred['bayesian_prob'] - pred['gb_prob'])
                agreements.append(agreement)
        
        if agreements:
            stats['avg_model_agreement'] = np.mean(agreements)
            stats['high_agreement_predictions'] = len([a for a in agreements if a > 0.8])
        
        return stats
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ensemble"""
        return {
            'ensemble_method': 'adaptive',  # weighted_average, confidence_weighted, adaptive
            'min_confidence_threshold': 0.3,
            'max_predictions_history': 1000,
            'weight_update_frequency': 50,  # Update weights every N predictions
            'explanation_detail_level': 'detailed'
        }
    
    def export_predictions(self, output_path: str) -> bool:
        """Export prediction history and explanations"""
        
        try:
            export_data = {
                'ensemble_config': self.config,
                'model_weights': self.model_weights,
                'prediction_history': self.prediction_history,
                'ensemble_stats': self.get_ensemble_stats()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Predictions exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False