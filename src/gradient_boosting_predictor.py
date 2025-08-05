import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import pickle
import os

# Optional imports with error handling
try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Gradient boosting functionality will be limited.")


@dataclass
class GradientBoostingPrediction:
    """Prediction result from gradient boosting model"""
    element_id: str
    click_probability: float
    confidence: float
    feature_importance: Dict[str, float]
    top_features: List[Tuple[str, float]]


class FeatureExtractor:
    """Extract and engineer features for gradient boosting"""
    
    def __init__(self):
        self.feature_names = []
        self.label_encoders = {}
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
    def extract_features(self, element_features: Dict[str, Any], 
                        user_context: Dict[str, Any],
                        task_context: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from input data"""
        
        features = []
        feature_names = []
        
        # Element-based features
        element_type = element_features.get('element_type', 'text')
        features.extend(self._encode_element_type(element_type))
        feature_names.extend(['is_button', 'is_link', 'is_form', 'is_text'])
        
        # Element properties
        features.append(element_features.get('prominence', 0.5))
        feature_names.append('prominence')
        
        features.append(element_features.get('confidence', 0.5))
        feature_names.append('detection_confidence')
        
        # Size features
        size = element_features.get('size', [0, 0])
        if len(size) >= 2:
            area = size[0] * size[1]
            aspect_ratio = size[0] / size[1] if size[1] > 0 else 1.0
        else:
            area = 0
            aspect_ratio = 1.0
            
        features.append(np.log1p(area))  # Log transform for area
        features.append(aspect_ratio)
        feature_names.extend(['log_area', 'aspect_ratio'])
        
        # Position features
        position_features = element_features.get('position_features', {})
        features.append(position_features.get('relative_x', 0.5))
        features.append(position_features.get('relative_y', 0.5))
        features.append(position_features.get('center_distance', 0.5))
        feature_names.extend(['relative_x', 'relative_y', 'center_distance'])
        
        # Color features
        color_features = element_features.get('color_features', {})
        features.append(color_features.get('contrast', 0.5))
        features.append(color_features.get('brightness', 128) / 255.0)
        feature_names.extend(['contrast', 'brightness'])
        
        # Text features
        text = element_features.get('text', '')
        text_features = self._extract_text_features(text)
        features.extend(text_features)
        feature_names.extend(['text_length', 'has_action_words', 'has_urgency_words', 
                            'word_count', 'has_numbers'])
        
        # User context features
        features.append(self._encode_categorical(user_context.get('tech_savviness', 'medium'), 
                                               'tech_savviness', ['low', 'medium', 'high']))
        features.append(self._encode_categorical(user_context.get('mood', 'neutral'), 
                                               'mood', ['negative', 'neutral', 'positive']))
        features.append(self._encode_categorical(user_context.get('device_type', 'desktop'), 
                                               'device_type', ['mobile', 'tablet', 'desktop']))
        feature_names.extend(['tech_savviness_encoded', 'mood_encoded', 'device_type_encoded'])
        
        # Task context features
        features.append(self._encode_categorical(task_context.get('task_type', 'browse'), 
                                               'task_type', ['browse', 'search', 'purchase']))
        features.append(task_context.get('urgency_level', 0.5))
        features.append(task_context.get('completion_stage', 0.5))
        feature_names.extend(['task_type_encoded', 'urgency_level', 'completion_stage'])
        
        # Interaction features
        features.extend(self._calculate_interaction_features(
            element_features, user_context, task_context))
        feature_names.extend(['user_element_match', 'task_element_match', 'complexity_score'])
        
        # Store feature names for interpretation
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(features, dtype=np.float32)
    
    def _encode_element_type(self, element_type: str) -> List[float]:
        """One-hot encode element type"""
        types = ['button', 'link', 'form', 'text']
        encoding = [1.0 if element_type == t else 0.0 for t in types]
        return encoding
    
    def _encode_categorical(self, value: str, category: str, possible_values: List[str]) -> float:
        """Encode categorical value as normalized number"""
        try:
            return float(possible_values.index(value)) / (len(possible_values) - 1)
        except ValueError:
            return 0.5  # Default for unknown values
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract features from text content"""
        if not text:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        text_lower = text.lower()
        
        # Text length (normalized)
        text_length = min(len(text) / 50.0, 1.0)  # Cap at 50 characters
        
        # Action words
        action_words = ['click', 'buy', 'purchase', 'continue', 'next', 'submit', 
                       'login', 'signup', 'register', 'save', 'confirm']
        has_action_words = float(any(word in text_lower for word in action_words))
        
        # Urgency words
        urgency_words = ['now', 'today', 'limited', 'urgent', 'quick', 'fast', 'hurry']
        has_urgency_words = float(any(word in text_lower for word in urgency_words))
        
        # Word count (normalized)
        word_count = min(len(text.split()) / 10.0, 1.0)  # Cap at 10 words
        
        # Has numbers
        has_numbers = float(any(char.isdigit() for char in text))
        
        return [text_length, has_action_words, has_urgency_words, word_count, has_numbers]
    
    def _calculate_interaction_features(self, element_features: Dict[str, Any], 
                                      user_context: Dict[str, Any],
                                      task_context: Dict[str, Any]) -> List[float]:
        """Calculate interaction features between different contexts"""
        
        # User-element match
        element_type = element_features.get('element_type', 'text')
        tech_savviness = user_context.get('tech_savviness', 'medium')
        
        user_element_match = 0.5
        if tech_savviness == 'low' and element_type == 'button':
            user_element_match = 0.8  # Novices prefer buttons
        elif tech_savviness == 'high' and element_type in ['link', 'form']:
            user_element_match = 0.7  # Experts comfortable with complex elements
        
        # Task-element match
        task_type = task_context.get('task_type', 'browse')
        task_element_match = 0.5
        
        if task_type == 'purchase' and element_type == 'button':
            task_element_match = 0.9  # Purchase tasks favor buttons
        elif task_type == 'search' and element_type == 'form':
            task_element_match = 0.8  # Search tasks favor forms
        elif task_type == 'browse' and element_type == 'link':
            task_element_match = 0.7  # Browsing tasks favor links
        
        # Complexity score (how complex the interaction is)
        complexity_factors = [
            element_features.get('size', [0, 0])[0] * element_features.get('size', [0, 0])[1] / 10000,  # Size complexity
            1.0 if element_features.get('text', '') else 0.5,  # Text presence
            element_features.get('prominence', 0.5),  # Visual complexity
        ]
        complexity_score = np.mean(complexity_factors)
        
        return [user_element_match, task_element_match, complexity_score]


class GradientBoostingPredictor:
    """
    Gradient boosting model for accurate click prediction
    Complements the explainable Bayesian network with higher accuracy
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_extractor = FeatureExtractor()
        self.classifier = None
        self.regressor = None
        self.is_trained = False
        self.model_path = model_path or "gradient_boosting_model.pkl"
        self.logger = logging.getLogger(__name__)
        
        if SKLEARN_AVAILABLE:
            self._initialize_models()
        else:
            self.logger.warning("Gradient boosting not available without scikit-learn")
    
    def _initialize_models(self):
        """Initialize gradient boosting models"""
        # Classifier for click/no-click prediction
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            validation_fraction=0.2,
            n_iter_no_change=10,
            tol=1e-4
        )
        
        # Regressor for probability estimation
        self.regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            validation_fraction=0.2,
            n_iter_no_change=10,
            tol=1e-4
        )
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the gradient boosting model
        
        Args:
            training_data: List of training examples with features and labels
            
        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            self.logger.error("Cannot train without scikit-learn")
            return {'error': 'scikit-learn not available'}
        
        if not training_data:
            self.logger.error("No training data provided")
            return {'error': 'no training data'}
        
        try:
            # Extract features and labels
            X, y_class, y_prob = self._prepare_training_data(training_data)
            
            if len(X) < 10:
                self.logger.warning("Very small training set, model may not be reliable")
            
            # Split data
            X_train, X_test, y_class_train, y_class_test, y_prob_train, y_prob_test = \
                train_test_split(X, y_class, y_prob, test_size=0.2, random_state=42, stratify=y_class)
            
            # Scale features
            X_train_scaled = self.feature_extractor.scaler.fit_transform(X_train)
            X_test_scaled = self.feature_extractor.scaler.transform(X_test)
            
            # Train classifier
            self.classifier.fit(X_train_scaled, y_class_train)
            
            # Train regressor
            self.regressor.fit(X_train_scaled, y_prob_train)
            
            # Evaluate models
            y_class_pred = self.classifier.predict(X_test_scaled)
            y_prob_pred = self.regressor.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_class_test, y_class_pred),
                'precision': precision_score(y_class_test, y_class_pred, average='weighted'),
                'recall': recall_score(y_class_test, y_class_pred, average='weighted'),
                'f1_score': f1_score(y_class_test, y_class_pred, average='weighted'),
                'mse_probability': np.mean((y_prob_test - y_prob_pred) ** 2),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            self.is_trained = True
            self.logger.info(f"Training completed. Accuracy: {metrics['accuracy']:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data in the correct format"""
        X = []
        y_class = []  # Binary classification labels
        y_prob = []   # Probability regression targets
        
        for example in training_data:
            # Extract features
            features = self.feature_extractor.extract_features(
                example.get('element_features', {}),
                example.get('user_context', {}),
                example.get('task_context', {})
            )
            X.append(features)
            
            # Extract labels
            clicked = example.get('clicked', False)
            click_probability = example.get('click_probability', 0.5 if not clicked else 0.8)
            
            y_class.append(1 if clicked else 0)
            y_prob.append(click_probability)
        
        return np.array(X), np.array(y_class), np.array(y_prob)
    
    def predict(self, element_features: Dict[str, Any], 
               user_context: Dict[str, Any],
               task_context: Dict[str, Any]) -> GradientBoostingPrediction:
        """
        Make prediction using trained gradient boosting model
        
        Args:
            element_features: Features of the UI element
            user_context: User characteristics and context
            task_context: Task-related context
            
        Returns:
            GradientBoostingPrediction with probability and feature importance
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return self._create_fallback_prediction(element_features)
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(
                element_features, user_context, task_context
            )
            features_scaled = self.feature_extractor.scaler.transform([features])
            
            # Get probability prediction
            click_probability = self.regressor.predict(features_scaled)[0]
            click_probability = max(0.01, min(0.99, click_probability))  # Clamp to valid range
            
            # Get classification confidence
            class_probabilities = self.classifier.predict_proba(features_scaled)[0]
            confidence = max(class_probabilities)  # Confidence is max class probability
            
            # Get feature importance
            feature_importance = self._get_feature_importance(features)
            top_features = sorted(feature_importance.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:5]
            
            return GradientBoostingPrediction(
                element_id=element_features.get('id', 'unknown'),
                click_probability=float(click_probability),
                confidence=float(confidence),
                feature_importance=feature_importance,
                top_features=top_features
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._create_fallback_prediction(element_features)
    
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for the prediction"""
        if not self.is_trained or not self.feature_extractor.feature_names:
            return {}
        
        # Use regressor feature importance (generally more stable)
        importance_scores = self.regressor.feature_importances_
        
        # Combine with actual feature values for interpretation
        feature_importance = {}
        for i, (name, importance) in enumerate(zip(self.feature_extractor.feature_names, importance_scores)):
            # Scale importance by feature value for better interpretation
            scaled_importance = importance * abs(features[i]) if i < len(features) else importance
            feature_importance[name] = float(scaled_importance)
        
        return feature_importance
    
    def _create_fallback_prediction(self, element_features: Dict[str, Any]) -> GradientBoostingPrediction:
        """Create fallback prediction when model is not available"""
        return GradientBoostingPrediction(
            element_id=element_features.get('id', 'unknown'),
            click_probability=0.5,
            confidence=0.3,
            feature_importance={},
            top_features=[('fallback', 1.0)]
        )
    
    def save_model(self, path: Optional[str] = None) -> bool:
        """Save trained model to disk"""
        if not self.is_trained:
            self.logger.warning("Cannot save untrained model")
            return False
        
        save_path = path or self.model_path
        
        try:
            model_data = {
                'classifier': self.classifier,
                'regressor': self.regressor,
                'feature_extractor': self.feature_extractor,
                'is_trained': self.is_trained
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """Load trained model from disk"""
        load_path = path or self.model_path
        
        if not os.path.exists(load_path):
            self.logger.warning(f"Model file not found: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.regressor = model_data['regressor']
            self.feature_extractor = model_data['feature_extractor']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Model loaded from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        info = {
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_extractor.feature_names),
            'feature_names': self.feature_extractor.feature_names
        }
        
        if SKLEARN_AVAILABLE and self.classifier:
            info.update({
                'classifier_n_estimators': self.classifier.n_estimators,
                'classifier_max_depth': self.classifier.max_depth,
                'regressor_n_estimators': self.regressor.n_estimators,
                'regressor_max_depth': self.regressor.max_depth
            })
        
        return info
    
    def cross_validate(self, training_data: List[Dict[str, Any]], cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on the model"""
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        try:
            X, y_class, y_prob = self._prepare_training_data(training_data)
            X_scaled = self.feature_extractor.scaler.fit_transform(X)
            
            # Cross-validate classifier
            cv_scores_class = cross_val_score(self.classifier, X_scaled, y_class, 
                                            cv=cv_folds, scoring='f1_weighted')
            
            # Cross-validate regressor (using negative MSE)
            cv_scores_reg = cross_val_score(self.regressor, X_scaled, y_prob, 
                                          cv=cv_folds, scoring='neg_mean_squared_error')
            
            return {
                'classification_f1_mean': float(np.mean(cv_scores_class)),
                'classification_f1_std': float(np.std(cv_scores_class)),
                'regression_mse_mean': float(-np.mean(cv_scores_reg)),
                'regression_mse_std': float(np.std(cv_scores_reg)),
                'cv_folds': cv_folds,
                'sample_count': len(X)
            }
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {'error': str(e)}