import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class CleanFeatures:
    """Clean, validated feature set for prediction models"""
    # Core element features
    element_id: str
    element_type: str
    prominence: float
    confidence: float
    
    # Position features
    relative_x: float
    relative_y: float
    center_distance: float
    
    # Size features
    area_normalized: float
    aspect_ratio: float
    
    # Text features
    has_text: bool
    text_length_normalized: float
    has_action_words: bool
    has_urgency_words: bool
    
    # Visual features
    contrast: float
    brightness_normalized: float
    
    # User context features
    tech_savviness_score: float
    intent_score: float  # 0 = exploring, 1 = goal-directed
    device_complexity: float
    
    # Task context features
    task_urgency: float
    task_type_score: float
    completion_stage: float
    
    # Interaction features
    user_element_compatibility: float
    task_element_alignment: float
    
    # Quality indicators
    feature_completeness: float  # 0-1, how many features are available
    data_quality_score: float   # 0-1, overall data quality


class CleanFeatureIntegrator:
    """
    Clean, robust feature integration focusing on reliability and interpretability
    Replaces the complex original feature integrator with a simpler, more maintainable approach
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Feature validation ranges
        self.feature_ranges = {
            'prominence': (0.0, 1.0),
            'confidence': (0.0, 1.0),
            'relative_x': (0.0, 1.0),
            'relative_y': (0.0, 1.0),
            'center_distance': (0.0, 1.0),
            'aspect_ratio': (0.1, 20.0),
            'contrast': (0.0, 1.0),
            'brightness_normalized': (0.0, 1.0),
            'tech_savviness_score': (0.0, 1.0),
            'intent_score': (0.0, 1.0),
            'device_complexity': (0.0, 1.0),
            'task_urgency': (0.0, 1.0),
            'task_type_score': (0.0, 1.0),
            'completion_stage': (0.0, 1.0)
        }
        
        # Action and urgency word lists
        self.action_words = {
            'click', 'buy', 'purchase', 'continue', 'next', 'submit', 
            'login', 'signup', 'register', 'save', 'confirm', 'apply',
            'start', 'begin', 'proceed', 'go', 'enter'
        }
        
        self.urgency_words = {
            'now', 'today', 'limited', 'urgent', 'quick', 'fast', 
            'hurry', 'immediate', 'asap', 'expires', 'deadline'
        }
        
        # Element type mappings
        self.element_type_scores = {
            'button': 0.9,
            'link': 0.7,
            'form': 0.6,
            'menu': 0.5,
            'text': 0.2,
            'image': 0.3,
            'unknown': 0.4
        }
        
        # Device complexity mappings
        self.device_complexity_scores = {
            'mobile': 0.3,
            'tablet': 0.5,
            'desktop': 0.7,
            'laptop': 0.8
        }
        
        # Task type mappings
        self.task_type_scores = {
            'browse': 0.2,
            'search': 0.4,
            'purchase': 0.8,
            'signup': 0.9,
            'urgent': 1.0
        }
    
    def integrate_features(self, element_features: Dict[str, Any], 
                          user_context: Dict[str, Any],
                          task_context: Dict[str, Any]) -> CleanFeatures:
        """
        Main integration method - converts raw features to clean, validated features
        
        Args:
            element_features: Raw UI element features
            user_context: User profile and context
            task_context: Task description and context
            
        Returns:
            CleanFeatures object with validated and normalized features
        """
        try:
            # Extract and validate core element features
            element_id = element_features.get('id', 'unknown')
            element_type = element_features.get('element_type', 'unknown')
            prominence = self._validate_range(element_features.get('prominence', 0.5), 'prominence')
            confidence = self._validate_range(element_features.get('confidence', 0.5), 'confidence')
            
            # Extract and validate position features
            position_features = element_features.get('position_features', {})
            relative_x = self._validate_range(position_features.get('relative_x', 0.5), 'relative_x')
            relative_y = self._validate_range(position_features.get('relative_y', 0.5), 'relative_y')
            center_distance = self._validate_range(position_features.get('center_distance', 0.5), 'center_distance')
            
            # Extract and validate size features
            size = element_features.get('size', [100, 30])
            area_normalized = self._normalize_area(size)
            aspect_ratio = self._calculate_aspect_ratio(size)
            
            # Extract and validate text features
            text = element_features.get('text', '')
            has_text = bool(text and text.strip())
            text_length_normalized = self._normalize_text_length(text)
            has_action_words = self._check_action_words(text)
            has_urgency_words = self._check_urgency_words(text)
            
            # Extract and validate visual features
            color_features = element_features.get('color_features', {})
            contrast = self._validate_range(color_features.get('contrast', 0.5), 'contrast')
            brightness = color_features.get('brightness', 128)
            brightness_normalized = self._validate_range(brightness / 255.0, 'brightness_normalized')
            
            # Extract and validate user context features
            tech_savviness_score = self._map_tech_savviness(user_context.get('tech_savviness', 'medium'))
            intent_score = self._infer_user_intent(user_context, task_context)
            device_complexity = self._map_device_complexity(user_context.get('device_type', 'desktop'))
            
            # Extract and validate task context features
            task_urgency = self._infer_task_urgency(task_context)
            task_type_score = self._map_task_type(task_context)
            completion_stage = self._infer_completion_stage(task_context)
            
            # Calculate interaction features
            user_element_compatibility = self._calculate_user_element_compatibility(
                tech_savviness_score, element_type, has_text
            )
            task_element_alignment = self._calculate_task_element_alignment(
                task_type_score, task_urgency, element_type, has_action_words
            )
            
            # Calculate quality indicators
            feature_completeness = self._calculate_feature_completeness(element_features, user_context, task_context)
            data_quality_score = self._calculate_data_quality(element_features, confidence, prominence)
            
            # Create and return clean features
            return CleanFeatures(
                element_id=element_id,
                element_type=element_type,
                prominence=prominence,
                confidence=confidence,
                relative_x=relative_x,
                relative_y=relative_y,
                center_distance=center_distance,
                area_normalized=area_normalized,
                aspect_ratio=aspect_ratio,
                has_text=has_text,
                text_length_normalized=text_length_normalized,
                has_action_words=has_action_words,
                has_urgency_words=has_urgency_words,
                contrast=contrast,
                brightness_normalized=brightness_normalized,
                tech_savviness_score=tech_savviness_score,
                intent_score=intent_score,
                device_complexity=device_complexity,
                task_urgency=task_urgency,
                task_type_score=task_type_score,
                completion_stage=completion_stage,
                user_element_compatibility=user_element_compatibility,
                task_element_alignment=task_element_alignment,
                feature_completeness=feature_completeness,
                data_quality_score=data_quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Feature integration failed: {e}")
            return self._create_fallback_features(element_features.get('id', 'unknown'))
    
    def _validate_range(self, value: float, feature_name: str) -> float:
        """Validate that a feature value is within expected range"""
        if feature_name not in self.feature_ranges:
            return max(0.0, min(1.0, float(value)))  # Default range [0, 1]
        
        min_val, max_val = self.feature_ranges[feature_name]
        validated_value = max(min_val, min(max_val, float(value)))
        
        if abs(validated_value - value) > 0.001:  # Log if clamping occurred
            self.logger.debug(f"Clamped {feature_name} from {value} to {validated_value}")
        
        return validated_value
    
    def _normalize_area(self, size: List[int]) -> float:
        """Normalize element area to [0, 1] range"""
        if len(size) < 2:
            return 0.1  # Small default area
        
        area = size[0] * size[1]
        
        # Normalize using log scale (handles wide range of areas better)
        # Typical UI elements range from 100 to 100,000 pixels²
        min_area = 100
        max_area = 100000
        
        if area <= min_area:
            return 0.0
        elif area >= max_area:
            return 1.0
        else:
            # Log-scale normalization
            log_area = np.log(area)
            log_min = np.log(min_area)
            log_max = np.log(max_area)
            return (log_area - log_min) / (log_max - log_min)
    
    def _calculate_aspect_ratio(self, size: List[int]) -> float:
        """Calculate and validate aspect ratio"""
        if len(size) < 2 or size[1] == 0:
            return 1.0  # Square default
        
        ratio = size[0] / size[1]
        return self._validate_range(ratio, 'aspect_ratio')
    
    def _normalize_text_length(self, text: str) -> float:
        """Normalize text length to [0, 1] range"""
        if not text:
            return 0.0
        
        length = len(text.strip())
        
        # Most UI text is under 100 characters
        max_length = 100
        return min(1.0, length / max_length)
    
    def _check_action_words(self, text: str) -> bool:
        """Check if text contains action words"""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(word in text_lower for word in self.action_words)
    
    def _check_urgency_words(self, text: str) -> bool:
        """Check if text contains urgency words"""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(word in text_lower for word in self.urgency_words)
    
    def _map_tech_savviness(self, tech_savviness: str) -> float:
        """Map tech savviness to numerical score"""
        mapping = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'expert': 1.0
        }
        return mapping.get(tech_savviness.lower(), 0.5)
    
    def _infer_user_intent(self, user_context: Dict[str, Any], 
                          task_context: Dict[str, Any]) -> float:
        """Infer user intent score (0 = exploring, 1 = goal-directed)"""
        
        # Start with neutral score
        intent_score = 0.5
        
        # Check user mood
        mood = user_context.get('mood', 'neutral').lower()
        if mood in ['focused', 'determined']:
            intent_score += 0.2
        elif mood in ['curious', 'relaxed']:
            intent_score -= 0.1
        
        # Check task description for goal-directed language
        task_description = task_context.get('task_description', '').lower()
        
        goal_directed_words = {'buy', 'purchase', 'complete', 'finish', 'submit', 'confirm'}
        exploring_words = {'browse', 'look', 'see', 'explore', 'check out'}
        
        if any(word in task_description for word in goal_directed_words):
            intent_score += 0.3
        elif any(word in task_description for word in exploring_words):
            intent_score -= 0.2
        
        return self._validate_range(intent_score, 'intent_score')
    
    def _map_device_complexity(self, device_type: str) -> float:
        """Map device type to complexity score"""
        return self.device_complexity_scores.get(device_type.lower(), 0.5)
    
    def _infer_task_urgency(self, task_context: Dict[str, Any]) -> float:
        """Infer task urgency from context"""
        
        # Check explicit urgency level
        urgency_level = task_context.get('urgency_level')
        if isinstance(urgency_level, (int, float)):
            return self._validate_range(urgency_level, 'task_urgency')
        elif isinstance(urgency_level, str):
            urgency_mapping = {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8,
                'critical': 1.0
            }
            return urgency_mapping.get(urgency_level.lower(), 0.5)
        
        # Infer from task description
        task_description = task_context.get('task_description', '').lower()
        
        urgency_score = 0.3  # Default low urgency
        
        for word in self.urgency_words:
            if word in task_description:
                urgency_score += 0.2
        
        return self._validate_range(urgency_score, 'task_urgency')
    
    def _map_task_type(self, task_context: Dict[str, Any]) -> float:
        """Map task type to numerical score"""
        
        # Check explicit task type
        task_type = task_context.get('task_type')
        if task_type:
            return self.task_type_scores.get(task_type.lower(), 0.5)
        
        # Infer from task description
        task_description = task_context.get('task_description', '').lower()
        
        # Check for task type keywords
        for task_type, score in self.task_type_scores.items():
            if task_type in task_description:
                return score
        
        return 0.4  # Default browse-like score
    
    def _infer_completion_stage(self, task_context: Dict[str, Any]) -> float:
        """Infer task completion stage"""
        
        # Check explicit completion stage
        stage = task_context.get('completion_stage')
        if isinstance(stage, (int, float)):
            return self._validate_range(stage, 'completion_stage')
        elif isinstance(stage, str):
            stage_mapping = {
                'start': 0.2,
                'beginning': 0.2,
                'middle': 0.5,
                'end': 0.8,
                'final': 0.9,
                'complete': 1.0
            }
            return stage_mapping.get(stage.lower(), 0.5)
        
        # Infer from task description
        task_description = task_context.get('task_description', '').lower()
        
        start_words = {'start', 'begin', 'first', 'initial'}
        middle_words = {'continue', 'next', 'proceed'}
        end_words = {'complete', 'finish', 'final', 'submit', 'checkout'}
        
        if any(word in task_description for word in end_words):
            return 0.8
        elif any(word in task_description for word in middle_words):
            return 0.6
        elif any(word in task_description for word in start_words):
            return 0.3
        
        return 0.5  # Default middle stage
    
    def _calculate_user_element_compatibility(self, tech_savviness: float, 
                                            element_type: str, has_text: bool) -> float:
        """Calculate how well the user profile matches the element"""
        
        base_compatibility = 0.5
        
        # Tech savviness vs element complexity
        element_complexity = {
            'button': 0.2,  # Simple
            'link': 0.4,    # Medium
            'form': 0.7,    # Complex
            'menu': 0.6,    # Medium-complex
            'text': 0.1,    # Very simple
            'image': 0.3    # Simple
        }.get(element_type, 0.5)
        
        # Good match if tech savviness aligns with element complexity
        complexity_match = 1.0 - abs(tech_savviness - element_complexity)
        base_compatibility += (complexity_match - 0.5) * 0.3
        
        # Text presence helps less tech-savvy users
        if has_text and tech_savviness < 0.6:
            base_compatibility += 0.1
        
        return self._validate_range(base_compatibility, 'user_element_compatibility')
    
    def _calculate_task_element_alignment(self, task_type_score: float, task_urgency: float,
                                        element_type: str, has_action_words: bool) -> float:
        """Calculate how well the element aligns with the task"""
        
        base_alignment = 0.5
        
        # High-urgency tasks favor buttons and action elements
        if task_urgency > 0.6 and element_type == 'button':
            base_alignment += 0.2
        
        # Purchase tasks favor buttons
        if task_type_score > 0.7 and element_type == 'button':
            base_alignment += 0.15
        
        # Browse tasks favor links
        if task_type_score < 0.4 and element_type == 'link':
            base_alignment += 0.1
        
        # Action words align with action-oriented tasks
        if has_action_words and task_type_score > 0.6:
            base_alignment += 0.15
        
        return self._validate_range(base_alignment, 'task_element_alignment')
    
    def _calculate_feature_completeness(self, element_features: Dict[str, Any],
                                      user_context: Dict[str, Any],
                                      task_context: Dict[str, Any]) -> float:
        """Calculate how complete the feature set is"""
        
        # Essential features checklist
        essential_features = [
            'element_type' in element_features,
            'prominence' in element_features,
            'size' in element_features,
            'position_features' in element_features,
            'tech_savviness' in user_context,
            'task_description' in task_context
        ]
        
        # Optional but valuable features
        optional_features = [
            'text' in element_features and element_features['text'],
            'color_features' in element_features,
            'confidence' in element_features,
            'mood' in user_context,
            'device_type' in user_context,
            'urgency_level' in task_context
        ]
        
        essential_score = sum(essential_features) / len(essential_features)
        optional_score = sum(optional_features) / len(optional_features)
        
        # Weight essential features more heavily
        return essential_score * 0.7 + optional_score * 0.3
    
    def _calculate_data_quality(self, element_features: Dict[str, Any],
                               confidence: float, prominence: float) -> float:
        """Calculate overall data quality score"""
        
        quality_factors = []
        
        # Detection confidence
        quality_factors.append(confidence)
        
        # Prominence reasonableness (not too extreme)
        prominence_quality = 1.0 - abs(prominence - 0.5) * 0.8  # Penalize extreme values slightly
        quality_factors.append(prominence_quality)
        
        # Size reasonableness
        size = element_features.get('size', [100, 30])
        if len(size) >= 2:
            area = size[0] * size[1]
            # Reasonable UI element size range
            if 50 <= area <= 50000:
                size_quality = 1.0
            elif area < 50:
                size_quality = area / 50.0  # Too small
            else:
                size_quality = max(0.3, 50000 / area)  # Too large
            quality_factors.append(size_quality)
        
        # Position reasonableness (within screen bounds)
        position_features = element_features.get('position_features', {})
        if position_features:
            rel_x = position_features.get('relative_x', 0.5)
            rel_y = position_features.get('relative_y', 0.5)
            
            # Should be within [0, 1] bounds with some tolerance
            x_quality = 1.0 if 0 <= rel_x <= 1 else max(0.0, 1.0 - abs(rel_x - 0.5))
            y_quality = 1.0 if 0 <= rel_y <= 1 else max(0.0, 1.0 - abs(rel_y - 0.5))
            
            quality_factors.extend([x_quality, y_quality])
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _create_fallback_features(self, element_id: str) -> CleanFeatures:
        """Create fallback features when integration fails"""
        
        return CleanFeatures(
            element_id=element_id,
            element_type='unknown',
            prominence=0.5,
            confidence=0.3,
            relative_x=0.5,
            relative_y=0.5,
            center_distance=0.5,
            area_normalized=0.1,
            aspect_ratio=2.0,
            has_text=False,
            text_length_normalized=0.0,
            has_action_words=False,
            has_urgency_words=False,
            contrast=0.5,
            brightness_normalized=0.5,
            tech_savviness_score=0.5,
            intent_score=0.5,
            device_complexity=0.5,
            task_urgency=0.3,
            task_type_score=0.4,
            completion_stage=0.5,
            user_element_compatibility=0.5,
            task_element_alignment=0.5,
            feature_completeness=0.2,
            data_quality_score=0.3
        )
    
    def to_dict(self, features: CleanFeatures) -> Dict[str, Any]:
        """Convert CleanFeatures to dictionary for model input"""
        
        return {
            'element_id': features.element_id,
            'element_type': features.element_type,
            'prominence': features.prominence,
            'confidence': features.confidence,
            'relative_x': features.relative_x,
            'relative_y': features.relative_y,
            'center_distance': features.center_distance,
            'area_normalized': features.area_normalized,
            'aspect_ratio': features.aspect_ratio,
            'has_text': float(features.has_text),
            'text_length_normalized': features.text_length_normalized,
            'has_action_words': float(features.has_action_words),
            'has_urgency_words': float(features.has_urgency_words),
            'contrast': features.contrast,
            'brightness_normalized': features.brightness_normalized,
            'tech_savviness_score': features.tech_savviness_score,
            'intent_score': features.intent_score,
            'device_complexity': features.device_complexity,
            'task_urgency': features.task_urgency,
            'task_type_score': features.task_type_score,
            'completion_stage': features.completion_stage,
            'user_element_compatibility': features.user_element_compatibility,
            'task_element_alignment': features.task_element_alignment,
            'feature_completeness': features.feature_completeness,
            'data_quality_score': features.data_quality_score
        }
    
    def to_array(self, features: CleanFeatures) -> np.ndarray:
        """Convert CleanFeatures to numpy array for ML models"""
        
        feature_dict = self.to_dict(features)
        # Remove non-numeric fields
        numeric_features = {k: v for k, v in feature_dict.items() 
                          if k not in ['element_id', 'element_type']}
        
        return np.array(list(numeric_features.values()), dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of numerical feature names in order"""
        
        return [
            'prominence', 'confidence', 'relative_x', 'relative_y', 'center_distance',
            'area_normalized', 'aspect_ratio', 'has_text', 'text_length_normalized',
            'has_action_words', 'has_urgency_words', 'contrast', 'brightness_normalized',
            'tech_savviness_score', 'intent_score', 'device_complexity',
            'task_urgency', 'task_type_score', 'completion_stage',
            'user_element_compatibility', 'task_element_alignment',
            'feature_completeness', 'data_quality_score'
        ]
    
    def validate_features(self, features: CleanFeatures) -> Tuple[bool, List[str]]:
        """Validate a CleanFeatures object and return issues"""
        
        issues = []
        
        # Check data quality threshold
        if features.data_quality_score < 0.3:
            issues.append(f"Low data quality score: {features.data_quality_score:.3f}")
        
        # Check feature completeness
        if features.feature_completeness < 0.5:
            issues.append(f"Low feature completeness: {features.feature_completeness:.3f}")
        
        # Check for extreme values that might indicate data issues
        if features.aspect_ratio > 10 or features.aspect_ratio < 0.2:
            issues.append(f"Extreme aspect ratio: {features.aspect_ratio:.2f}")
        
        if features.center_distance > 0.8:
            issues.append(f"Element very far from center: {features.center_distance:.3f}")
        
        # Check confidence levels
        if features.confidence < 0.2:
            issues.append(f"Very low detection confidence: {features.confidence:.3f}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_feature_summary(self, features: CleanFeatures) -> Dict[str, Any]:
        """Get a human-readable summary of the features"""
        
        return {
            'element_info': {
                'id': features.element_id,
                'type': features.element_type,
                'prominence': f"{features.prominence:.2f}",
                'confidence': f"{features.confidence:.2f}"
            },
            'position': {
                'relative_position': f"({features.relative_x:.2f}, {features.relative_y:.2f})",
                'center_distance': f"{features.center_distance:.2f}",
                'area_score': f"{features.area_normalized:.2f}"
            },
            'text_analysis': {
                'has_text': features.has_text,
                'text_length_score': f"{features.text_length_normalized:.2f}",
                'has_action_words': features.has_action_words,
                'has_urgency_words': features.has_urgency_words
            },
            'user_context': {
                'tech_savviness': f"{features.tech_savviness_score:.2f}",
                'intent': 'goal-directed' if features.intent_score > 0.6 else 'exploring',
                'device_complexity': f"{features.device_complexity:.2f}"
            },
            'task_context': {
                'urgency': f"{features.task_urgency:.2f}",
                'task_type_score': f"{features.task_type_score:.2f}",
                'completion_stage': f"{features.completion_stage:.2f}"
            },
            'compatibility': {
                'user_element': f"{features.user_element_compatibility:.2f}",
                'task_element': f"{features.task_element_alignment:.2f}"
            },
            'quality_metrics': {
                'feature_completeness': f"{features.feature_completeness:.2f}",
                'data_quality': f"{features.data_quality_score:.2f}"
            }
        }