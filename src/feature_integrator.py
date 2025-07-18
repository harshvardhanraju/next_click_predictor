from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
import json


@dataclass
class UserProfile:
    """User attributes and behavioral characteristics"""
    age_group: str
    tech_savviness: str
    mood: str
    device_type: str
    browsing_speed: str
    session_context: Dict[str, Any]


@dataclass
class TaskContext:
    """Task description and contextual information"""
    task_description: str
    task_type: str
    urgency_level: str
    completion_stage: str
    expected_actions: List[str]
    time_constraint: str


@dataclass
class IntegratedFeatures:
    """Combined feature set for Bayesian network"""
    user_features: Dict[str, Any]
    ui_features: List[Dict[str, Any]]
    task_features: Dict[str, Any]
    interaction_features: Dict[str, Any]
    feature_weights: Dict[str, float]


class FeatureIntegrator:
    """
    Combines user attributes, UI features, and task context into 
    a unified feature representation for Bayesian network modeling
    """
    
    def __init__(self):
        self.user_mappings = self._initialize_user_mappings()
        self.task_mappings = self._initialize_task_mappings()
        self.scaler = StandardScaler()
        
    def integrate_features(self, 
                          user_attributes: Dict[str, Any],
                          ui_elements: List[Dict[str, Any]], 
                          task_description: str) -> IntegratedFeatures:
        """
        Main integration pipeline
        
        Args:
            user_attributes: User profile data
            ui_elements: Extracted UI elements from screenshot
            task_description: Task description string
            
        Returns:
            IntegratedFeatures object with combined feature set
        """
        # Parse inputs
        user_profile = self._parse_user_profile(user_attributes)
        task_context = self._parse_task_context(task_description)
        
        # Extract features
        user_features = self._extract_user_features(user_profile)
        ui_features = self._extract_ui_features(ui_elements)
        task_features = self._extract_task_features(task_context)
        
        # Calculate interaction features
        interaction_features = self._calculate_interaction_features(
            user_profile, task_context, ui_elements
        )
        
        # Calculate feature weights
        feature_weights = self._calculate_feature_weights(
            user_profile, task_context, ui_elements
        )
        
        return IntegratedFeatures(
            user_features=user_features,
            ui_features=ui_features,
            task_features=task_features,
            interaction_features=interaction_features,
            feature_weights=feature_weights
        )
    
    def _initialize_user_mappings(self) -> Dict[str, Dict[str, float]]:
        """Initialize mappings for user attributes to numerical values"""
        return {
            'age_group': {
                '18-24': 0.1, '25-34': 0.3, '35-44': 0.5, 
                '45-54': 0.7, '55-64': 0.9, '65+': 1.0
            },
            'tech_savviness': {
                'low': 0.2, 'medium': 0.5, 'high': 0.8, 'expert': 1.0
            },
            'mood': {
                'frustrated': 0.1, 'neutral': 0.5, 'focused': 0.8, 'excited': 1.0
            },
            'device_type': {
                'mobile': 0.3, 'tablet': 0.5, 'desktop': 0.7, 'laptop': 0.9
            },
            'browsing_speed': {
                'slow': 0.2, 'medium': 0.5, 'fast': 0.8, 'very_fast': 1.0
            }
        }
    
    def _initialize_task_mappings(self) -> Dict[str, Dict[str, float]]:
        """Initialize mappings for task attributes"""
        return {
            'task_type': {
                'browse': 0.2, 'search': 0.4, 'purchase': 0.6, 
                'signup': 0.8, 'urgent': 1.0
            },
            'urgency_level': {
                'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0
            },
            'completion_stage': {
                'start': 0.2, 'middle': 0.5, 'near_end': 0.8, 'final': 1.0
            },
            'time_constraint': {
                'none': 0.1, 'flexible': 0.3, 'moderate': 0.6, 
                'tight': 0.8, 'immediate': 1.0
            }
        }
    
    def _parse_user_profile(self, user_attributes: Dict[str, Any]) -> UserProfile:
        """Parse user attributes into structured profile"""
        return UserProfile(
            age_group=user_attributes.get('age_group', 'unknown'),
            tech_savviness=user_attributes.get('tech_savviness', 'medium'),
            mood=user_attributes.get('mood', 'neutral'),
            device_type=user_attributes.get('device_type', 'desktop'),
            browsing_speed=user_attributes.get('browsing_speed', 'medium'),
            session_context=user_attributes.get('session_context', {})
        )
    
    def _parse_task_context(self, task_description: str) -> TaskContext:
        """Parse task description into structured context"""
        # Analyze task description for key information
        task_type = self._infer_task_type(task_description)
        urgency_level = self._infer_urgency_level(task_description)
        completion_stage = self._infer_completion_stage(task_description)
        expected_actions = self._extract_expected_actions(task_description)
        time_constraint = self._infer_time_constraint(task_description)
        
        return TaskContext(
            task_description=task_description,
            task_type=task_type,
            urgency_level=urgency_level,
            completion_stage=completion_stage,
            expected_actions=expected_actions,
            time_constraint=time_constraint
        )
    
    def _infer_task_type(self, task_description: str) -> str:
        """Infer task type from description"""
        desc_lower = task_description.lower()
        
        if any(word in desc_lower for word in ['buy', 'purchase', 'checkout', 'order']):
            return 'purchase'
        elif any(word in desc_lower for word in ['sign up', 'register', 'create account']):
            return 'signup'
        elif any(word in desc_lower for word in ['search', 'find', 'look for']):
            return 'search'
        elif any(word in desc_lower for word in ['urgent', 'immediate', 'asap']):
            return 'urgent'
        else:
            return 'browse'
    
    def _infer_urgency_level(self, task_description: str) -> str:
        """Infer urgency level from description"""
        desc_lower = task_description.lower()
        
        if any(word in desc_lower for word in ['urgent', 'immediate', 'asap', 'quickly']):
            return 'high'
        elif any(word in desc_lower for word in ['soon', 'today', 'now']):
            return 'medium'
        else:
            return 'low'
    
    def _infer_completion_stage(self, task_description: str) -> str:
        """Infer completion stage from description"""
        desc_lower = task_description.lower()
        
        if any(word in desc_lower for word in ['complete', 'finish', 'final', 'checkout']):
            return 'final'
        elif any(word in desc_lower for word in ['continue', 'next', 'proceed']):
            return 'middle'
        else:
            return 'start'
    
    def _extract_expected_actions(self, task_description: str) -> List[str]:
        """Extract expected actions from description"""
        desc_lower = task_description.lower()
        actions = []
        
        action_keywords = {
            'click': ['click', 'tap', 'press'],
            'fill': ['fill', 'enter', 'type'],
            'select': ['select', 'choose', 'pick'],
            'submit': ['submit', 'send', 'confirm'],
            'navigate': ['go to', 'navigate', 'visit']
        }
        
        for action, keywords in action_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                actions.append(action)
        
        return actions if actions else ['click']
    
    def _infer_time_constraint(self, task_description: str) -> str:
        """Infer time constraint from description"""
        desc_lower = task_description.lower()
        
        if any(word in desc_lower for word in ['immediate', 'asap', 'urgent']):
            return 'immediate'
        elif any(word in desc_lower for word in ['quickly', 'fast', 'soon']):
            return 'tight'
        elif any(word in desc_lower for word in ['when convenient', 'later']):
            return 'flexible'
        else:
            return 'moderate'
    
    def _extract_user_features(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Extract numerical features from user profile"""
        features = {}
        
        # Map categorical attributes to numerical values
        for attr in ['age_group', 'tech_savviness', 'mood', 'device_type', 'browsing_speed']:
            value = getattr(user_profile, attr)
            if value in self.user_mappings[attr]:
                features[attr] = self.user_mappings[attr][value]
            else:
                features[attr] = 0.5  # Default neutral value
        
        # Add derived features
        features['experience_level'] = (
            features['tech_savviness'] * 0.6 + 
            features['browsing_speed'] * 0.4
        )
        
        features['patience_level'] = (
            features['mood'] * 0.7 + 
            (1.0 - features['browsing_speed']) * 0.3
        )
        
        return features
    
    def _extract_ui_features(self, ui_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and normalize UI element features"""
        enhanced_elements = []
        
        for element in ui_elements:
            features = element.copy()
            
            # Add element type encoding
            features['type_encoding'] = self._encode_element_type(element['type'])
            
            # Add position category
            pos_features = element.get('position_features', {})
            features['position_category'] = self._categorize_position(pos_features)
            
            # Add size category
            size = element.get('size', [0, 0])
            features['size_category'] = self._categorize_size(size)
            
            # Add text analysis
            text = element.get('text', '')
            features['text_analysis'] = self._analyze_text(text)
            
            enhanced_elements.append(features)
        
        return enhanced_elements
    
    def _encode_element_type(self, element_type: str) -> Dict[str, float]:
        """Encode element type as feature vector"""
        type_mapping = {
            'button': 0.9,
            'link': 0.7,
            'form': 0.5,
            'menu': 0.3,
            'text': 0.1,
            'image': 0.2
        }
        
        return {
            'clickability': type_mapping.get(element_type, 0.5),
            'importance': type_mapping.get(element_type, 0.5),
            'interaction_type': element_type
        }
    
    def _categorize_position(self, position_features: Dict[str, Any]) -> str:
        """Categorize element position"""
        if not position_features:
            return 'unknown'
        
        rel_x = position_features.get('relative_x', 0.5)
        rel_y = position_features.get('relative_y', 0.5)
        
        if rel_y < 0.3:
            return 'top'
        elif rel_y > 0.7:
            return 'bottom'
        elif rel_x < 0.3:
            return 'left'
        elif rel_x > 0.7:
            return 'right'
        else:
            return 'center'
    
    def _categorize_size(self, size: List[int]) -> str:
        """Categorize element size"""
        if len(size) < 2:
            return 'unknown'
        
        area = size[0] * size[1]
        
        if area < 1000:
            return 'small'
        elif area < 5000:
            return 'medium'
        elif area < 15000:
            return 'large'
        else:
            return 'very_large'
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text content for semantic features"""
        if not text:
            return {'has_text': False, 'action_words': 0, 'urgency_words': 0}
        
        text_lower = text.lower()
        
        action_words = ['click', 'buy', 'purchase', 'continue', 'next', 'submit', 'login']
        urgency_words = ['now', 'today', 'limited', 'urgent', 'quick', 'fast']
        
        action_count = sum(1 for word in action_words if word in text_lower)
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        
        return {
            'has_text': True,
            'text_length': len(text),
            'action_words': action_count,
            'urgency_words': urgency_count,
            'word_count': len(text.split())
        }
    
    def _extract_task_features(self, task_context: TaskContext) -> Dict[str, Any]:
        """Extract numerical features from task context"""
        features = {}
        
        # Map categorical attributes to numerical values
        for attr in ['task_type', 'urgency_level', 'completion_stage', 'time_constraint']:
            value = getattr(task_context, attr)
            if value in self.task_mappings[attr]:
                features[attr] = self.task_mappings[attr][value]
            else:
                features[attr] = 0.5  # Default neutral value
        
        # Add derived features
        features['task_complexity'] = len(task_context.expected_actions) / 5.0
        features['goal_directedness'] = (
            features['urgency_level'] * 0.5 + 
            features['completion_stage'] * 0.5
        )
        
        return features
    
    def _calculate_interaction_features(self, 
                                      user_profile: UserProfile,
                                      task_context: TaskContext,
                                      ui_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate features representing user-task-UI interactions"""
        
        # User-task compatibility
        user_task_compatibility = self._calculate_user_task_compatibility(
            user_profile, task_context
        )
        
        # UI-task alignment
        ui_task_alignment = self._calculate_ui_task_alignment(
            ui_elements, task_context
        )
        
        # User-UI compatibility
        user_ui_compatibility = self._calculate_user_ui_compatibility(
            user_profile, ui_elements
        )
        
        return {
            'user_task_compatibility': user_task_compatibility,
            'ui_task_alignment': ui_task_alignment,
            'user_ui_compatibility': user_ui_compatibility,
            'overall_alignment': np.mean([
                user_task_compatibility, 
                ui_task_alignment, 
                user_ui_compatibility
            ])
        }
    
    def _calculate_user_task_compatibility(self, 
                                         user_profile: UserProfile,
                                         task_context: TaskContext) -> float:
        """Calculate how well user attributes match task requirements"""
        
        # High tech savviness + complex task = good compatibility
        tech_task_match = 0.0
        if task_context.task_type == 'purchase' and user_profile.tech_savviness in ['high', 'expert']:
            tech_task_match = 0.8
        elif task_context.task_type == 'browse' and user_profile.tech_savviness in ['low', 'medium']:
            tech_task_match = 0.6
        
        # Mood-urgency alignment
        mood_urgency_match = 0.0
        if task_context.urgency_level == 'high' and user_profile.mood == 'focused':
            mood_urgency_match = 0.9
        elif task_context.urgency_level == 'low' and user_profile.mood in ['neutral', 'excited']:
            mood_urgency_match = 0.7
        
        return (tech_task_match + mood_urgency_match) / 2.0
    
    def _calculate_ui_task_alignment(self, 
                                   ui_elements: List[Dict[str, Any]],
                                   task_context: TaskContext) -> float:
        """Calculate how well UI elements support the task"""
        
        relevant_elements = 0
        total_elements = len(ui_elements)
        
        for element in ui_elements:
            element_type = element.get('type', '')
            element_text = element.get('text', '').lower()
            
            # Check if element supports expected actions
            if any(action in element_text for action in task_context.expected_actions):
                relevant_elements += 1
            
            # Check type-task alignment
            if task_context.task_type == 'purchase' and element_type == 'button':
                if any(word in element_text for word in ['buy', 'purchase', 'checkout']):
                    relevant_elements += 1
        
        return relevant_elements / total_elements if total_elements > 0 else 0.0
    
    def _calculate_user_ui_compatibility(self, 
                                       user_profile: UserProfile,
                                       ui_elements: List[Dict[str, Any]]) -> float:
        """Calculate how well UI design matches user preferences"""
        
        compatibility_score = 0.0
        
        # Check for high prominence elements for low tech savviness users
        if user_profile.tech_savviness == 'low':
            high_prominence_elements = [
                e for e in ui_elements 
                if e.get('prominence', 0) > 0.7
            ]
            compatibility_score += len(high_prominence_elements) / len(ui_elements)
        
        # Check for multiple options for high tech savviness users
        if user_profile.tech_savviness in ['high', 'expert']:
            if len(ui_elements) > 5:  # More options available
                compatibility_score += 0.3
        
        return min(1.0, compatibility_score)
    
    def _calculate_feature_weights(self, 
                                 user_profile: UserProfile,
                                 task_context: TaskContext,
                                 ui_elements: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weights for different feature categories"""
        
        weights = {
            'user_features': 0.3,
            'ui_features': 0.4,
            'task_features': 0.2,
            'interaction_features': 0.1
        }
        
        # Adjust weights based on context
        if task_context.urgency_level == 'high':
            weights['task_features'] += 0.1
            weights['ui_features'] -= 0.1
        
        if user_profile.tech_savviness == 'low':
            weights['ui_features'] += 0.1
            weights['user_features'] -= 0.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}
    
    def get_feature_summary(self, integrated_features: IntegratedFeatures) -> Dict[str, Any]:
        """Get a summary of integrated features for debugging/analysis"""
        return {
            'user_feature_count': len(integrated_features.user_features),
            'ui_element_count': len(integrated_features.ui_features),
            'task_feature_count': len(integrated_features.task_features),
            'interaction_feature_count': len(integrated_features.interaction_features),
            'feature_weights': integrated_features.feature_weights,
            'key_user_attributes': {
                k: v for k, v in integrated_features.user_features.items()
                if k in ['tech_savviness', 'mood', 'experience_level']
            },
            'key_task_attributes': {
                k: v for k, v in integrated_features.task_features.items()
                if k in ['task_type', 'urgency_level', 'goal_directedness']
            }
        }