from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass
from feature_integrator import IntegratedFeatures


@dataclass
class ExplanationFactor:
    """Represents a factor contributing to the prediction"""
    factor_name: str
    factor_type: str  # 'user', 'task', 'ui', 'interaction'
    influence: float  # -1 to 1, negative means reduces click probability
    importance: float  # 0 to 1, how important this factor is
    description: str
    evidence_value: str


class ExplanationGenerator:
    """
    Generates human-readable explanations for click predictions
    """
    
    def __init__(self):
        self.explanation_templates = self._initialize_explanation_templates()
        self.factor_descriptions = self._initialize_factor_descriptions()
    
    def generate_explanation(self, 
                           predictions: List[Dict[str, Any]],
                           integrated_features: IntegratedFeatures,
                           top_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for the top prediction
        
        Args:
            predictions: List of all predictions
            integrated_features: Feature set used for prediction
            top_prediction: The highest probability prediction
            
        Returns:
            Dictionary containing explanation components
        """
        
        # Identify key factors
        key_factors = self._identify_key_factors(top_prediction, integrated_features)
        
        # Generate main explanation
        main_explanation = self._generate_main_explanation(top_prediction, key_factors)
        
        # Generate factor-specific explanations
        factor_explanations = self._generate_factor_explanations(key_factors)
        
        # Generate confidence explanation
        confidence_explanation = self._generate_confidence_explanation(top_prediction)
        
        # Generate alternative explanations
        alternative_explanations = self._generate_alternative_explanations(predictions)
        
        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(key_factors, top_prediction)
        
        return {
            'main_explanation': main_explanation,
            'reasoning_chain': reasoning_chain,
            'key_factors': [self._factor_to_dict(f) for f in key_factors],
            'factor_explanations': factor_explanations,
            'confidence_explanation': confidence_explanation,
            'alternative_explanations': alternative_explanations,
            'prediction_summary': {
                'element_id': top_prediction['element_id'],
                'element_type': top_prediction['element_type'],
                'element_text': top_prediction['element_text'],
                'probability': top_prediction['click_probability'],
                'confidence': top_prediction['confidence']
            }
        }
    
    def _identify_key_factors(self, 
                            prediction: Dict[str, Any],
                            integrated_features: IntegratedFeatures) -> List[ExplanationFactor]:
        """Identify the most important factors contributing to the prediction"""
        
        factors = []
        
        # User factors
        user_factors = self._analyze_user_factors(integrated_features.user_features)
        factors.extend(user_factors)
        
        # Task factors
        task_factors = self._analyze_task_factors(integrated_features.task_features)
        factors.extend(task_factors)
        
        # UI factors
        ui_factors = self._analyze_ui_factors(prediction, integrated_features.ui_features)
        factors.extend(ui_factors)
        
        # Interaction factors
        interaction_factors = self._analyze_interaction_factors(integrated_features.interaction_features)
        factors.extend(interaction_factors)
        
        # Sort by importance and return top factors
        factors.sort(key=lambda x: x.importance, reverse=True)
        return factors[:5]  # Return top 5 factors
    
    def _analyze_user_factors(self, user_features: Dict[str, Any]) -> List[ExplanationFactor]:
        """Analyze user-related factors"""
        factors = []
        
        # Tech savviness
        tech_savviness = user_features.get('tech_savviness', 0.5)
        if tech_savviness > 0.7:
            factors.append(ExplanationFactor(
                factor_name='high_tech_savviness',
                factor_type='user',
                influence=0.3,
                importance=0.8,
                description="User's high technical proficiency increases likelihood of efficient navigation",
                evidence_value='high'
            ))
        elif tech_savviness < 0.3:
            factors.append(ExplanationFactor(
                factor_name='low_tech_savviness',
                factor_type='user',
                influence=-0.2,
                importance=0.7,
                description="User's limited technical experience may require more prominent interface elements",
                evidence_value='low'
            ))
        
        # Mood
        mood = user_features.get('mood', 0.5)
        if mood > 0.7:
            factors.append(ExplanationFactor(
                factor_name='positive_mood',
                factor_type='user',
                influence=0.2,
                importance=0.6,
                description="User's positive mood facilitates confident decision-making",
                evidence_value='positive'
            ))
        elif mood < 0.3:
            factors.append(ExplanationFactor(
                factor_name='negative_mood',
                factor_type='user',
                influence=-0.3,
                importance=0.8,
                description="User's negative mood may lead to more cautious behavior",
                evidence_value='negative'
            ))
        
        # Experience level
        experience = user_features.get('experience_level', 0.5)
        if experience > 0.7:
            factors.append(ExplanationFactor(
                factor_name='high_experience',
                factor_type='user',
                influence=0.25,
                importance=0.7,
                description="User's experience enables quick identification of relevant interface elements",
                evidence_value='high'
            ))
        
        return factors
    
    def _analyze_task_factors(self, task_features: Dict[str, Any]) -> List[ExplanationFactor]:
        """Analyze task-related factors"""
        factors = []
        
        # Task urgency
        urgency = task_features.get('urgency_level', 0.5)
        if urgency > 0.7:
            factors.append(ExplanationFactor(
                factor_name='high_urgency',
                factor_type='task',
                influence=0.4,
                importance=0.9,
                description="High task urgency drives focus toward the most direct path to completion",
                evidence_value='high'
            ))
        elif urgency < 0.3:
            factors.append(ExplanationFactor(
                factor_name='low_urgency',
                factor_type='task',
                influence=-0.1,
                importance=0.5,
                description="Low task urgency allows for more exploratory behavior",
                evidence_value='low'
            ))
        
        # Task type
        task_type = task_features.get('task_type', 0.5)
        if task_type > 0.7:  # Purchase/action task
            factors.append(ExplanationFactor(
                factor_name='action_oriented_task',
                factor_type='task',
                influence=0.3,
                importance=0.8,
                description="Action-oriented task increases focus on interactive elements",
                evidence_value='action/purchase'
            ))
        
        # Goal directedness
        goal_directedness = task_features.get('goal_directedness', 0.5)
        if goal_directedness > 0.7:
            factors.append(ExplanationFactor(
                factor_name='high_goal_directedness',
                factor_type='task',
                influence=0.35,
                importance=0.8,
                description="Clear goal direction strongly influences element selection",
                evidence_value='high'
            ))
        
        return factors
    
    def _analyze_ui_factors(self, 
                          prediction: Dict[str, Any],
                          ui_features: List[Dict[str, Any]]) -> List[ExplanationFactor]:
        """Analyze UI-related factors for the predicted element"""
        factors = []
        
        # Find the predicted element
        predicted_element = None
        for element in ui_features:
            if element.get('id') == prediction['element_id']:
                predicted_element = element
                break
        
        if not predicted_element:
            return factors
        
        # Element prominence
        prominence = predicted_element.get('prominence', 0.5)
        if prominence > 0.7:
            factors.append(ExplanationFactor(
                factor_name='high_prominence',
                factor_type='ui',
                influence=0.4,
                importance=0.9,
                description="Element's high visual prominence makes it a natural focus point",
                evidence_value='high'
            ))
        elif prominence < 0.3:
            factors.append(ExplanationFactor(
                factor_name='low_prominence',
                factor_type='ui',
                influence=-0.3,
                importance=0.7,
                description="Element's low visual prominence reduces its likelihood of selection",
                evidence_value='low'
            ))
        
        # Element type
        element_type = predicted_element.get('type', 'unknown')
        if element_type == 'button':
            factors.append(ExplanationFactor(
                factor_name='button_element',
                factor_type='ui',
                influence=0.5,
                importance=0.8,
                description="Button elements are designed for user interaction and action",
                evidence_value='button'
            ))
        elif element_type == 'link':
            factors.append(ExplanationFactor(
                factor_name='link_element',
                factor_type='ui',
                influence=0.3,
                importance=0.6,
                description="Link elements suggest navigation or additional information",
                evidence_value='link'
            ))
        
        # Element position
        position_features = predicted_element.get('position_features', {})
        center_distance = position_features.get('center_distance', 0.5)
        if center_distance < 0.3:  # Close to center
            factors.append(ExplanationFactor(
                factor_name='central_position',
                factor_type='ui',
                influence=0.3,
                importance=0.7,
                description="Central positioning increases element visibility and accessibility",
                evidence_value='center'
            ))
        
        # Text analysis
        text_analysis = predicted_element.get('text_analysis', {})
        if text_analysis.get('action_words', 0) > 0:
            factors.append(ExplanationFactor(
                factor_name='action_text',
                factor_type='ui',
                influence=0.4,
                importance=0.8,
                description="Action-oriented text clearly indicates the element's purpose",
                evidence_value=f"Contains action words: {text_analysis.get('action_words', 0)}"
            ))
        
        return factors
    
    def _analyze_interaction_factors(self, interaction_features: Dict[str, Any]) -> List[ExplanationFactor]:
        """Analyze interaction-related factors"""
        factors = []
        
        # User-task compatibility
        user_task_compatibility = interaction_features.get('user_task_compatibility', 0.5)
        if user_task_compatibility > 0.7:
            factors.append(ExplanationFactor(
                factor_name='high_user_task_compatibility',
                factor_type='interaction',
                influence=0.3,
                importance=0.8,
                description="Strong alignment between user characteristics and task requirements",
                evidence_value='high'
            ))
        
        # UI-task alignment
        ui_task_alignment = interaction_features.get('ui_task_alignment', 0.5)
        if ui_task_alignment > 0.7:
            factors.append(ExplanationFactor(
                factor_name='high_ui_task_alignment',
                factor_type='interaction',
                influence=0.35,
                importance=0.8,
                description="UI elements strongly support the user's task objectives",
                evidence_value='high'
            ))
        
        return factors
    
    def _generate_main_explanation(self, 
                                 prediction: Dict[str, Any],
                                 key_factors: List[ExplanationFactor]) -> str:
        """Generate the main explanation text"""
        
        element_text = prediction['element_text']
        element_type = prediction['element_type']
        probability = prediction['click_probability']
        
        # Start with prediction statement
        explanation = f"The system predicts you'll click the '{element_text}' {element_type} "
        explanation += f"with {probability:.0%} probability because:\n\n"
        
        # Add top factors
        for i, factor in enumerate(key_factors[:3]):
            explanation += f"{i+1}. {factor.description}\n"
        
        return explanation
    
    def _generate_factor_explanations(self, key_factors: List[ExplanationFactor]) -> List[str]:
        """Generate detailed explanations for each factor"""
        explanations = []
        
        for factor in key_factors:
            if factor.factor_type == 'user':
                explanation = self._explain_user_factor(factor)
            elif factor.factor_type == 'task':
                explanation = self._explain_task_factor(factor)
            elif factor.factor_type == 'ui':
                explanation = self._explain_ui_factor(factor)
            elif factor.factor_type == 'interaction':
                explanation = self._explain_interaction_factor(factor)
            else:
                explanation = factor.description
            
            explanations.append(explanation)
        
        return explanations
    
    def _explain_user_factor(self, factor: ExplanationFactor) -> str:
        """Generate explanation for user factors"""
        templates = {
            'high_tech_savviness': "Your high technical proficiency means you can quickly navigate complex interfaces and identify the most efficient paths to your goals.",
            'low_tech_savviness': "Your preference for simpler interfaces means you're more likely to click on prominent, clearly labeled elements.",
            'positive_mood': "Your positive mood state facilitates confident decision-making and action-taking.",
            'negative_mood': "Your current mood may lead to more cautious behavior and preference for familiar interface patterns.",
            'high_experience': "Your experience with similar interfaces helps you quickly identify the most relevant elements."
        }
        
        return templates.get(factor.factor_name, factor.description)
    
    def _explain_task_factor(self, factor: ExplanationFactor) -> str:
        """Generate explanation for task factors"""
        templates = {
            'high_urgency': "The urgency of your task drives you to focus on the most direct path to completion, bypassing secondary options.",
            'action_oriented_task': "Your goal-oriented task increases the likelihood of selecting interactive elements that advance your objective.",
            'high_goal_directedness': "Your clear objective creates strong preference for elements that directly support task completion."
        }
        
        return templates.get(factor.factor_name, factor.description)
    
    def _explain_ui_factor(self, factor: ExplanationFactor) -> str:
        """Generate explanation for UI factors"""
        templates = {
            'high_prominence': "The element's prominent visual design (size, color, position) makes it naturally attract attention.",
            'button_element': "Buttons are specifically designed as interactive elements that clearly invite user action.",
            'central_position': "The element's central positioning ensures high visibility and easy access.",
            'action_text': "The element's text clearly communicates its action-oriented purpose."
        }
        
        return templates.get(factor.factor_name, factor.description)
    
    def _explain_interaction_factor(self, factor: ExplanationFactor) -> str:
        """Generate explanation for interaction factors"""
        templates = {
            'high_user_task_compatibility': "Your personal characteristics align well with the task requirements, enabling efficient decision-making.",
            'high_ui_task_alignment': "The interface design strongly supports your current task, making the relevant elements easily identifiable."
        }
        
        return templates.get(factor.factor_name, factor.description)
    
    def _generate_confidence_explanation(self, prediction: Dict[str, Any]) -> str:
        """Generate explanation for confidence level"""
        confidence = prediction['confidence']
        probability = prediction['click_probability']
        
        if confidence > 0.8:
            return f"High confidence ({confidence:.0%}): The prediction is based on strong alignment between multiple factors."
        elif confidence > 0.6:
            return f"Medium confidence ({confidence:.0%}): The prediction is supported by several consistent factors."
        else:
            return f"Lower confidence ({confidence:.0%}): The prediction involves some uncertainty due to conflicting factors."
    
    def _generate_alternative_explanations(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate explanations for alternative predictions"""
        alternatives = []
        
        # Get top 3 alternatives (excluding the top prediction)
        for prediction in predictions[1:4]:
            element_text = prediction['element_text']
            element_type = prediction['element_type']
            probability = prediction['click_probability']
            
            if probability > 0.1:  # Only include reasonable alternatives
                alternative = f"Alternative: '{element_text}' {element_type} ({probability:.0%} probability)"
                if probability > 0.3:
                    alternative += " - Also a strong candidate based on your profile"
                alternatives.append(alternative)
        
        return alternatives
    
    def _generate_reasoning_chain(self, 
                                key_factors: List[ExplanationFactor],
                                prediction: Dict[str, Any]) -> List[str]:
        """Generate step-by-step reasoning chain"""
        chain = []
        
        # Step 1: Context assessment
        user_factors = [f for f in key_factors if f.factor_type == 'user']
        task_factors = [f for f in key_factors if f.factor_type == 'task']
        
        if user_factors and task_factors:
            chain.append(f"1. Context Analysis: {user_factors[0].description} combined with {task_factors[0].description}")
        
        # Step 2: UI element evaluation
        ui_factors = [f for f in key_factors if f.factor_type == 'ui']
        if ui_factors:
            chain.append(f"2. Element Evaluation: {ui_factors[0].description}")
        
        # Step 3: Interaction analysis
        interaction_factors = [f for f in key_factors if f.factor_type == 'interaction']
        if interaction_factors:
            chain.append(f"3. Interaction Analysis: {interaction_factors[0].description}")
        
        # Step 4: Final prediction
        chain.append(f"4. Prediction: Based on these factors, the '{prediction['element_text']}' element emerges as the most likely choice")
        
        return chain
    
    def _factor_to_dict(self, factor: ExplanationFactor) -> Dict[str, Any]:
        """Convert ExplanationFactor to dictionary"""
        return {
            'name': factor.factor_name,
            'type': factor.factor_type,
            'influence': factor.influence,
            'importance': factor.importance,
            'description': factor.description,
            'evidence': factor.evidence_value
        }
    
    def _initialize_explanation_templates(self) -> Dict[str, str]:
        """Initialize explanation templates"""
        return {
            'high_probability': "The system is highly confident in this prediction due to strong alignment across multiple factors.",
            'medium_probability': "The system identifies this as the most likely choice based on the available evidence.",
            'low_probability': "While this is the top prediction, there's significant uncertainty due to conflicting factors.",
            'fallback': "The system used heuristic-based prediction due to insufficient probabilistic evidence."
        }
    
    def _initialize_factor_descriptions(self) -> Dict[str, str]:
        """Initialize factor descriptions"""
        return {
            'user_factors': {
                'tech_savviness': "User's technical proficiency level",
                'mood': "User's current emotional state",
                'experience': "User's experience with similar interfaces"
            },
            'task_factors': {
                'urgency': "Time pressure and task urgency",
                'type': "Nature of the task being performed",
                'goal_directedness': "Clarity of the user's objective"
            },
            'ui_factors': {
                'prominence': "Visual prominence and salience",
                'type': "Element type and interaction affordances",
                'position': "Spatial positioning and accessibility"
            },
            'interaction_factors': {
                'compatibility': "Alignment between user, task, and interface"
            }
        }
    
    def generate_simple_explanation(self, prediction: Dict[str, Any]) -> str:
        """Generate a simple, one-sentence explanation"""
        element_text = prediction['element_text']
        element_type = prediction['element_type']
        probability = prediction['click_probability']
        
        if probability > 0.7:
            return f"High likelihood of clicking '{element_text}' {element_type} due to strong task alignment and element prominence."
        elif probability > 0.5:
            return f"Moderate likelihood of clicking '{element_text}' {element_type} based on user profile and interface design."
        else:
            return f"'{element_text}' {element_type} is the top choice, though with some uncertainty due to multiple viable options."