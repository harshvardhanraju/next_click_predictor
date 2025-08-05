import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import json

# Optional imports with error handling
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    try:
        # Try older pgmpy version
        from pgmpy.models import BayesianModel as BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
        PGMPY_AVAILABLE = True
    except ImportError:
        PGMPY_AVAILABLE = False
        logging.warning("pgmpy not available. Using fallback Bayesian network implementation.")


@dataclass
class ExplanationComponent:
    """Component of the explanation for a prediction"""
    factor_name: str
    factor_value: str
    influence_score: float  # -1 to 1, negative discourages, positive encourages
    explanation_text: str


@dataclass
class BayesianPrediction:
    """Prediction result with explainable components"""
    element_id: str
    click_probability: float
    confidence: float
    explanation_components: List[ExplanationComponent]
    reasoning_chain: List[str]


class SimpleBayesianNetwork:
    """
    Simplified, explainable Bayesian network for click prediction
    Focuses on interpretability and clear reasoning chains
    """
    
    def __init__(self):
        self.model = None
        self.inference_engine = None
        self.node_states = self._initialize_node_states()
        self.explanation_templates = self._initialize_explanation_templates()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_node_states(self) -> Dict[str, List[str]]:
        """Initialize simplified node states for explainability"""
        return {
            # User context (simplified to 2 states each for clarity)
            'user_expertise': ['novice', 'expert'],
            'user_intent': ['exploring', 'goal_directed'],
            
            # Element properties (key predictive features)
            'element_type': ['button', 'link', 'text', 'form'],
            'element_prominence': ['low', 'high'],
            'element_position': ['peripheral', 'central'],
            
            # Task context
            'task_urgency': ['low', 'high'],
            
            # Decision node
            'will_click': ['no', 'yes']
        }
    
    def _initialize_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize human-readable explanation templates"""
        return {
            'user_expertise': {
                'novice': "User has limited technical experience",
                'expert': "User is technically proficient"
            },
            'user_intent': {
                'exploring': "User is browsing and exploring options",
                'goal_directed': "User has a specific goal in mind"
            },
            'element_type': {
                'button': "Element is a clickable button",
                'link': "Element is a navigation link",
                'text': "Element is primarily text content",
                'form': "Element is an input field"
            },
            'element_prominence': {
                'low': "Element has low visual prominence",
                'high': "Element stands out visually"
            },
            'element_position': {
                'peripheral': "Element is in a less prominent screen position",
                'central': "Element is in a prominent central position"
            },
            'task_urgency': {
                'low': "Task has low urgency",
                'high': "Task is time-sensitive or urgent"
            }
        }
    
    def build_network(self, training_data: Optional[List[Dict]] = None) -> bool:
        """
        Build the simplified Bayesian network
        
        Args:
            training_data: Optional training data for learning CPDs
            
        Returns:
            True if network built successfully
        """
        try:
            if PGMPY_AVAILABLE:
                return self._build_pgmpy_network(training_data)
            else:
                return self._build_fallback_network(training_data)
                
        except Exception as e:
            self.logger.error(f"Failed to build Bayesian network: {e}")
            return False
    
    def _build_pgmpy_network(self, training_data: Optional[List[Dict]] = None) -> bool:
        """Build network using pgmpy library"""
        
        # Define simplified network structure for explainability
        edges = [
            # User factors influence decision
            ('user_expertise', 'will_click'),
            ('user_intent', 'will_click'),
            
            # Element properties influence decision
            ('element_type', 'will_click'),
            ('element_prominence', 'will_click'),
            ('element_position', 'will_click'),
            
            # Task context influences decision
            ('task_urgency', 'will_click'),
            
            # Interaction effects
            ('user_expertise', 'element_type'),  # Experts prefer different element types
            ('user_intent', 'task_urgency')     # Intent affects perceived urgency
        ]
        
        # Create network
        self.model = BayesianNetwork(edges)
        
        # Add CPDs based on training data or use defaults
        if training_data:
            self._learn_cpds_from_data(training_data)
        else:
            self._set_default_cpds()
        
        # Validate model
        if not self.model.check_model():
            raise ValueError("Invalid Bayesian network model")
        
        # Initialize inference
        self.inference_engine = VariableElimination(self.model)
        
        self.logger.info("Bayesian network built successfully with pgmpy")
        return True
    
    def _build_fallback_network(self, training_data: Optional[List[Dict]] = None) -> bool:
        """Build fallback network without pgmpy"""
        # Simple rule-based network for when pgmpy is not available
        self.model = "fallback"
        self.inference_engine = None
        
        self.logger.info("Using fallback Bayesian network implementation")
        return True
    
    def _set_default_cpds(self):
        """Set default Conditional Probability Distributions based on UX principles"""
        
        # User expertise (prior)
        cpd_expertise = TabularCPD(
            variable='user_expertise',
            variable_card=2,
            values=[[0.6], [0.4]]  # 60% novice, 40% expert
        )
        
        # User intent (prior)
        cpd_intent = TabularCPD(
            variable='user_intent',
            variable_card=2,
            values=[[0.7], [0.3]]  # 70% exploring, 30% goal-directed
        )
        
        # Element prominence (prior)
        cpd_prominence = TabularCPD(
            variable='element_prominence',
            variable_card=2,
            values=[[0.7], [0.3]]  # 70% low prominence, 30% high
        )
        
        # Element position (prior)
        cpd_position = TabularCPD(
            variable='element_position',
            variable_card=2,
            values=[[0.6], [0.4]]  # 60% peripheral, 40% central
        )
        
        # Task urgency depends on user intent
        cpd_urgency = TabularCPD(
            variable='task_urgency',
            variable_card=2,
            values=[
                [0.8, 0.3],  # Low urgency: 80% when exploring, 30% when goal-directed
                [0.2, 0.7]   # High urgency: 20% when exploring, 70% when goal-directed
            ],
            evidence=['user_intent'],
            evidence_card=[2]
        )
        
        # Element type depends on user expertise
        cpd_element_type = TabularCPD(
            variable='element_type',
            variable_card=4,
            values=[
                [0.4, 0.3],  # Button: 40% for novice, 30% for expert
                [0.2, 0.3],  # Link: 20% for novice, 30% for expert
                [0.3, 0.2],  # Text: 30% for novice, 20% for expert
                [0.1, 0.2]   # Form: 10% for novice, 20% for expert
            ],
            evidence=['user_expertise'],
            evidence_card=[2]
        )
        
        # Click decision depends on all factors
        # This is the most complex CPD but still manageable (2^6 = 64 combinations)
        click_values = self._generate_click_probabilities()
        
        cpd_click = TabularCPD(
            variable='will_click',
            variable_card=2,
            values=click_values,
            evidence=['user_expertise', 'user_intent', 'element_type', 
                     'element_prominence', 'element_position', 'task_urgency'],
            evidence_card=[2, 2, 4, 2, 2, 2]
        )
        
        # Add all CPDs to model
        self.model.add_cpds(cpd_expertise, cpd_intent, cpd_prominence, 
                           cpd_position, cpd_urgency, cpd_element_type, cpd_click)
    
    def _generate_click_probabilities(self) -> List[List[float]]:
        """Generate click probabilities based on UX principles"""
        # Evidence order: user_expertise, user_intent, element_type, 
        #                element_prominence, element_position, task_urgency
        
        click_probs = []
        
        # Iterate through all combinations
        for expertise in range(2):  # novice=0, expert=1
            for intent in range(2):  # exploring=0, goal_directed=1
                for elem_type in range(4):  # button=0, link=1, text=2, form=3
                    for prominence in range(2):  # low=0, high=1
                        for position in range(2):  # peripheral=0, central=1
                            for urgency in range(2):  # low=0, high=1
                                
                                # Base click probability
                                base_prob = 0.2
                                
                                # Element type effects
                                type_multipliers = [1.8, 1.2, 0.3, 0.8]  # button, link, text, form
                                base_prob *= type_multipliers[elem_type]
                                
                                # Prominence effect
                                if prominence == 1:  # high prominence
                                    base_prob *= 1.5
                                
                                # Position effect
                                if position == 1:  # central position
                                    base_prob *= 1.3
                                
                                # User expertise effect
                                if expertise == 0:  # novice
                                    if elem_type == 0:  # button
                                        base_prob *= 1.2  # Novices prefer buttons
                                    elif elem_type == 3:  # form
                                        base_prob *= 0.7  # Novices hesitant with forms
                                else:  # expert
                                    if elem_type == 1:  # link
                                        base_prob *= 1.3  # Experts comfortable with links
                                
                                # User intent effect
                                if intent == 1:  # goal-directed
                                    base_prob *= 1.4
                                
                                # Task urgency effect
                                if urgency == 1:  # high urgency
                                    base_prob *= 1.2
                                
                                # Cap probability
                                final_prob = min(0.9, base_prob)
                                
                                # Store as [P(no_click), P(click)]
                                click_probs.append([1.0 - final_prob, final_prob])
        
        # Transpose for pgmpy format
        return np.array(click_probs).T.tolist()
    
    def predict_click(self, element_features: Dict[str, Any], 
                     user_context: Dict[str, Any],
                     task_context: Dict[str, Any]) -> BayesianPrediction:
        """
        Predict click probability with full explanation
        
        Args:
            element_features: Features of the UI element
            user_context: User characteristics and context
            task_context: Task-related context
            
        Returns:
            BayesianPrediction with probability and explanations
        """
        try:
            # Map features to network states
            evidence = self._map_features_to_evidence(element_features, user_context, task_context)
            
            if PGMPY_AVAILABLE and self.inference_engine:
                return self._predict_with_pgmpy(evidence, element_features)
            else:
                return self._predict_with_fallback(evidence, element_features)
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._create_fallback_prediction(element_features)
    
    def _map_features_to_evidence(self, element_features: Dict[str, Any], 
                                 user_context: Dict[str, Any],
                                 task_context: Dict[str, Any]) -> Dict[str, int]:
        """Map input features to network evidence states"""
        evidence = {}
        
        # Map user expertise
        tech_savviness = user_context.get('tech_savviness', 0.5)
        evidence['user_expertise'] = 1 if tech_savviness > 0.6 else 0
        
        # Map user intent
        urgency = task_context.get('urgency_level', 0.5)
        evidence['user_intent'] = 1 if urgency > 0.5 else 0
        
        # Map element type
        element_type = element_features.get('element_type', 'text')
        type_mapping = {'button': 0, 'link': 1, 'text': 2, 'form': 3}
        evidence['element_type'] = type_mapping.get(element_type, 2)
        
        # Map element prominence
        prominence = element_features.get('prominence', 0.5)
        evidence['element_prominence'] = 1 if prominence > 0.5 else 0
        
        # Map element position
        position_features = element_features.get('position_features', {})
        center_distance = position_features.get('center_distance', 0.5)
        evidence['element_position'] = 1 if center_distance < 0.4 else 0  # central if close to center
        
        # Map task urgency
        evidence['task_urgency'] = 1 if urgency > 0.6 else 0
        
        return evidence
    
    def _predict_with_pgmpy(self, evidence: Dict[str, int], 
                           element_features: Dict[str, Any]) -> BayesianPrediction:
        """Make prediction using pgmpy inference"""
        
        # Run inference
        result = self.inference_engine.query(
            variables=['will_click'],
            evidence=evidence
        )
        
        click_probability = result.values[1]  # P(will_click = yes)
        
        # Calculate confidence based on probability extremeness
        confidence = 1.0 - (2.0 * abs(click_probability - 0.5))
        
        # Generate explanations
        explanation_components = self._generate_explanations(evidence, click_probability)
        reasoning_chain = self._generate_reasoning_chain(evidence, click_probability)
        
        return BayesianPrediction(
            element_id=element_features.get('id', 'unknown'),
            click_probability=float(click_probability),
            confidence=float(confidence),
            explanation_components=explanation_components,
            reasoning_chain=reasoning_chain
        )
    
    def _predict_with_fallback(self, evidence: Dict[str, int], 
                              element_features: Dict[str, Any]) -> BayesianPrediction:
        """Make prediction using fallback logic"""
        
        # Simple rule-based prediction
        base_prob = 0.3
        
        # Element type effect
        type_effects = [0.5, 0.3, -0.2, 0.1]  # button, link, text, form
        base_prob += type_effects[evidence['element_type']]
        
        # Prominence effect
        if evidence['element_prominence'] == 1:
            base_prob += 0.2
        
        # Position effect
        if evidence['element_position'] == 1:
            base_prob += 0.15
        
        # User factors
        if evidence['user_expertise'] == 0 and evidence['element_type'] == 0:  # novice + button
            base_prob += 0.1
        
        if evidence['user_intent'] == 1:  # goal-directed
            base_prob += 0.2
        
        click_probability = max(0.05, min(0.95, base_prob))
        confidence = 0.7  # Fixed confidence for fallback
        
        # Generate explanations
        explanation_components = self._generate_explanations(evidence, click_probability)
        reasoning_chain = self._generate_reasoning_chain(evidence, click_probability)
        
        return BayesianPrediction(
            element_id=element_features.get('id', 'unknown'),
            click_probability=click_probability,
            confidence=confidence,
            explanation_components=explanation_components,
            reasoning_chain=reasoning_chain
        )
    
    def _generate_explanations(self, evidence: Dict[str, int], 
                              click_probability: float) -> List[ExplanationComponent]:
        """Generate human-readable explanations for the prediction"""
        components = []
        
        # Analyze each evidence factor's contribution
        for factor, state_idx in evidence.items():
            if factor == 'will_click':  # Skip the target variable
                continue
            
            state_name = self.node_states[factor][state_idx]
            explanation_text = self.explanation_templates[factor][state_name]
            
            # Calculate influence score based on factor and probability
            influence_score = self._calculate_influence_score(factor, state_idx, click_probability)
            
            components.append(ExplanationComponent(
                factor_name=factor,
                factor_value=state_name,
                influence_score=influence_score,
                explanation_text=explanation_text
            ))
        
        # Sort by absolute influence score (most important first)
        components.sort(key=lambda x: abs(x.influence_score), reverse=True)
        
        return components
    
    def _calculate_influence_score(self, factor: str, state_idx: int, 
                                  click_probability: float) -> float:
        """Calculate how much each factor influences the prediction"""
        
        # Define influence patterns based on UX knowledge
        influence_patterns = {
            'element_type': {
                0: 0.4,   # button - strong positive
                1: 0.2,   # link - moderate positive
                2: -0.3,  # text - negative
                3: 0.1    # form - slight positive
            },
            'element_prominence': {
                0: -0.2,  # low prominence - negative
                1: 0.3    # high prominence - positive
            },
            'element_position': {
                0: -0.1,  # peripheral - slight negative
                1: 0.2    # central - positive
            },
            'user_expertise': {
                0: 0.0,   # novice - neutral baseline
                1: 0.1    # expert - slight positive (more likely to interact)
            },
            'user_intent': {
                0: -0.1,  # exploring - slight negative
                1: 0.3    # goal-directed - strong positive
            },
            'task_urgency': {
                0: -0.1,  # low urgency - slight negative
                1: 0.2    # high urgency - positive
            }
        }
        
        base_influence = influence_patterns.get(factor, {}).get(state_idx, 0.0)
        
        # Scale influence by how far the probability is from neutral
        probability_factor = abs(click_probability - 0.5) * 2
        
        return base_influence * probability_factor
    
    def _generate_reasoning_chain(self, evidence: Dict[str, int], 
                                 click_probability: float) -> List[str]:
        """Generate step-by-step reasoning chain"""
        reasoning = []
        
        # Start with element assessment
        element_type_idx = evidence.get('element_type', 2)
        element_type = self.node_states['element_type'][element_type_idx]
        reasoning.append(f"Starting with a {element_type} element")
        
        # Add prominence assessment
        prominence_idx = evidence.get('element_prominence', 0)
        prominence = self.node_states['element_prominence'][prominence_idx]
        if prominence == 'high':
            reasoning.append("Element has high visual prominence, increasing click likelihood")
        else:
            reasoning.append("Element has low visual prominence, reducing click likelihood")
        
        # Add position assessment
        position_idx = evidence.get('element_position', 0)
        position = self.node_states['element_position'][position_idx]
        if position == 'central':
            reasoning.append("Element is centrally positioned, which favors clicking")
        else:
            reasoning.append("Element is peripherally positioned, which reduces click probability")
        
        # Add user context
        intent_idx = evidence.get('user_intent', 0)
        intent = self.node_states['user_intent'][intent_idx]
        if intent == 'goal_directed':
            reasoning.append("User has a specific goal, increasing likelihood of focused interactions")
        else:
            reasoning.append("User is exploring, leading to more cautious interaction patterns")
        
        # Add final assessment
        if click_probability > 0.7:
            reasoning.append("Overall assessment: High likelihood of user clicking this element")
        elif click_probability > 0.4:
            reasoning.append("Overall assessment: Moderate likelihood of user clicking this element")
        else:
            reasoning.append("Overall assessment: Low likelihood of user clicking this element")
        
        return reasoning
    
    def _create_fallback_prediction(self, element_features: Dict[str, Any]) -> BayesianPrediction:
        """Create fallback prediction when inference fails"""
        return BayesianPrediction(
            element_id=element_features.get('id', 'unknown'),
            click_probability=0.5,
            confidence=0.1,
            explanation_components=[
                ExplanationComponent(
                    factor_name='error',
                    factor_value='inference_failed',
                    influence_score=0.0,
                    explanation_text="Prediction failed, using default probability"
                )
            ],
            reasoning_chain=["Inference engine encountered an error", 
                           "Using fallback probability of 50%"]
        )
    
    def get_network_structure(self) -> Dict[str, Any]:
        """Get human-readable network structure"""
        if not self.model:
            return {'error': 'Network not built'}
        
        if PGMPY_AVAILABLE and hasattr(self.model, 'nodes'):
            return {
                'nodes': list(self.model.nodes()),
                'edges': list(self.model.edges()),
                'node_states': self.node_states,
                'explanation_templates': self.explanation_templates
            }
        else:
            return {
                'type': 'fallback',
                'node_states': self.node_states,
                'explanation_templates': self.explanation_templates
            }
    
    def export_explanations(self, predictions: List[BayesianPrediction]) -> Dict[str, Any]:
        """Export explanations in a structured format for analysis"""
        
        explanations = []
        
        for pred in predictions:
            explanation = {
                'element_id': pred.element_id,
                'click_probability': pred.click_probability,
                'confidence': pred.confidence,
                'key_factors': [
                    {
                        'factor': comp.factor_name,
                        'value': comp.factor_value,
                        'influence': comp.influence_score,
                        'explanation': comp.explanation_text
                    }
                    for comp in pred.explanation_components[:3]  # Top 3 factors
                ],
                'reasoning_chain': pred.reasoning_chain
            }
            explanations.append(explanation)
        
        return {
            'predictions': explanations,
            'network_info': self.get_network_structure(),
            'summary': {
                'total_predictions': len(predictions),
                'avg_confidence': np.mean([p.confidence for p in predictions]),
                'high_confidence_predictions': len([p for p in predictions if p.confidence > 0.7])
            }
        }