import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import itertools
import logging
from feature_integrator import IntegratedFeatures

# Optional imports with error handling
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    try:
        # Try older pgmpy version
        from pgmpy.models import BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
        PGMPY_AVAILABLE = True
    except ImportError:
        try:
            # Try even older pgmpy version
            from pgmpy.models import BayesianModel as BayesianNetwork
            from pgmpy.factors.discrete import TabularCPD
            from pgmpy.inference import VariableElimination
            PGMPY_AVAILABLE = True
        except ImportError:
            PGMPY_AVAILABLE = False
            logging.warning("pgmpy not available. Bayesian network functionality will be limited.")
            
            # Create dummy classes for when pgmpy is not available
            class BayesianNetwork:
                def __init__(self, *args, **kwargs):
                    pass
                
            class TabularCPD:
                def __init__(self, *args, **kwargs):
                    pass
                
            class VariableElimination:
                def __init__(self, *args, **kwargs):
                    pass


@dataclass
class NetworkNode:
    """Represents a node in the Bayesian network"""
    name: str
    node_type: str  # 'user', 'task', 'ui', 'decision'
    states: List[str]
    parents: List[str]
    cpd: TabularCPD = None


class BayesianNetworkEngine:
    """
    Dynamically constructs and manages Bayesian networks for click prediction
    """
    
    def __init__(self):
        self.model = None
        self.inference_engine = None
        self.node_definitions = self._initialize_node_definitions()
        self.cpd_templates = self._initialize_cpd_templates()
        
    def build_network(self, integrated_features: IntegratedFeatures) -> BayesianNetwork:
        """
        Dynamically build Bayesian network based on integrated features
        
        Args:
            integrated_features: Combined feature set from FeatureIntegrator
            
        Returns:
            Constructed Bayesian network model
        """
        if not PGMPY_AVAILABLE:
            logging.warning("pgmpy not available, using fallback network")
            self.model = BayesianNetwork()
            return self.model
        
        try:
            # Define network structure
            network_structure = self._define_network_structure(integrated_features)
            
            # Create Bayesian model
            self.model = BayesianNetwork(network_structure)
            
            # Add CPDs (Conditional Probability Distributions)
            self._add_cpds(integrated_features)
            
            # Validate model
            if hasattr(self.model, 'check_model') and not self.model.check_model():
                raise ValueError("Invalid Bayesian network model")
            
            # Initialize inference engine
            self.inference_engine = VariableElimination(self.model)
            
            return self.model
        except Exception as e:
            logging.error(f"Failed to build Bayesian network: {e}")
            self.model = BayesianNetwork()
            return self.model
    
    def _define_network_structure(self, integrated_features: IntegratedFeatures) -> List[Tuple[str, str]]:
        """Define the structure (edges) of the Bayesian network"""
        edges = []
        
        # User attribute nodes
        user_nodes = ['user_tech_savviness', 'user_mood', 'user_experience']
        
        # Task context nodes
        task_nodes = ['task_type', 'task_urgency', 'task_stage']
        
        # UI element nodes (one per element)
        ui_nodes = []
        for i, element in enumerate(integrated_features.ui_features):
            element_id = element.get('id', f'element_{i}')
            ui_nodes.append(f'ui_{element_id}_prominence')
            ui_nodes.append(f'ui_{element_id}_type')
            ui_nodes.append(f'ui_{element_id}_position')
        
        # Decision nodes (one per UI element)
        decision_nodes = []
        for i, element in enumerate(integrated_features.ui_features):
            element_id = element.get('id', f'element_{i}')
            decision_node = f'click_{element_id}'
            decision_nodes.append(decision_node)
            
            # Connect user attributes to decision
            for user_node in user_nodes:
                edges.append((user_node, decision_node))
            
            # Connect task context to decision
            for task_node in task_nodes:
                edges.append((task_node, decision_node))
            
            # Connect relevant UI features to decision
            edges.append((f'ui_{element_id}_prominence', decision_node))
            edges.append((f'ui_{element_id}_type', decision_node))
            edges.append((f'ui_{element_id}_position', decision_node))
        
        # Add interaction nodes
        edges.append(('user_tech_savviness', 'user_task_compatibility'))
        edges.append(('task_type', 'user_task_compatibility'))
        edges.append(('task_urgency', 'user_task_compatibility'))
        
        # Connect interaction nodes to decisions
        for decision_node in decision_nodes:
            edges.append(('user_task_compatibility', decision_node))
        
        return edges
    
    def _add_cpds(self, integrated_features: IntegratedFeatures):
        """Add Conditional Probability Distributions to the network"""
        
        # Add user attribute CPDs
        self._add_user_cpds(integrated_features)
        
        # Add task context CPDs
        self._add_task_cpds(integrated_features)
        
        # Add UI element CPDs
        self._add_ui_cpds(integrated_features)
        
        # Add interaction CPDs
        self._add_interaction_cpds(integrated_features)
        
        # Add decision CPDs
        self._add_decision_cpds(integrated_features)
    
    def _add_user_cpds(self, integrated_features: IntegratedFeatures):
        """Add CPDs for user attribute nodes"""
        user_features = integrated_features.user_features
        
        # User tech savviness
        tech_savviness_values = self._discretize_value(user_features.get('tech_savviness', 0.5))
        cpd_tech = TabularCPD(
            variable='user_tech_savviness',
            variable_card=3,
            values=self._get_prior_distribution(tech_savviness_values, ['low', 'medium', 'high'])
        )
        self.model.add_cpds(cpd_tech)
        
        # User mood
        mood_values = self._discretize_value(user_features.get('mood', 0.5))
        cpd_mood = TabularCPD(
            variable='user_mood',
            variable_card=3,
            values=self._get_prior_distribution(mood_values, ['negative', 'neutral', 'positive'])
        )
        self.model.add_cpds(cpd_mood)
        
        # User experience (derived)
        experience_values = self._discretize_value(user_features.get('experience_level', 0.5))
        cpd_experience = TabularCPD(
            variable='user_experience',
            variable_card=3,
            values=self._get_prior_distribution(experience_values, ['low', 'medium', 'high'])
        )
        self.model.add_cpds(cpd_experience)
    
    def _add_task_cpds(self, integrated_features: IntegratedFeatures):
        """Add CPDs for task context nodes"""
        task_features = integrated_features.task_features
        
        # Task type
        task_type_values = self._discretize_value(task_features.get('task_type', 0.5))
        cpd_task_type = TabularCPD(
            variable='task_type',
            variable_card=3,
            values=self._get_prior_distribution(task_type_values, ['browse', 'action', 'purchase'])
        )
        self.model.add_cpds(cpd_task_type)
        
        # Task urgency
        urgency_values = self._discretize_value(task_features.get('urgency_level', 0.5))
        cpd_urgency = TabularCPD(
            variable='task_urgency',
            variable_card=3,
            values=self._get_prior_distribution(urgency_values, ['low', 'medium', 'high'])
        )
        self.model.add_cpds(cpd_urgency)
        
        # Task stage
        stage_values = self._discretize_value(task_features.get('completion_stage', 0.5))
        cpd_stage = TabularCPD(
            variable='task_stage',
            variable_card=3,
            values=self._get_prior_distribution(stage_values, ['start', 'middle', 'end'])
        )
        self.model.add_cpds(cpd_stage)
    
    def _add_ui_cpds(self, integrated_features: IntegratedFeatures):
        """Add CPDs for UI element nodes"""
        for i, element in enumerate(integrated_features.ui_features):
            element_id = element.get('id', f'element_{i}')
            
            # UI element prominence
            prominence = element.get('prominence', 0.5)
            prominence_values = self._discretize_value(prominence)
            cpd_prominence = TabularCPD(
                variable=f'ui_{element_id}_prominence',
                variable_card=3,
                values=self._get_prior_distribution(prominence_values, ['low', 'medium', 'high'])
            )
            self.model.add_cpds(cpd_prominence)
            
            # UI element type
            element_type = element.get('type', 'unknown')
            type_values = self._encode_element_type_for_cpd(element_type)
            cpd_type = TabularCPD(
                variable=f'ui_{element_id}_type',
                variable_card=3,
                values=self._get_prior_distribution(type_values, ['passive', 'interactive', 'action'])
            )
            self.model.add_cpds(cpd_type)
            
            # UI element position
            position = element.get('position_features', {}).get('center_distance', 0.5)
            position_values = self._discretize_value(1.0 - position)  # Invert so center = high
            cpd_position = TabularCPD(
                variable=f'ui_{element_id}_position',
                variable_card=3,
                values=self._get_prior_distribution(position_values, ['edge', 'mid', 'center'])
            )
            self.model.add_cpds(cpd_position)
    
    def _add_interaction_cpds(self, integrated_features: IntegratedFeatures):
        """Add CPDs for interaction nodes"""
        
        # User-task compatibility
        cpd_compatibility = TabularCPD(
            variable='user_task_compatibility',
            variable_card=2,
            values=self._get_user_task_compatibility_cpd(),
            evidence=['user_tech_savviness', 'task_type', 'task_urgency'],
            evidence_card=[3, 3, 3]
        )
        self.model.add_cpds(cpd_compatibility)
    
    def _add_decision_cpds(self, integrated_features: IntegratedFeatures):
        """Add CPDs for decision nodes (click predictions)"""
        
        for i, element in enumerate(integrated_features.ui_features):
            element_id = element.get('id', f'element_{i}')
            decision_node = f'click_{element_id}'
            
            # Get CPD values based on element characteristics
            cpd_values = self._get_decision_cpd(element)
            
            cpd_decision = TabularCPD(
                variable=decision_node,
                variable_card=2,  # click / no_click
                values=cpd_values,
                evidence=[
                    'user_tech_savviness', 'user_mood', 'user_experience',
                    'task_type', 'task_urgency', 'task_stage',
                    f'ui_{element_id}_prominence', f'ui_{element_id}_type', f'ui_{element_id}_position',
                    'user_task_compatibility'
                ],
                evidence_card=[3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
            )
            self.model.add_cpds(cpd_decision)
    
    def _discretize_value(self, value: float) -> int:
        """Convert continuous value to discrete category"""
        if value < 0.33:
            return 0
        elif value < 0.67:
            return 1
        else:
            return 2
    
    def _get_prior_distribution(self, observed_value: int, states: List[str]) -> List[List[float]]:
        """Get prior probability distribution with bias toward observed value"""
        probs = [0.1, 0.1, 0.1]  # Base probabilities
        probs[observed_value] = 0.8  # High probability for observed state
        return [[prob] for prob in probs]
    
    def _encode_element_type_for_cpd(self, element_type: str) -> int:
        """Encode element type for CPD"""
        type_mapping = {
            'text': 0,      # passive
            'image': 0,     # passive
            'link': 1,      # interactive
            'menu': 1,      # interactive
            'form': 1,      # interactive
            'button': 2     # action
        }
        return type_mapping.get(element_type, 1)
    
    def _get_user_task_compatibility_cpd(self) -> List[List[float]]:
        """Get CPD for user-task compatibility"""
        # Create CPD table for user_task_compatibility
        # Evidence: user_tech_savviness(3) x task_type(3) x task_urgency(3) = 27 combinations
        
        cpd_table = []
        
        # Generate all combinations of evidence states
        for tech in range(3):      # low, medium, high
            for task in range(3):   # browse, action, purchase
                for urgency in range(3):  # low, medium, high
                    
                    # Calculate compatibility based on heuristics
                    compatibility = 0.5  # base compatibility
                    
                    # High tech + complex task = good compatibility
                    if tech == 2 and task == 2:  # high tech + purchase
                        compatibility = 0.8
                    
                    # Low tech + simple task = good compatibility
                    if tech == 0 and task == 0:  # low tech + browse
                        compatibility = 0.7
                    
                    # High urgency requires higher tech savviness
                    if urgency == 2 and tech < 2:  # high urgency + low/medium tech
                        compatibility *= 0.7
                    
                    # Add to table: [incompatible, compatible]
                    cpd_table.append([1.0 - compatibility, compatibility])
        
        return np.array(cpd_table).T.tolist()
    
    def _get_decision_cpd(self, element: Dict[str, Any]) -> List[List[float]]:
        """Get CPD for click decision based on element characteristics"""
        
        # Total combinations: 3^9 * 2 = 39,366 combinations
        # This is computationally intensive, so we'll use heuristics
        
        base_click_prob = 0.1  # Base probability of clicking any element
        
        # Adjust based on element characteristics
        element_type = element.get('type', 'unknown')
        prominence = element.get('prominence', 0.5)
        text_analysis = element.get('text_analysis', {})
        
        # Type-based adjustment
        type_multiplier = {
            'button': 3.0,
            'link': 2.0,
            'form': 1.5,
            'menu': 1.2,
            'text': 0.5,
            'image': 0.8
        }.get(element_type, 1.0)
        
        # Prominence-based adjustment
        prominence_multiplier = 1.0 + (prominence * 2.0)
        
        # Text-based adjustment
        text_multiplier = 1.0
        if text_analysis.get('action_words', 0) > 0:
            text_multiplier += 0.5
        if text_analysis.get('urgency_words', 0) > 0:
            text_multiplier += 0.3
        
        # Calculate adjusted probability
        adjusted_prob = base_click_prob * type_multiplier * prominence_multiplier * text_multiplier
        adjusted_prob = min(0.9, adjusted_prob)  # Cap at 90%
        
        # Create simplified CPD (we'll use the same probability for all combinations)
        # In a real implementation, this would vary based on evidence combinations
        total_combinations = 3**9 * 2  # 39,366
        
        cpd_values = []
        for _ in range(total_combinations):
            cpd_values.append([1.0 - adjusted_prob, adjusted_prob])
        
        return np.array(cpd_values).T.tolist()
    
    def predict_clicks(self, integrated_features: IntegratedFeatures) -> List[Dict[str, Any]]:
        """
        Predict click probabilities for all UI elements
        
        Args:
            integrated_features: Integrated feature set
            
        Returns:
            List of click predictions with probabilities
        """
        if not PGMPY_AVAILABLE or not self.model or not self.inference_engine:
            logging.warning("Bayesian network not available, using fallback predictions")
            return self._fallback_predictions(integrated_features)
        
        predictions = []
        
        try:
            # Set evidence based on integrated features
            evidence = self._prepare_evidence(integrated_features)
            
            # Query each decision node
            for i, element in enumerate(integrated_features.ui_features):
                element_id = element.get('id', f'element_{i}')
                decision_node = f'click_{element_id}'
                
                try:
                    # Run inference
                    result = self.inference_engine.query(
                        variables=[decision_node],
                        evidence=evidence
                    )
                    
                    # Extract probability
                    click_prob = result.values[1]  # probability of clicking
                    
                    predictions.append({
                        'element_id': element_id,
                        'element_type': element.get('type', 'unknown'),
                        'element_text': element.get('text', ''),
                        'click_probability': float(click_prob),
                        'confidence': self._calculate_confidence(click_prob),
                        'prominence': element.get('prominence', 0.0)
                    })
                    
                except Exception as e:
                    # Fallback to heuristic-based prediction
                    predictions.append({
                        'element_id': element_id,
                        'element_type': element.get('type', 'unknown'),
                        'element_text': element.get('text', ''),
                        'click_probability': element.get('prominence', 0.5),
                        'confidence': 0.5,
                        'prominence': element.get('prominence', 0.0),
                        'fallback_used': True
                    })
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return self._fallback_predictions(integrated_features)
        
        # Sort by probability (descending)
        predictions.sort(key=lambda x: x['click_probability'], reverse=True)
        
        # Add ranks
        for i, pred in enumerate(predictions):
            pred['rank'] = i + 1
        
        return predictions
    
    def _fallback_predictions(self, integrated_features: IntegratedFeatures) -> List[Dict[str, Any]]:
        """Generate fallback predictions when Bayesian network is not available"""
        predictions = []
        
        for i, element in enumerate(integrated_features.ui_features):
            element_id = element.get('id', f'element_{i}')
            
            # Simple heuristic: use prominence as probability
            base_prob = element.get('prominence', 0.5)
            
            # Adjust based on element type
            type_multiplier = {
                'button': 1.5,
                'link': 1.2,
                'form': 1.0,
                'text': 0.3,
                'image': 0.7
            }.get(element.get('type', 'unknown'), 1.0)
            
            click_prob = min(0.9, base_prob * type_multiplier)
            
            predictions.append({
                'element_id': element_id,
                'element_type': element.get('type', 'unknown'),
                'element_text': element.get('text', ''),
                'click_probability': click_prob,
                'confidence': 0.6,
                'prominence': element.get('prominence', 0.0),
                'fallback_used': True
            })
        
        # Sort by probability (descending)
        predictions.sort(key=lambda x: x['click_probability'], reverse=True)
        
        # Add ranks
        for i, pred in enumerate(predictions):
            pred['rank'] = i + 1
        
        return predictions
    
    def _prepare_evidence(self, integrated_features: IntegratedFeatures) -> Dict[str, int]:
        """Prepare evidence dictionary for inference"""
        evidence = {}
        
        # User evidence
        user_features = integrated_features.user_features
        evidence['user_tech_savviness'] = self._discretize_value(user_features.get('tech_savviness', 0.5))
        evidence['user_mood'] = self._discretize_value(user_features.get('mood', 0.5))
        evidence['user_experience'] = self._discretize_value(user_features.get('experience_level', 0.5))
        
        # Task evidence
        task_features = integrated_features.task_features
        evidence['task_type'] = self._discretize_value(task_features.get('task_type', 0.5))
        evidence['task_urgency'] = self._discretize_value(task_features.get('urgency_level', 0.5))
        evidence['task_stage'] = self._discretize_value(task_features.get('completion_stage', 0.5))
        
        # UI evidence for each element
        for i, element in enumerate(integrated_features.ui_features):
            element_id = element.get('id', f'element_{i}')
            
            prominence = element.get('prominence', 0.5)
            evidence[f'ui_{element_id}_prominence'] = self._discretize_value(prominence)
            
            element_type = element.get('type', 'unknown')
            evidence[f'ui_{element_id}_type'] = self._encode_element_type_for_cpd(element_type)
            
            position = element.get('position_features', {}).get('center_distance', 0.5)
            evidence[f'ui_{element_id}_position'] = self._discretize_value(1.0 - position)
        
        # Interaction evidence
        interaction_features = integrated_features.interaction_features
        compatibility = interaction_features.get('user_task_compatibility', 0.5)
        evidence['user_task_compatibility'] = 1 if compatibility > 0.5 else 0
        
        return evidence
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence score based on probability"""
        # High confidence when probability is very high or very low
        # Low confidence when probability is around 0.5
        return 1.0 - (2.0 * abs(probability - 0.5))
    
    def get_network_structure(self) -> Dict[str, Any]:
        """Get summary of network structure for debugging"""
        if not self.model:
            return {'error': 'Network not built'}
        
        return {
            'nodes': list(self.model.nodes()),
            'edges': list(self.model.edges()),
            'num_nodes': len(self.model.nodes()),
            'num_edges': len(self.model.edges()),
            'cpd_count': len(self.model.get_cpds())
        }
    
    def _initialize_node_definitions(self) -> Dict[str, Any]:
        """Initialize standard node definitions"""
        return {
            'user_nodes': {
                'user_tech_savviness': ['low', 'medium', 'high'],
                'user_mood': ['negative', 'neutral', 'positive'],
                'user_experience': ['low', 'medium', 'high']
            },
            'task_nodes': {
                'task_type': ['browse', 'action', 'purchase'],
                'task_urgency': ['low', 'medium', 'high'],
                'task_stage': ['start', 'middle', 'end']
            },
            'ui_nodes': {
                'prominence': ['low', 'medium', 'high'],
                'type': ['passive', 'interactive', 'action'],
                'position': ['edge', 'mid', 'center']
            },
            'interaction_nodes': {
                'user_task_compatibility': ['incompatible', 'compatible']
            }
        }
    
    def _initialize_cpd_templates(self) -> Dict[str, Any]:
        """Initialize CPD templates for common patterns"""
        return {
            'high_tech_high_urgency': {
                'button_click_prob': 0.8,
                'link_click_prob': 0.6,
                'form_click_prob': 0.5
            },
            'low_tech_low_urgency': {
                'button_click_prob': 0.6,
                'link_click_prob': 0.3,
                'form_click_prob': 0.2
            }
        }