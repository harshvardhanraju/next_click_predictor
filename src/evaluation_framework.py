import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports with error handling
try:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                               roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix)
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Advanced metrics will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Plotting libraries not available. Visualizations will be skipped.")


@dataclass
class PredictionResult:
    """Individual prediction result for evaluation"""
    element_id: str
    predicted_probability: float
    predicted_class: bool  # True if probability > threshold
    actual_class: bool
    confidence: float
    prediction_time: float
    element_features: Dict[str, Any]
    explanation_quality: float  # How good the explanation was (0-1)


@dataclass
class ModelMetrics:
    """Comprehensive metrics for a model"""
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    
    # Probability calibration metrics
    brier_score: float
    calibration_error: float
    
    # Ranking metrics
    average_precision: float
    ndcg_at_5: float
    
    # Efficiency metrics
    avg_prediction_time: float
    predictions_per_second: float
    
    # Explainability metrics
    avg_explanation_quality: float
    explanation_coverage: float
    
    # Additional statistics
    total_predictions: int
    positive_rate: float
    threshold_used: float


@dataclass
class ComparisonReport:
    """Comparison between different models/systems"""
    model_names: List[str]
    metrics: Dict[str, ModelMetrics]
    statistical_significance: Dict[str, Dict[str, float]]  # p-values
    improvement_summary: Dict[str, str]
    recommendations: List[str]


class EvaluationFramework:
    """
    Comprehensive evaluation framework for click prediction systems
    Supports multiple models, detailed metrics, and comparative analysis
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Evaluation configuration
        self.config = {
            'default_threshold': 0.5,
            'top_k_for_ranking': 5,
            'calibration_bins': 10,
            'bootstrap_samples': 1000,
            'significance_level': 0.05
        }
        
        # Results storage
        self.predictions_history = {}  # model_name -> List[PredictionResult]
        self.evaluation_cache = {}
    
    def add_predictions(self, model_name: str, predictions: List[PredictionResult]):
        """Add predictions from a model for evaluation"""
        if model_name not in self.predictions_history:
            self.predictions_history[model_name] = []
        
        self.predictions_history[model_name].extend(predictions)
        self.logger.info(f"Added {len(predictions)} predictions for model {model_name}")
        
        # Clear cache for this model
        if model_name in self.evaluation_cache:
            del self.evaluation_cache[model_name]
    
    def evaluate_model(self, model_name: str, threshold: float = None) -> ModelMetrics:
        """
        Comprehensive evaluation of a single model
        
        Args:
            model_name: Name of the model to evaluate
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            ModelMetrics with all computed metrics
        """
        if model_name not in self.predictions_history:
            raise ValueError(f"No predictions found for model {model_name}")
        
        threshold = threshold or self.config['default_threshold']
        cache_key = f"{model_name}_{threshold}"
        
        # Check cache
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        predictions = self.predictions_history[model_name]
        
        if not predictions:
            raise ValueError(f"No predictions available for model {model_name}")
        
        try:
            metrics = self._calculate_comprehensive_metrics(predictions, threshold)
            
            # Cache results
            self.evaluation_cache[cache_key] = metrics
            
            self.logger.info(f"Evaluated model {model_name}: Accuracy={metrics.accuracy:.3f}, F1={metrics.f1_score:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for model {model_name}: {e}")
            raise
    
    def _calculate_comprehensive_metrics(self, predictions: List[PredictionResult], 
                                       threshold: float) -> ModelMetrics:
        """Calculate all metrics for a set of predictions"""
        
        # Extract arrays for calculations
        y_true = np.array([p.actual_class for p in predictions])
        y_proba = np.array([p.predicted_probability for p in predictions])
        y_pred = y_proba >= threshold
        confidences = np.array([p.confidence for p in predictions])
        prediction_times = np.array([p.prediction_time for p in predictions])
        explanation_qualities = np.array([p.explanation_quality for p in predictions])
        
        # Basic classification metrics
        if SKLEARN_AVAILABLE:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            # ROC AUC (only if we have both classes)
            if len(np.unique(y_true)) > 1:
                auc_roc = roc_auc_score(y_true, y_proba)
                
                # PR AUC
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
                auc_pr = np.trapz(precision_curve, recall_curve)
            else:
                auc_roc = 0.5
                auc_pr = np.mean(y_true)  # Baseline for imbalanced data
        else:
            # Fallback calculations without sklearn
            accuracy = np.mean(y_true == y_pred)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            auc_roc = 0.5  # Placeholder
            auc_pr = np.mean(y_true)
        
        # Probability calibration metrics
        brier_score = np.mean((y_proba - y_true) ** 2)
        calibration_error = self._calculate_calibration_error(y_true, y_proba)
        
        # Ranking metrics
        average_precision = self._calculate_average_precision(y_true, y_proba)
        ndcg_at_5 = self._calculate_ndcg(y_true, y_proba, k=5)
        
        # Efficiency metrics
        avg_prediction_time = np.mean(prediction_times)
        predictions_per_second = 1.0 / avg_prediction_time if avg_prediction_time > 0 else 0
        
        # Explainability metrics
        avg_explanation_quality = np.mean(explanation_qualities)
        explanation_coverage = np.mean(explanation_qualities > 0.1)  # Fraction with decent explanations
        
        # Additional statistics
        total_predictions = len(predictions)
        positive_rate = np.mean(y_true)
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            brier_score=brier_score,
            calibration_error=calibration_error,
            average_precision=average_precision,
            ndcg_at_5=ndcg_at_5,
            avg_prediction_time=avg_prediction_time,
            predictions_per_second=predictions_per_second,
            avg_explanation_quality=avg_explanation_quality,
            explanation_coverage=explanation_coverage,
            total_predictions=total_predictions,
            positive_rate=positive_rate,
            threshold_used=threshold
        )
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        
        if SKLEARN_AVAILABLE:
            try:
                # Use sklearn's calibration_curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_proba, n_bins=self.config['calibration_bins']
                )
                
                # Calculate ECE
                bin_boundaries = np.linspace(0, 1, self.config['calibration_bins'] + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                    prop_in_bin = np.mean(in_bin)
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = np.mean(y_true[in_bin])
                        avg_confidence_in_bin = np.mean(y_proba[in_bin])
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                return ece
                
            except Exception:
                # Fallback to simple calculation
                pass
        
        # Simple binning-based ECE calculation
        bins = np.linspace(0, 1, self.config['calibration_bins'] + 1)
        ece = 0
        
        for i in range(len(bins) - 1):
            bin_mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_confidence = np.mean(y_proba[bin_mask])
                bin_weight = np.sum(bin_mask) / len(y_proba)
                ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
        
        return ece
    
    def _calculate_average_precision(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Average Precision (AP)"""
        
        if SKLEARN_AVAILABLE:
            try:
                from sklearn.metrics import average_precision_score
                return average_precision_score(y_true, y_proba)
            except Exception:
                pass
        
        # Fallback calculation
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate precision at each position
        precisions = []
        for i in range(1, len(y_true_sorted) + 1):
            precision_at_i = np.sum(y_true_sorted[:i]) / i
            precisions.append(precision_at_i)
        
        # Average precision is the average of precisions at positive positions
        positive_positions = np.where(y_true_sorted == 1)[0]
        if len(positive_positions) == 0:
            return 0.0
        
        return np.mean([precisions[pos] for pos in positive_positions])
    
    def _calculate_ndcg(self, y_true: np.ndarray, y_proba: np.ndarray, k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate DCG@k
        dcg = 0.0
        for i in range(min(k, len(y_true_sorted))):
            relevance = y_true_sorted[i]
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG@k (ideal DCG)
        ideal_sorted = np.sort(y_true)[::-1]  # Sort true labels in descending order
        idcg = 0.0
        for i in range(min(k, len(ideal_sorted))):
            relevance = ideal_sorted[i]
            idcg += relevance / np.log2(i + 2)
        
        # NDCG = DCG / IDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def compare_models(self, model_names: List[str], threshold: float = None) -> ComparisonReport:
        """
        Compare multiple models and generate comprehensive report
        
        Args:
            model_names: List of model names to compare
            threshold: Classification threshold for all models
            
        Returns:
            ComparisonReport with detailed comparison
        """
        if len(model_names) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        threshold = threshold or self.config['default_threshold']
        
        # Evaluate all models
        metrics = {}
        for model_name in model_names:
            metrics[model_name] = self.evaluate_model(model_name, threshold)
        
        # Statistical significance testing
        significance = self._calculate_statistical_significance(model_names, threshold)
        
        # Generate improvement summary
        improvement_summary = self._generate_improvement_summary(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, significance)
        
        return ComparisonReport(
            model_names=model_names,
            metrics=metrics,
            statistical_significance=significance,
            improvement_summary=improvement_summary,
            recommendations=recommendations
        )
    
    def _calculate_statistical_significance(self, model_names: List[str], 
                                          threshold: float) -> Dict[str, Dict[str, float]]:
        """Calculate statistical significance between model pairs"""
        
        significance = {}
        
        for i, model1 in enumerate(model_names):
            significance[model1] = {}
            
            for j, model2 in enumerate(model_names):
                if i == j:
                    significance[model1][model2] = 1.0  # Same model
                    continue
                
                try:
                    # Bootstrap test for statistical significance
                    p_value = self._bootstrap_test(model1, model2, threshold)
                    significance[model1][model2] = p_value
                    
                except Exception as e:
                    self.logger.warning(f"Significance test failed for {model1} vs {model2}: {e}")
                    significance[model1][model2] = 1.0  # Assume no significance
        
        return significance
    
    def _bootstrap_test(self, model1: str, model2: str, threshold: float, 
                       n_bootstrap: int = 1000) -> float:
        """Bootstrap test for statistical significance"""
        
        preds1 = self.predictions_history[model1]
        preds2 = self.predictions_history[model2]
        
        # Find common element IDs for fair comparison
        ids1 = {p.element_id for p in preds1}
        ids2 = {p.element_id for p in preds2}
        common_ids = ids1.intersection(ids2)
        
        if len(common_ids) < 10:
            return 1.0  # Not enough data for meaningful test
        
        # Extract paired predictions
        paired_preds1 = {p.element_id: p for p in preds1 if p.element_id in common_ids}
        paired_preds2 = {p.element_id: p for p in preds2 if p.element_id in common_ids}
        
        # Calculate observed difference in F1 scores
        f1_1 = self._calculate_f1_for_predictions([paired_preds1[id] for id in common_ids], threshold)
        f1_2 = self._calculate_f1_for_predictions([paired_preds2[id] for id in common_ids], threshold)
        observed_diff = f1_1 - f1_2
        
        # Bootstrap sampling
        bootstrap_diffs = []
        ids_list = list(common_ids)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_ids = np.random.choice(ids_list, size=len(ids_list), replace=True)
            
            bootstrap_preds1 = [paired_preds1[id] for id in bootstrap_ids]
            bootstrap_preds2 = [paired_preds2[id] for id in bootstrap_ids]
            
            bootstrap_f1_1 = self._calculate_f1_for_predictions(bootstrap_preds1, threshold)
            bootstrap_f1_2 = self._calculate_f1_for_predictions(bootstrap_preds2, threshold)
            
            bootstrap_diffs.append(bootstrap_f1_1 - bootstrap_f1_2)
        
        # Calculate p-value (two-tailed test)
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= observed_diff),
            np.mean(bootstrap_diffs <= observed_diff)
        )
        
        return p_value
    
    def _calculate_f1_for_predictions(self, predictions: List[PredictionResult], 
                                     threshold: float) -> float:
        """Calculate F1 score for a list of predictions"""
        if not predictions:
            return 0.0
        
        y_true = np.array([p.actual_class for p in predictions])
        y_pred = np.array([p.predicted_probability >= threshold for p in predictions])
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    def _generate_improvement_summary(self, metrics: Dict[str, ModelMetrics]) -> Dict[str, str]:
        """Generate summary of improvements between models"""
        
        summary = {}
        model_names = list(metrics.keys())
        
        if len(model_names) < 2:
            return summary
        
        # Find best model for each metric
        best_accuracy = max(model_names, key=lambda m: metrics[m].accuracy)
        best_f1 = max(model_names, key=lambda m: metrics[m].f1_score)
        best_auc = max(model_names, key=lambda m: metrics[m].auc_roc)
        best_speed = max(model_names, key=lambda m: metrics[m].predictions_per_second)
        best_explanation = max(model_names, key=lambda m: metrics[m].avg_explanation_quality)
        
        summary['best_accuracy'] = f"{best_accuracy} ({metrics[best_accuracy].accuracy:.3f})"
        summary['best_f1'] = f"{best_f1} ({metrics[best_f1].f1_score:.3f})"
        summary['best_auc'] = f"{best_auc} ({metrics[best_auc].auc_roc:.3f})"
        summary['best_speed'] = f"{best_speed} ({metrics[best_speed].predictions_per_second:.1f} preds/sec)"
        summary['best_explanation'] = f"{best_explanation} ({metrics[best_explanation].avg_explanation_quality:.3f})"
        
        # Calculate relative improvements
        baseline_model = model_names[0]  # Use first model as baseline
        
        for model_name in model_names[1:]:
            acc_improvement = ((metrics[model_name].accuracy - metrics[baseline_model].accuracy) / 
                             metrics[baseline_model].accuracy * 100)
            f1_improvement = ((metrics[model_name].f1_score - metrics[baseline_model].f1_score) / 
                            metrics[baseline_model].f1_score * 100)
            
            summary[f'{model_name}_vs_{baseline_model}'] = (
                f"Accuracy: {acc_improvement:+.1f}%, F1: {f1_improvement:+.1f}%"
            )
        
        return summary
    
    def _generate_recommendations(self, metrics: Dict[str, ModelMetrics], 
                                significance: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""
        
        recommendations = []
        model_names = list(metrics.keys())
        
        if len(model_names) < 2:
            recommendations.append("Need at least 2 models for meaningful comparison")
            return recommendations
        
        # Find the best overall model
        # Weighted score considering accuracy, F1, and explanation quality
        scores = {}
        for model_name in model_names:
            m = metrics[model_name]
            score = (0.4 * m.accuracy + 0.4 * m.f1_score + 0.2 * m.avg_explanation_quality)
            scores[model_name] = score
        
        best_model = max(scores.keys(), key=lambda k: scores[k])
        
        recommendations.append(f"Overall best model: {best_model} (composite score: {scores[best_model]:.3f})")
        
        # Check for significant differences
        significant_pairs = []
        for model1 in model_names:
            for model2 in model_names:
                if model1 != model2:
                    p_value = significance[model1][model2]
                    if p_value < self.config['significance_level']:
                        f1_diff = metrics[model1].f1_score - metrics[model2].f1_score
                        if f1_diff > 0:
                            significant_pairs.append(f"{model1} significantly outperforms {model2} (p < {p_value:.3f})")
        
        if significant_pairs:
            recommendations.append("Statistically significant improvements found:")
            recommendations.extend(significant_pairs)
        else:
            recommendations.append("No statistically significant differences found between models")
        
        # Specific recommendations based on metrics
        for model_name, m in metrics.items():
            model_recommendations = []
            
            if m.calibration_error > 0.1:
                model_recommendations.append(f"{model_name}: Consider probability calibration (ECE: {m.calibration_error:.3f})")
            
            if m.avg_explanation_quality < 0.5:
                model_recommendations.append(f"{model_name}: Improve explanation quality (current: {m.avg_explanation_quality:.3f})")
            
            if m.predictions_per_second < 1.0:
                model_recommendations.append(f"{model_name}: Optimize for speed (current: {m.predictions_per_second:.2f} pred/sec)")
            
            if m.precision < 0.6 and m.recall > 0.8:
                model_recommendations.append(f"{model_name}: High recall but low precision - consider increasing threshold")
            elif m.precision > 0.8 and m.recall < 0.6:
                model_recommendations.append(f"{model_name}: High precision but low recall - consider decreasing threshold")
            
            recommendations.extend(model_recommendations)
        
        return recommendations
    
    def generate_report(self, model_names: List[str], output_file: str = None) -> str:
        """Generate comprehensive evaluation report"""
        
        output_file = output_file or str(self.output_dir / "evaluation_report.json")
        
        # Get comprehensive comparison
        comparison = self.compare_models(model_names)
        
        # Generate detailed report
        report = {
            'evaluation_summary': {
                'models_evaluated': len(model_names),
                'total_predictions': sum(m.total_predictions for m in comparison.metrics.values()),
                'evaluation_timestamp': time.time(),
                'configuration': self.config
            },
            'model_metrics': {name: self._metrics_to_dict(metrics) 
                            for name, metrics in comparison.metrics.items()},
            'comparison_results': {
                'statistical_significance': comparison.statistical_significance,
                'improvement_summary': comparison.improvement_summary,
                'recommendations': comparison.recommendations
            },
            'detailed_analysis': self._generate_detailed_analysis(comparison)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation report saved to {output_file}")
        
        return output_file
    
    def _metrics_to_dict(self, metrics: ModelMetrics) -> Dict[str, Any]:
        """Convert ModelMetrics to dictionary"""
        return {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'auc_roc': metrics.auc_roc,
            'auc_pr': metrics.auc_pr,
            'brier_score': metrics.brier_score,
            'calibration_error': metrics.calibration_error,
            'average_precision': metrics.average_precision,
            'ndcg_at_5': metrics.ndcg_at_5,
            'avg_prediction_time': metrics.avg_prediction_time,
            'predictions_per_second': metrics.predictions_per_second,
            'avg_explanation_quality': metrics.avg_explanation_quality,
            'explanation_coverage': metrics.explanation_coverage,
            'total_predictions': metrics.total_predictions,
            'positive_rate': metrics.positive_rate,
            'threshold_used': metrics.threshold_used
        }
    
    def _generate_detailed_analysis(self, comparison: ComparisonReport) -> Dict[str, Any]:
        """Generate detailed analysis section"""
        
        analysis = {
            'performance_ranking': [],
            'trade_offs': {},
            'robustness_analysis': {},
            'deployment_considerations': []
        }
        
        # Performance ranking
        models_by_f1 = sorted(comparison.model_names, 
                             key=lambda m: comparison.metrics[m].f1_score, reverse=True)
        analysis['performance_ranking'] = [
            {
                'rank': i + 1,
                'model': model,
                'f1_score': comparison.metrics[model].f1_score,
                'accuracy': comparison.metrics[model].accuracy
            }
            for i, model in enumerate(models_by_f1)
        ]
        
        # Trade-offs analysis
        for model_name in comparison.model_names:
            m = comparison.metrics[model_name]
            analysis['trade_offs'][model_name] = {
                'accuracy_vs_speed': {
                    'accuracy': m.accuracy,
                    'speed': m.predictions_per_second,
                    'efficiency_score': m.accuracy * np.log1p(m.predictions_per_second)
                },
                'performance_vs_explainability': {
                    'f1_score': m.f1_score,
                    'explanation_quality': m.avg_explanation_quality,
                    'combined_score': 0.7 * m.f1_score + 0.3 * m.avg_explanation_quality
                }
            }
        
        # Deployment considerations
        for model_name in comparison.model_names:
            m = comparison.metrics[model_name]
            considerations = []
            
            if m.predictions_per_second > 10:
                considerations.append("Suitable for high-throughput applications")
            elif m.predictions_per_second < 1:
                considerations.append("May need optimization for real-time use")
            
            if m.avg_explanation_quality > 0.7:
                considerations.append("Provides high-quality explanations")
            elif m.avg_explanation_quality < 0.3:
                considerations.append("Limited explainability")
            
            if m.calibration_error < 0.05:
                considerations.append("Well-calibrated probabilities")
            elif m.calibration_error > 0.15:
                considerations.append("Probabilities may need calibration")
            
            analysis['deployment_considerations'].append({
                'model': model_name,
                'considerations': considerations
            })
        
        return analysis
    
    def visualize_results(self, model_names: List[str], output_dir: str = None):
        """Generate visualization plots for model comparison"""
        
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Plotting libraries not available, skipping visualizations")
            return
        
        output_dir = Path(output_dir) if output_dir else self.output_dir / "plots"
        output_dir.mkdir(exist_ok=True)
        
        # Get metrics for all models
        metrics = {name: self.evaluate_model(name) for name in model_names}
        
        # 1. Performance comparison bar chart
        self._plot_performance_comparison(metrics, output_dir / "performance_comparison.png")
        
        # 2. ROC curves (if sklearn available)
        if SKLEARN_AVAILABLE:
            self._plot_roc_curves(model_names, output_dir / "roc_curves.png")
        
        # 3. Calibration plots
        self._plot_calibration_curves(model_names, output_dir / "calibration_plots.png")
        
        # 4. Speed vs accuracy scatter plot
        self._plot_speed_accuracy_tradeoff(metrics, output_dir / "speed_accuracy_tradeoff.png")
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def _plot_performance_comparison(self, metrics: Dict[str, ModelMetrics], output_path: Path):
        """Plot performance comparison bar chart"""
        
        model_names = list(metrics.keys())
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC']
        
        data = []
        for metric_name in metric_names:
            values = []
            for model_name in model_names:
                m = metrics[model_name]
                if metric_name == 'Accuracy':
                    values.append(m.accuracy)
                elif metric_name == 'Precision':
                    values.append(m.precision)
                elif metric_name == 'Recall':
                    values.append(m.recall)
                elif metric_name == 'F1 Score':
                    values.append(m.f1_score)
                elif metric_name == 'AUC ROC':
                    values.append(m.auc_roc)
            data.append(values)
        
        # Create bar chart
        x = np.arange(len(model_names))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (metric_name, values) in enumerate(zip(metric_names, data)):
            offset = (i - len(metric_names) // 2) * width
            ax.bar(x + offset, values, width, label=metric_name)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, model_names: List[str], output_path: Path):
        """Plot ROC curves for all models"""
        
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            predictions = self.predictions_history[model_name]
            
            y_true = np.array([p.actual_class for p in predictions])
            y_proba = np.array([p.predicted_probability for p in predictions])
            
            if len(np.unique(y_true)) > 1:  # Need both classes for ROC
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc = roc_auc_score(y_true, y_proba)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curves(self, model_names: List[str], output_path: Path):
        """Plot calibration curves for all models"""
        
        fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5))
        if len(model_names) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            predictions = self.predictions_history[model_name]
            
            y_true = np.array([p.actual_class for p in predictions])
            y_proba = np.array([p.predicted_probability for p in predictions])
            
            if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
                fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
                
                axes[i].plot(mean_pred, fraction_pos, 's-', label=model_name)
                axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                axes[i].set_xlabel('Mean Predicted Probability')
                axes[i].set_ylabel('Fraction of Positives')
                axes[i].set_title(f'Calibration Curve - {model_name}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speed_accuracy_tradeoff(self, metrics: Dict[str, ModelMetrics], output_path: Path):
        """Plot speed vs accuracy trade-off"""
        
        model_names = list(metrics.keys())
        accuracies = [metrics[name].accuracy for name in model_names]
        speeds = [metrics[name].predictions_per_second for name in model_names]
        
        plt.figure(figsize=(10, 6))
        
        scatter = plt.scatter(speeds, accuracies, s=100, alpha=0.7)
        
        for i, model_name in enumerate(model_names):
            plt.annotate(model_name, (speeds[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Predictions per Second')
        plt.ylabel('Accuracy')
        plt.title('Speed vs Accuracy Trade-off')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()