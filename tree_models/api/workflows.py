"""High-level workflow orchestration functions.

This module contains complete end-to-end ML workflows that orchestrate
multiple components to provide comprehensive analysis pipelines.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.timer import timed_operation
from ..models.trainer import StandardModelTrainer
from ..models.evaluator import StandardModelEvaluator
from ..config.model_config import ModelConfig, XGBoostConfig, LightGBMConfig, CatBoostConfig
from ..utils.exceptions import ConfigurationError

logger = get_logger(__name__)


def complete_model_analysis(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight_train: Optional[np.ndarray] = None,
    sample_weight_test: Optional[np.ndarray] = None,
    config: Optional[ModelConfig] = None,
    n_trials: int = 100,
    scoring_function: str = "roc_auc",
    output_dir: str = "model_analysis",
    **kwargs: Any
) -> Dict[str, Any]:
    """Complete end-to-end model analysis pipeline.
    
    Performs hyperparameter tuning, model training, evaluation, explainability
    analysis, and robustness testing in a single function call.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        sample_weight_train: Optional training sample weights
        sample_weight_test: Optional test sample weights
        config: Optional model configuration
        n_trials: Number of hyperparameter tuning trials
        scoring_function: Primary scoring metric
        output_dir: Directory to save analysis results
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary containing all analysis results
        
    Example:
        >>> results = complete_model_analysis(
        ...     model_type='xgboost',
        ...     X_train=X_train, y_train=y_train,
        ...     X_test=X_test, y_test=y_test,
        ...     sample_weight_train=weights_train,
        ...     scoring_function='recall',
        ...     n_trials=50
        ... )
        >>> print(f"Best score: {results['tuning']['best_score']:.4f}")
    """
    pipeline = ModelAnalysisPipeline(
        model_type=model_type,
        config=config,
        n_trials=n_trials,
        scoring_function=scoring_function,
        output_dir=output_dir,
        **kwargs
    )
    return pipeline.run(X_train, y_train, X_test, y_test, sample_weight_train, sample_weight_test)


def fraud_detection_pipeline(
    model_type: str,
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight_train: Optional[np.ndarray] = None,
    sample_weight_test: Optional[np.ndarray] = None,
    focus: str = "recall",
    output_dir: str = "fraud_detection_analysis",
    **kwargs: Any
) -> Dict[str, Any]:
    """Specialized pipeline optimized for fraud detection use cases.
    
    Provides fraud-specific configurations, evaluation metrics, and
    analysis focused on maximizing fraud detection performance.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        X_train: Training features
        y_train: Training targets (0=legitimate, 1=fraud)
        X_test: Test features
        y_test: Test targets
        sample_weight_train: Optional training sample weights
        sample_weight_test: Optional test sample weights  
        focus: Optimization focus ('recall', 'precision', 'balanced', 'pr_auc')
        output_dir: Directory to save analysis results
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary containing fraud-specific analysis results
        
    Example:
        >>> results = fraud_detection_pipeline(
        ...     model_type='xgboost',
        ...     X_train=X_train, y_train=y_train,
        ...     X_test=X_test, y_test=y_test,
        ...     sample_weight_train=fraud_weights,
        ...     focus='recall'  # Maximize fraud detection
        ... )
    """
    # Map focus to appropriate scoring function
    focus_mapping = {
        'recall': 'recall',
        'precision': 'precision', 
        'balanced': 'f1',
        'pr_auc': 'average_precision'
    }
    
    if focus not in focus_mapping:
        raise ConfigurationError(f"Invalid focus '{focus}'. Must be one of: {list(focus_mapping.keys())}")
    
    scoring_function = focus_mapping[focus]
    
    logger.info(f"🕵️ Starting fraud detection pipeline (focus: {focus})")
    
    # Use fraud-specific configuration
    if model_type == 'xgboost':
        config = XGBoostConfig.for_fraud_detection()
    elif model_type == 'lightgbm':
        config = LightGBMConfig.for_fraud_detection()
    elif model_type == 'catboost':
        config = CatBoostConfig.for_fraud_detection()
    else:
        config = None
        logger.warning(f"No fraud-specific config available for {model_type}")
    
    # Run complete analysis with fraud-specific settings
    results = complete_model_analysis(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train, 
        X_test=X_test,
        y_test=y_test,
        sample_weight_train=sample_weight_train,
        sample_weight_test=sample_weight_test,
        config=config,
        scoring_function=scoring_function,
        output_dir=output_dir,
        **kwargs
    )
    
    # Add fraud-specific analysis
    results['fraud_analysis'] = {
        'focus': focus,
        'fraud_rate_train': float(y_train.mean()),
        'fraud_rate_test': float(y_test.mean()),
        'class_balance': {
            'legitimate': int((y_train == 0).sum()),
            'fraud': int((y_train == 1).sum())
        }
    }
    
    # Calculate fraud-specific metrics if we have probabilities
    if results['evaluation']['probabilities'] is not None:
        from sklearn.metrics import precision_recall_curve
        
        y_pred_proba = results['evaluation']['probabilities']
        
        # Precision-Recall curve analysis
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Find optimal threshold for different objectives
        # The last precision and recall values are 1. and 0. with no threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        optimal_f1_idx = np.argmax(f1_scores)

        results['fraud_analysis']['optimal_thresholds'] = {
            'f1_score': {
                'threshold': float(pr_thresholds[optimal_f1_idx]),
                'precision': float(precision[optimal_f1_idx]),
                'recall': float(recall[optimal_f1_idx]),
                'f1': float(f1_scores[optimal_f1_idx])
            }
        }
        
    logger.info(f"🎯 Fraud detection pipeline completed!")
    logger.info(f"   Focus: {focus} ({scoring_function})")
    logger.info(f"   Fraud rate: {results['fraud_analysis']['fraud_rate_test']:.2%}")
    
    return results


class ModelAnalysisPipeline:
    """Orchestrates complete model analysis workflow.
    
    This class breaks down the monolithic complete_model_analysis function
    into manageable, testable components while maintaining the same interface.
    """
    
    def __init__(
        self,
        model_type: str,
        config: Optional[ModelConfig] = None,
        n_trials: int = 100,
        scoring_function: str = "roc_auc",
        output_dir: str = "model_analysis",
        **kwargs: Any
    ) -> None:
        """Initialize the analysis pipeline."""
        self.model_type = model_type
        self.config = config
        self.n_trials = n_trials
        self.scoring_function = scoring_function
        self.output_dir = Path(output_dir)
        self.kwargs = kwargs
        self.results: Dict[str, Any] = {}
        
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sample_weight_train: Optional[np.ndarray] = None,
        sample_weight_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        
        logger.info(f"🚀 Starting complete model analysis for {self.model_type}")
        logger.info(f"   Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"   Testing: {X_test.shape[0]} samples")
        logger.info(f"   Scoring: {self.scoring_function}, Trials: {self.n_trials}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store metadata
        self.results['metadata'] = {
            'model_type': self.model_type,
            'scoring_function': self.scoring_function,
            'n_trials': self.n_trials,
            'output_dir': str(self.output_dir),
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'has_sample_weights': sample_weight_train is not None
        }
        
        try:
            # Step 1: Hyperparameter tuning
            self._run_hyperparameter_tuning(X_train, y_train, sample_weight_train)
            
            # Step 2: Model training
            self._run_model_training(X_train, y_train, sample_weight_train)
            
            # Step 3: Model evaluation  
            self._run_model_evaluation(X_test, y_test, sample_weight_test)
            
            # Step 4: Explainability analysis
            self._run_explainability_analysis(X_test, y_test, sample_weight_test)
            
            # Step 5: Robustness testing
            self._run_robustness_testing(X_train, y_train, sample_weight_train)
            
            # Save results summary
            self._save_results_summary()
            
            logger.info(f"🎉 Complete model analysis finished!")
            logger.info(f"   📁 Results saved to: {self.output_dir}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Complete model analysis failed: {e}")
            raise
    
    def _run_hyperparameter_tuning(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Execute hyperparameter tuning step."""
        logger.info("📊 Step 1/5: Hyperparameter tuning")
        
        with timed_operation("hyperparameter_tuning") as timing:
            # Import here to avoid circular imports
            from .quick_functions import tune_hyperparameters
            
            best_params, best_score = tune_hyperparameters(
                model_type=self.model_type,
                X=X, y=y,
                sample_weight=sample_weight,
                scoring_function=self.scoring_function,
                n_trials=self.n_trials,
                **self.kwargs
            )
        
        self.results['tuning'] = {
            'best_params': best_params,
            'best_score': best_score,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Best {self.scoring_function}: {best_score:.4f}")
    
    def _run_model_training(
        self,
        X: pd.DataFrame,
        y: pd.Series, 
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Execute model training step."""
        logger.info("🎯 Step 2/5: Final model training")
        
        with timed_operation("model_training") as timing:
            trainer = StandardModelTrainer(self.model_type, config=self.config)
            best_params = self.results['tuning']['best_params']
            model = trainer.get_model(best_params)
            
            if sample_weight is not None:
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
        
        self.results['training'] = {
            'model': model,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Model trained in {timing['duration']:.2f}s")
    
    def _run_model_evaluation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Execute model evaluation step."""
        logger.info("📈 Step 3/5: Model evaluation")
        
        with timed_operation("model_evaluation") as timing:
            evaluator = StandardModelEvaluator()
            model = self.results['training']['model']
            eval_result = evaluator.evaluate(model, X, y, sample_weight)
        
        self.results['evaluation'] = {
            'metrics': eval_result.metrics,
            'predictions': eval_result.predictions,
            'probabilities': eval_result.probabilities,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Test {self.scoring_function}: {eval_result.metrics.get(self.scoring_function, 'N/A')}")
    
    def _run_explainability_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Execute explainability analysis step."""
        logger.info("🔍 Step 4/5: Explainability analysis")
        
        with timed_operation("explainability") as timing:
            # Import here to avoid circular imports
            from .quick_functions import quick_shap_analysis, convert_to_scorecard
            
            model = self.results['training']['model']
            
            # SHAP analysis (limit to 1000 samples for performance)
            X_sample = X.iloc[:1000] if len(X) > 1000 else X
            sample_weight_sample = sample_weight[:1000] if sample_weight is not None and len(X) > 1000 else sample_weight
            
            shap_results = quick_shap_analysis(
                model, X_sample,
                sample_weight=sample_weight_sample,
                save_dir=self.output_dir / "explainability"
            )
            
            # Scorecard conversion
            y_pred_proba = self.results['evaluation']['probabilities']
            if y_pred_proba is not None:
                scores, converter = convert_to_scorecard(y_pred_proba, sample_weight)
            else:
                scores, converter = None, None
        
        self.results['explainability'] = {
            'shap_results': shap_results,
            'scorecard_scores': scores,
            'scorecard_converter': converter,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Explainability completed")
    
    def _run_robustness_testing(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Execute robustness testing step."""
        logger.info("🛡️ Step 5/5: Robustness testing")
        
        with timed_operation("robustness") as timing:
            # Import here to avoid circular imports
            from .quick_functions import quick_robustness_test
            
            best_params = self.results['tuning']['best_params']
            
            # Sample data for performance (max 5000 samples)
            if len(X) > 5000:
                X_sample = X.iloc[:5000]
                y_sample = y.iloc[:5000]
                sample_weight_sample = sample_weight[:5000] if sample_weight is not None else None
            else:
                X_sample = X
                y_sample = y
                sample_weight_sample = sample_weight
            
            robustness_results = quick_robustness_test(
                model_type=self.model_type,
                best_params=best_params,
                X=X_sample,
                y=y_sample,
                sample_weight=sample_weight_sample,
                n_seeds=5,
                scoring_function=self.scoring_function
            )
        
        self.results['robustness'] = {
            **robustness_results,
            'duration': timing['duration']
        }
        logger.info(f"   ✅ Robustness testing completed")
    
    def _save_results_summary(self) -> None:
        """Save analysis results summary."""
        import json
        
        summary_path = self.output_dir / "analysis_summary.json"
        summary = {
            'metadata': self.results['metadata'],
            'tuning_summary': {
                'best_score': self.results['tuning']['best_score'],
                'best_params': self.results['tuning']['best_params']
            },
            'evaluation_summary': self.results['evaluation']['metrics'],
            'robustness_summary': {
                k: v for k, v in self.results['robustness'].items() 
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"   📋 Summary saved to: {summary_path}")