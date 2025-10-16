"""Enhanced models module for tree-based machine learning.

This module provides comprehensive machine learning capabilities including:
- Advanced hyperparameter tuning with custom scoring
- Model explainability and interpretability tools
- Model robustness and stability testing
- Sample weights integration throughout

Key Features:
- Custom scoring functions and direction control
- SHAP explainability with scorecard conversion
- Seed robustness and sensitivity analysis
- Distribution drift detection (PSI)
- Comprehensive reporting and visualization

Basic Usage:
    >>> from tree_model_helper.models import tune_hyperparameters, ScoringConfig
    >>> 
    >>> # Quick tuning with custom scoring
    >>> best_params, best_score = tune_hyperparameters(
    ...     model_type='xgboost',
    ...     X=X, y=y, sample_weight=weights,
    ...     scoring_function='recall',
    ...     additional_metrics=['precision', 'f1', 'roc_auc'],
    ...     n_trials=100
    ... )

Advanced Usage:
    >>> from tree_model_helper.models import ModelExplainerSuite, RobustnessSuite
    >>> 
    >>> # Generate comprehensive explainability report
    >>> explainer = ModelExplainerSuite(model, X_train, y_train)
    >>> report = explainer.generate_full_report(X_test, sample_weight=weights)
    >>> 
    >>> # Run comprehensive robustness testing
    >>> robustness = RobustnessSuite(model_class=xgb.XGBClassifier, model_params=params)
    >>> results = robustness.run_full_robustness_test(X_train, y_train, X_test, y_test)
"""

# Hyperparameter tuning
from .tuner import (
    # Main tuner classes
    EnhancedOptunaHyperparameterTuner,
    OptunaHyperparameterTuner,  # Legacy alias
    MultiModelTuner,

    # Configuration classes
    ScoringConfig,
    CustomScorer,

    # Convenience functions
    tune_hyperparameters,
    compare_models,
)

# Model explainability
from .explain import (
    # Main explainer classes
    ModelExplainerSuite,
    SHAPExplainer,
    ScorecardConverter,
    ReasonCodeGenerator,
    PartialDependencePlotter,

    # Configuration classes
    ScorecardConfig,

    # Convenience functions
    quick_shap_analysis,
    convert_to_scorecard,
)

# Model robustness testing
from .robustness import (
    # Main robustness classes
    SeedRobustnessTester,
    SensitivityAnalyzer,
    PopulationStabilityIndex,

    # Configuration classes
    RobustnessConfig,

    # Convenience functions
    quick_robustness_test,
    calculate_psi_simple,
)

# Version and metadata
__version__ = '0.3.0'
__author__ = 'Tree Model Helper Enhanced'

# Main exports for easy access
__all__ = [
    # Hyperparameter tuning
    'EnhancedOptunaHyperparameterTuner',
    'OptunaHyperparameterTuner',
    'MultiModelTuner',
    'ScoringConfig',
    'CustomScorer',
    'tune_hyperparameters',
    'compare_models',

    # Model explainability
    'ModelExplainerSuite',
    'SHAPExplainer',
    'ScorecardConverter',
    'ReasonCodeGenerator',
    'PartialDependencePlotter',
    'ScorecardConfig',
    'quick_shap_analysis',
    'convert_to_scorecard',

    # Model robustness
    'SeedRobustnessTester',
    'SensitivityAnalyzer',
    'PopulationStabilityIndex',
    'RobustnessConfig',
    'quick_robustness_test',
    'calculate_psi_simple',
]

# Quick access to common configurations
class CommonScoringConfigs:
    """Pre-configured scoring setups for common use cases."""

    @staticmethod
    def fraud_detection_recall():
        """Optimize for maximum fraud detection (recall)."""
        return ScoringConfig(
            scoring_function='recall',
            direction='maximize',
            additional_metrics=['precision', 'f1', 'roc_auc', 'average_precision'],
            run_full_evaluation=True
        )

    @staticmethod
    def fraud_detection_precision():
        """Optimize for minimum false alarms (precision)."""
        return ScoringConfig(
            scoring_function='precision',
            direction='maximize',
            additional_metrics=['recall', 'f1', 'roc_auc', 'average_precision'],
            run_full_evaluation=True
        )

    @staticmethod
    def fraud_detection_balanced():
        """Balanced fraud detection optimization (F1)."""
        return ScoringConfig(
            scoring_function='f1',
            direction='maximize',
            additional_metrics=['precision', 'recall', 'roc_auc', 'average_precision'],
            run_full_evaluation=True
        )

    @staticmethod
    def fraud_detection_pr_auc():
        """Optimize for PR-AUC (best for imbalanced datasets)."""
        return ScoringConfig(
            scoring_function='average_precision',
            direction='maximize',
            additional_metrics=['roc_auc', 'precision', 'recall', 'f1'],
            run_full_evaluation=True
        )

    @staticmethod
    def general_classification():
        """General classification optimization."""
        return ScoringConfig(
            scoring_function='roc_auc',
            direction='maximize',
            additional_metrics=['accuracy', 'precision', 'recall', 'f1'],
            run_full_evaluation=True
        )

# Add to exports
__all__.append('CommonScoringConfigs')

# Convenience shortcuts for comprehensive workflows
def complete_model_analysis(model_class, model_params: dict, 
                           X_train, y_train, X_test, y_test=None,
                           sample_weight_train=None, sample_weight_test=None,
                           output_dir: str = "complete_analysis") -> dict:
    """Run complete model analysis including tuning, explainability, and robustness.

    Args:
        model_class: Model class to analyze
        model_params: Model parameters for tuning
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target (optional)
        sample_weight_train: Training sample weights
        sample_weight_test: Test sample weights
        output_dir: Directory to save all outputs

    Returns:
        Dictionary with all analysis results

    Example:
        >>> results = complete_model_analysis(
        ...     xgb.XGBClassifier, params,
        ...     X_train, y_train, X_test, y_test,
        ...     sample_weight_train, sample_weight_test,
        ...     output_dir="fraud_model_analysis"
        ... )
    """
    from pathlib import Path
    import time

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting complete model analysis in {output_dir}")
    start_time = time.time()

    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'config': {
            'model_class': str(model_class),
            'model_params': model_params,
            'output_dir': str(output_dir)
        }
    }

    # 1. Hyperparameter Tuning
    print("\n1️⃣ Running hyperparameter tuning...")
    tuning_config = CommonScoringConfigs.fraud_detection_pr_auc()

    tuner = EnhancedOptunaHyperparameterTuner(
        model_type=model_class.__name__.replace('Classifier', '').lower(),
        n_trials=50,
        scoring_config=tuning_config
    )

    # Combine data for tuning
    if y_test is not None:
        X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
        if sample_weight_train is not None and sample_weight_test is not None:
            sw_combined = np.concatenate([sample_weight_train, sample_weight_test])
        elif sample_weight_train is not None:
            sw_combined = np.concatenate([sample_weight_train, np.ones(len(X_test))])
        else:
            sw_combined = None
    else:
        X_combined = X_train
        y_combined = y_train
        sw_combined = sample_weight_train

    best_params, best_score = tuner.optimize(X_combined, y_combined, sw_combined)
    results['tuning'] = {
        'best_params': best_params,
        'best_score': best_score,
        'trial_results': tuner.get_trial_results_dataframe()
    }

    # Train best model
    best_model = model_class(**best_params)
    if sample_weight_train is not None:
        try:
            best_model.fit(X_train, y_train, sample_weight=sample_weight_train)
        except TypeError:
            best_model.fit(X_train, y_train)
    else:
        best_model.fit(X_train, y_train)

    # 2. Model Explainability
    print("\n2️⃣ Generating explainability report...")
    explainer = ModelExplainerSuite(best_model, X_train, y_train)
    explainability_report = explainer.generate_full_report(
        X_test, sample_weight_test, 
        output_dir=output_dir / "explainability"
    )
    results['explainability'] = explainability_report

    # 3. Robustness Testing
    print("\n3️⃣ Running robustness testing...")

    # Create updated params with best hyperparameters
    robustness_params = model_params.copy()
    robustness_params.update(best_params)

    robustness_suite = RobustnessSuite(
        model_class=model_class,
        model_params=robustness_params,
        config=RobustnessConfig(n_seeds=10)
    )

    robustness_results = robustness_suite.run_full_robustness_test(
        X_train, y_train, X_test, y_test,
        sample_weight_train, sample_weight_test,
        model_for_sensitivity=best_model,
        output_dir=output_dir / "robustness"
    )
    results['robustness'] = robustness_results

    # 4. Summary Report
    print("\n4️⃣ Generating final summary...")

    total_time = time.time() - start_time

    summary = {
        'analysis_duration_minutes': total_time / 60,
        'best_model_performance': {
            'tuning_score': best_score,
            'primary_metric': tuning_config.scoring_function
        },
        'explainability_files': len(explainability_report.get('files_generated', [])),
        'robustness_summary': robustness_results.get('summary', {}),
        'recommendations': []
    }

    # Add recommendations based on results
    if robustness_results.get('summary', {}).get('overall_assessment') == 'Poor':
        summary['recommendations'].append("Model shows poor robustness - consider ensemble methods")

    if explainability_report.get('summary_stats', {}).get('scorecard_stats', {}).get('high_risk_pct', 0) > 20:
        summary['recommendations'].append("High percentage of high-risk predictions - validate model calibration")

    results['summary'] = summary

    # Save complete results
    import json
    with open(output_dir / "complete_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Complete model analysis finished in {total_time/60:.1f} minutes!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Best {tuning_config.scoring_function}: {best_score:.4f}")
    print(f"🔍 Generated {summary['explainability_files']} explainability files")
    print(f"🛡️ Robustness assessment: {robustness_results.get('summary', {}).get('overall_assessment', 'Unknown')}")

    return results


def fraud_detection_pipeline(model_class, X_train, y_train, X_test, y_test=None,
                           sample_weight_train=None, sample_weight_test=None,
                           focus: str = 'balanced',
                           output_dir: str = "fraud_detection_pipeline") -> dict:
    """Specialized pipeline for fraud detection model development.

    Args:
        model_class: Model class (XGBoost, LightGBM, CatBoost)
        X_train: Training features
        y_train: Training target
        X_test: Test features  
        y_test: Test target (optional)
        sample_weight_train: Training sample weights
        sample_weight_test: Test sample weights
        focus: Optimization focus ('recall', 'precision', 'balanced', 'pr_auc')
        output_dir: Output directory

    Returns:
        Dictionary with pipeline results
    """
    print(f"🕵️ Starting fraud detection pipeline with {focus} focus")

    # Choose appropriate scoring configuration
    if focus == 'recall':
        scoring_config = CommonScoringConfigs.fraud_detection_recall()
    elif focus == 'precision':
        scoring_config = CommonScoringConfigs.fraud_detection_precision()
    elif focus == 'balanced':
        scoring_config = CommonScoringConfigs.fraud_detection_balanced()
    elif focus == 'pr_auc':
        scoring_config = CommonScoringConfigs.fraud_detection_pr_auc()
    else:
        raise ValueError(f"Unknown focus: {focus}")

    # Run complete analysis with fraud-specific configuration
    return complete_model_analysis(
        model_class=model_class,
        model_params={},  # Use defaults, will be tuned
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        sample_weight_train=sample_weight_train,
        sample_weight_test=sample_weight_test,
        output_dir=output_dir
    )


# Add convenience functions to exports
__all__.extend(['complete_model_analysis', 'fraud_detection_pipeline'])

# Display usage help
def show_usage_examples():
    """Show comprehensive usage examples."""
    examples = """
    🎯 COMPREHENSIVE MODEL DEVELOPMENT - USAGE EXAMPLES
    ===================================================

    1. 🚀 COMPLETE MODEL ANALYSIS PIPELINE:

       from tree_model_helper.models import complete_model_analysis
       import xgboost as xgb

       # Run everything: tuning + explainability + robustness
       results = complete_model_analysis(
           model_class=xgb.XGBClassifier,
           model_params={'objective': 'binary:logistic'},
           X_train=X_train, y_train=y_train,
           X_test=X_test, y_test=y_test,
           sample_weight_train=weights_train,
           sample_weight_test=weights_test,
           output_dir="complete_fraud_analysis"
       )

    2. 🕵️ FRAUD DETECTION SPECIALIZED PIPELINE:

       from tree_model_helper.models import fraud_detection_pipeline

       # Optimized for fraud detection
       results = fraud_detection_pipeline(
           model_class=xgb.XGBClassifier,
           X_train=X_train, y_train=y_train,
           X_test=X_test, y_test=y_test,
           sample_weight_train=weights,
           focus='recall',  # Catch maximum frauds
           output_dir="fraud_model_pipeline"
       )

    3. 🔧 INDIVIDUAL COMPONENT USAGE:

       # Hyperparameter tuning
       from tree_model_helper.models import tune_hyperparameters, ScoringConfig

       best_params, best_score = tune_hyperparameters(
           'xgboost', X, y, sample_weight=weights,
           scoring_function='recall',
           additional_metrics=['precision', 'f1', 'roc_auc']
       )

       # Model explainability
       from tree_model_helper.models import ModelExplainerSuite

       explainer = ModelExplainerSuite(model, X_train, y_train)
       report = explainer.generate_full_report(X_test, sample_weight=weights)

       # Robustness testing
       from tree_model_helper.models import quick_robustness_test

       robustness = quick_robustness_test(
           xgb.XGBClassifier, params, X, y, sample_weight=weights
       )

    4. 📊 SCORECARD CONVERSION:

       from tree_model_helper.models import convert_to_scorecard

       # Convert probabilities to business-friendly scores
       scores, converter = convert_to_scorecard(y_pred_proba, sample_weight=weights)

       # Interpret individual scores
       interpretation = converter.interpret_score(scores[0])
       print(f"Risk: {interpretation['risk_category']} ({scores[0]:.0f}/850)")

    5. 📈 SHAP ANALYSIS:

       from tree_model_helper.models import quick_shap_analysis

       # Quick SHAP analysis with plots
       shap_results = quick_shap_analysis(
           model, X_test, sample_weight=weights, save_dir="shap_analysis"
       )

       print(f"Top feature: {shap_results['feature_importance'].iloc[0]['feature']}")

    6. 🛡️ ROBUSTNESS & DRIFT DETECTION:

       from tree_model_helper.models import calculate_psi_simple

       # Check for data drift
       psi_results = calculate_psi_simple(X_train, X_test)
       high_drift_features = psi_results[psi_results['psi'] > 0.2]

       print(f"Features with significant drift: {len(high_drift_features)}")

    📋 ALL MODULES WORK WITH SAMPLE WEIGHTS THROUGHOUT! 
    Perfect for fraud detection and imbalanced datasets.
    """

    print(examples)

# Add to exports
__all__.append('show_usage_examples')
