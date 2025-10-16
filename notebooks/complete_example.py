# examples/quickstart/complete_example.py
"""Complete example demonstrating the refactored tree_models package.

This example shows how to use the refactored tree_models package for
a complete machine learning workflow including:
- Type-safe configuration
- Hyperparameter tuning with error handling
- Model training and evaluation
- Comprehensive explainability analysis
- Robustness testing
- Production-ready error handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, Any
import logging

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# Import the refactored tree_models package
import tree_models as tm
from tree_models.models import (
    StandardModelTrainer,
    OptunaHyperparameterTuner,
    ScoringConfig,
    tune_hyperparameters
)
from tree_models.config import XGBoostConfig, create_model_config
from tree_models.explainability import (
    SHAPExplainer,
    ScorecardConverter,
    ReasonCodeGenerator,
    quick_shap_analysis,
    convert_to_scorecard
)
from tree_models.models.robustness import (
    SeedRobustnessTester,
    quick_robustness_test
)
from tree_models.data import DataValidator, validate_dataset
from tree_models.utils.logger import get_logger, configure_logging
from tree_models.utils.timer import timer, timed_operation
from tree_models.utils.exceptions import (
    TreeModelsError,
    ModelTrainingError,
    ConfigurationError
)

# Configure package logging
configure_logging(
    level="INFO",
    log_file="tree_models_example.log",
    format_style="detailed"
)

logger = get_logger(__name__)


def generate_sample_fraud_data(n_samples: int = 5000, fraud_rate: float = 0.05) -> tuple:
    """Generate sample fraud detection dataset with realistic characteristics.
    
    Args:
        n_samples: Number of samples to generate
        fraud_rate: Proportion of fraudulent transactions
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, sample_weights)
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight
    
    logger.info(f"Generating sample fraud dataset: {n_samples} samples, {fraud_rate:.1%} fraud rate")
    
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=25,
        n_informative=20,
        n_redundant=3,
        n_clusters_per_class=1,
        weights=[1-fraud_rate, fraud_rate],
        class_sep=0.8,
        random_state=42
    )
    
    # Create realistic feature names
    feature_names = [
        'transaction_amount', 'account_age_days', 'num_transactions_30d',
        'avg_transaction_amount', 'time_since_last_transaction', 'merchant_category',
        'is_weekend', 'hour_of_day', 'day_of_week', 'payment_method',
        'merchant_risk_score', 'customer_risk_score', 'geo_distance_km',
        'velocity_1h', 'velocity_24h', 'declined_count_30d',
        'cross_border_flag', 'high_risk_country', 'device_fingerprint_age',
        'account_verified', 'email_verified', 'phone_verified',
        'previous_chargeback_count', 'credit_utilization', 'account_balance_ratio'
    ]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='is_fraud')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, 
        test_size=0.3, 
        random_state=42, 
        stratify=y_series
    )
    
    # Compute sample weights for imbalanced data
    sample_weights = compute_sample_weight('balanced', y_train)
    
    logger.info(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"Fraud rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, sample_weights


@timer(name="complete_fraud_detection_example")
def main():
    """Main function demonstrating complete fraud detection workflow."""
    
    print("🚀 Tree Models v2.0 - Complete Fraud Detection Example")
    print("=" * 60)
    
    try:
        # Step 1: Data Generation and Validation
        print("\n📊 Step 1: Data Generation and Validation")
        with timed_operation("data_generation"):
            X_train, X_test, y_train, y_test, sample_weights = generate_sample_fraud_data(
                n_samples=5000, 
                fraud_rate=0.05
            )
        
        # Validate dataset
        validator = DataValidator()
        validation_result = validator.validate_dataset(
            X=pd.concat([X_train, X_test]),
            y=pd.concat([y_train, y_test]),
            sample_weight=np.concatenate([sample_weights, np.ones(len(y_test))])
        )
        
        print(f"   ✅ Data validation: {validation_result['status']}")
        print(f"   📈 Features: {validation_result['n_features']}, Samples: {validation_result['n_samples']}")
        
        # Step 2: Configuration Setup
        print("\n⚙️ Step 2: Model Configuration")
        
        # Method 1: Using preset configuration
        fraud_config = XGBoostConfig.for_fraud_detection()
        print(f"   ✅ Fraud detection config: {fraud_config.n_estimators} estimators, depth={fraud_config.max_depth}")
        
        # Method 2: Using factory with customization
        custom_config = create_model_config(
            'xgboost', 
            'fraud_detection',
            n_estimators=750,  # Custom override
            learning_rate=0.03
        )
        print(f"   ✅ Custom config: {custom_config.n_estimators} estimators, lr={custom_config.learning_rate}")
        
        # Step 3: Quick Hyperparameter Tuning
        print("\n🎯 Step 3: Hyperparameter Tuning")
        
        best_params, best_score = tune_hyperparameters(
            model_type='xgboost',
            X=X_train,
            y=y_train,
            sample_weight=sample_weights,
            scoring_function='recall',  # Optimize for fraud detection
            additional_metrics=['precision', 'f1', 'roc_auc', 'average_precision'],
            n_trials=20,
            timeout=300,  # 5 minute timeout
            random_state=42
        )
        
        print(f"   ✅ Best recall score: {best_score:.4f}")
        print(f"   🔧 Best parameters: {best_params}")
        
        # Step 4: Advanced Hyperparameter Tuning with Full Control
        print("\n🔬 Step 4: Advanced Hyperparameter Tuning")
        
        # Create trainer and tuner with full configuration
        trainer = StandardModelTrainer('xgboost', config=fraud_config, random_state=42)
        
        scoring_config = ScoringConfig(
            scoring_function='average_precision',  # PR-AUC for imbalanced data
            direction='maximize',
            additional_metrics=['precision', 'recall', 'f1', 'roc_auc'],
            cv_folds=5,
            run_full_evaluation=True,
            timeout_per_trial=60
        )
        
        tuner = OptunaHyperparameterTuner(
            model_trainer=trainer,
            n_trials=15,
            scoring_config=scoring_config,
            sampler='tpe',
            pruner='median',
            random_state=42
        )
        
        # Custom search space for fraud detection
        fraud_search_space = {
            'n_estimators': ('int', 200, 1000),
            'max_depth': ('int', 6, 12),
            'learning_rate': ('float', 0.01, 0.1, True),  # Log scale
            'subsample': ('float', 0.7, 0.95),
            'colsample_bytree': ('float', 0.7, 0.95),
            'reg_alpha': ('float', 0.01, 1.0, True),
            'reg_lambda': ('float', 0.01, 2.0, True),
            'gamma': ('float', 0.0, 0.5)
        }
        
        advanced_params, advanced_score = tuner.optimize(
            X=X_train,
            y=y_train,
            search_space=fraud_search_space,
            sample_weight=sample_weights
        )
        
        print(f"   ✅ Advanced PR-AUC score: {advanced_score:.4f}")
        
        # Get optimization history
        history_df = tuner.get_optimization_history()
        print(f"   📊 Optimization history: {len(history_df)} trials completed")
        
        # Step 5: Final Model Training
        print("\n🎓 Step 5: Final Model Training")
        
        # Use the better parameters from tuning
        final_params = advanced_params if advanced_score > best_score else best_params
        final_model = trainer.get_model(final_params)
        
        with timed_operation("final_training"):
            final_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        print(f"   ✅ Model trained with parameters: {final_params}")
        
        # Step 6: Model Evaluation  
        print("\n📈 Step 6: Model Evaluation")
        
        from tree_models.models.evaluator import StandardModelEvaluator
        
        evaluator = StandardModelEvaluator(
            metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
        )
        
        eval_result = evaluator.evaluate(
            model=final_model,
            X=X_test,
            y=y_test,
            sample_weight=np.ones(len(y_test))  # Equal weights for test evaluation
        )
        
        print("   📊 Evaluation Results:")
        for metric, value in eval_result.metrics.items():
            print(f"      {metric:>20}: {value:.4f}")
        
        # Step 7: Explainability Analysis
        print("\n🔍 Step 7: Explainability Analysis")
        
        # Quick SHAP analysis
        shap_results = quick_shap_analysis(
            model=final_model,
            X=X_test[:500],  # Subsample for speed
            sample_weight=None,
            save_dir="fraud_explainability"
        )
        
        print(f"   ✅ SHAP analysis completed")
        print(f"   📊 Feature importance (top 5):")
        
        top_features = shap_results['feature_importance'].head()
        for _, row in top_features.iterrows():
            print(f"      {row['feature']:>25}: {row['importance']:.4f}")
        
        # Scorecard conversion
        y_pred_proba = eval_result.probabilities
        if y_pred_proba is not None:
            scores, converter = convert_to_scorecard(y_pred_proba)
            
            print(f"\n   💳 Scorecard Analysis:")
            print(f"      Score range: {scores.min():.0f} - {scores.max():.0f}")
            print(f"      Mean score: {scores.mean():.0f}")
            
            # Show risk distribution
            sample_interpretation = converter.interpret_score(scores[0])
            print(f"      Sample interpretation: {sample_interpretation['risk_category']}")
        
        # Reason code generation
        explainer = SHAPExplainer(final_model)
        shap_values = explainer.compute_shap_values(X_test[:100])  # Small sample
        
        reason_generator = ReasonCodeGenerator(max_reasons=3)
        reason_codes = reason_generator.generate_reason_codes(
            shap_values=shap_values,
            feature_names=list(X_test.columns),
            X=X_test[:100]
        )
        
        print(f"\n   📝 Sample Reason Codes:")
        for i, reasons in enumerate(reason_codes[:3]):
            print(f"      Transaction {i+1}:")
            for reason in reasons['top_reasons']:
                print(f"        - {reason['text']}")
        
        # Step 8: Robustness Testing
        print("\n🛡️ Step 8: Robustness Testing")
        
        # Quick robustness test
        robustness_results = quick_robustness_test(
            model_type='xgboost',
            best_params=final_params,
            X=X_train[:1000],  # Subsample for speed
            y=y_train[:1000],
            sample_weight=sample_weights[:1000],
            n_seeds=5,
            scoring_function='average_precision'
        )
        
        print(f"   📊 Robustness Results:")
        print(f"      Mean PR-AUC: {robustness_results['mean_score']:.4f}")
        print(f"      Std Dev: {robustness_results['score_std']:.4f}")
        print(f"      Stability Score: {robustness_results.get('stability_score', 'N/A')}")
        
        # Advanced robustness testing
        robustness_tester = SeedRobustnessTester(
            n_seeds=7,
            scoring_function='average_precision'
        )
        
        detailed_robustness = robustness_tester.test_robustness(
            model_trainer=trainer,
            X=X_train[:1000],
            y=y_train[:1000],
            model_params=final_params,
            sample_weight=sample_weights[:1000]
        )
        
        stability_metrics = robustness_tester.get_stability_metrics()
        print(f"   🔬 Detailed Stability Metrics:")
        for metric, value in stability_metrics.items():
            print(f"      {metric:>20}: {value:.4f}")
        
        # Step 9: Complete Pipeline Demonstration
        print("\n🚀 Step 9: One-Line Complete Analysis")
        
        # Demonstrate the complete pipeline function
        complete_results = tm.complete_model_analysis(
            model_type='xgboost',
            X_train=X_train[:2000],  # Subset for demo speed
            y_train=y_train[:2000],
            X_test=X_test[:500],
            y_test=y_test[:500],
            sample_weight_train=sample_weights[:2000],
            n_trials=10,
            scoring_function='average_precision',
            output_dir='complete_fraud_analysis'
        )
        
        print(f"   ✅ Complete pipeline finished!")
        print(f"   📊 Final test PR-AUC: {complete_results['evaluation']['metrics']['average_precision']:.4f}")
        print(f"   📁 Results saved to: complete_fraud_analysis/")
        
        # Step 10: Fraud Detection Pipeline
        print("\n🕵️ Step 10: Specialized Fraud Detection Pipeline")
        
        fraud_results = tm.fraud_detection_pipeline(
            model_type='xgboost',
            X_train=X_train[:2000],
            y_train=y_train[:2000],
            X_test=X_test[:500],
            y_test=y_test[:500],
            sample_weight_train=sample_weights[:2000],
            focus='recall',  # Optimize for fraud detection
            n_trials=8,
            output_dir='specialized_fraud_pipeline'
        )
        
        print(f"   🎯 Fraud pipeline completed!")
        print(f"   📈 Fraud detection rate (recall): {fraud_results['evaluation']['metrics']['recall']:.4f}")
        print(f"   📉 False alarm rate: {1 - fraud_results['evaluation']['metrics']['precision']:.4f}")
        
        if 'optimal_thresholds' in fraud_results['fraud_analysis']:
            opt_thresh = fraud_results['fraud_analysis']['optimal_thresholds']['f1_score']
            print(f"   ⚖️ Optimal threshold: {opt_thresh['threshold']:.3f}")
            print(f"   🎯 At threshold - Precision: {opt_thresh['precision']:.3f}, Recall: {opt_thresh['recall']:.3f}")
        
        # Step 11: Error Handling Demonstration
        print("\n❌ Step 11: Error Handling Demonstration")
        
        try:
            # Demonstrate graceful error handling
            invalid_result = tune_hyperparameters(
                model_type='invalid_model',  # This should fail
                X=X_train[:100],
                y=y_train[:100],
                n_trials=1
            )
        except (ConfigurationError, TreeModelsError) as e:
            print(f"   ✅ Caught expected error: {type(e).__name__}")
            print(f"      Message: {str(e)}")
        
        try:
            # Test with invalid data
            empty_X = pd.DataFrame()
            empty_y = pd.Series(dtype=float)
            
            invalid_trainer = StandardModelTrainer('xgboost')
            invalid_trainer.validate_input_data(empty_X, empty_y)
            
        except (ConfigurationError, TreeModelsError) as e:
            print(f"   ✅ Caught data validation error: {type(e).__name__}")
        
        print("\n🎉 Example completed successfully!")
        print("=" * 60)
        print("Key improvements in v2.0:")
        print("  ✅ Type-safe configuration with validation")
        print("  ✅ Comprehensive error handling and recovery")
        print("  ✅ Performance monitoring and optimization")
        print("  ✅ Standardized interfaces and protocols")
        print("  ✅ Enhanced explainability and business integration")
        print("  ✅ Production-ready robustness testing")
        print("  ✅ One-line complete workflows")
        print("  ✅ Specialized fraud detection pipeline")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise


if __name__ == "__main__":
    # Show package information
    tm.show_package_info()
    
    # Run the complete example
    main()
    
    # Print performance summary
    print(f"\n📊 Performance Summary:")
    tm.print_performance_summary()