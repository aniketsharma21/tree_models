"""Comprehensive experiment script demonstrating sample weights, feature selection, and tuning.

This script demonstrates the complete workflow including:
- Sample weights handling
- Feature selection (variance, RFECV, Boruta)
- Hyperparameter tuning with Optuna
- MLflow experiment tracking
- Comprehensive evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_model_helper.data.data_loader import load_csv
from tree_model_helper.data.data_preprocessor import split_data, DataPreprocessor
from tree_model_helper.models.trainer import train_model
from tree_model_helper.models.evaluator import evaluate_model
from tree_model_helper.models.tuner import OptunaHyperparameterTuner
from tree_model_helper.models.feature_selector import ComprehensiveFeatureSelector
from tree_model_helper.tracking.mlflow_logger import MLflowLogger
from tree_model_helper.utils.logger import get_logger
from tree_model_helper.config.model_config import SEED

# Setup logging
logger = get_logger(__name__)


def create_sample_data_with_weights(n_samples: int = 10000, n_features: int = 50,
                                   imbalance_ratio: float = 0.1,
                                   save_path: str = "data/sample_data_weighted.csv") -> str:
    """Create sample binary classification dataset with sample weights.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        imbalance_ratio: Ratio of positive to negative samples
        save_path: Path to save the dataset

    Returns:
        Path to the saved dataset
    """
    logger.info(f"Creating weighted sample dataset: {n_samples} samples, {n_features} features")
    logger.info(f"Imbalance ratio: {imbalance_ratio}")

    # Set random seed for reproducibility
    np.random.seed(SEED)

    # Generate features with different patterns
    X = np.random.randn(n_samples, n_features)

    # Add correlation structure
    for i in range(1, min(5, n_features)):
        X[:, i] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_samples)

    # Add some irrelevant features (low variance)
    for i in range(n_features//2, n_features//2 + 3):
        if i < n_features:
            X[:, i] = np.random.normal(0, 0.01, n_samples)  # Very low variance

    # Generate target with imbalance
    n_positive = int(n_samples * imbalance_ratio)
    y = np.concatenate([
        np.ones(n_positive),
        np.zeros(n_samples - n_positive)
    ])

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Create signal in features for target
    signal_features = [0, 1, 2, 3, 5]
    linear_combination = np.zeros(n_samples)

    for i, feat_idx in enumerate(signal_features):
        if feat_idx < n_features:
            coeff = 0.5 * ((-1) ** i)  # Alternating positive/negative coefficients
            linear_combination += coeff * X[:, feat_idx]

    # Add non-linear interactions
    if n_features > 10:
        linear_combination += 0.2 * X[:, 5] * X[:, 7]
        linear_combination += 0.15 * np.sin(X[:, 8])

    # Adjust target based on linear combination for positive class
    positive_indices = np.where(y == 1)[0]
    for idx in positive_indices:
        if np.random.random() < 0.7:  # 70% of positives follow the signal
            # Modify features to create stronger signal
            X[idx, signal_features] += 0.5 * np.random.randn(len(signal_features))

    # Generate sample weights
    # Weight positive samples higher to address imbalance
    sample_weights = np.ones(n_samples)
    sample_weights[y == 1] = 1.0 / imbalance_ratio  # Upweight minority class

    # Add some noise to weights
    sample_weights *= (1 + 0.1 * np.random.randn(n_samples))
    sample_weights = np.clip(sample_weights, 0.1, None)  # Ensure positive weights

    # Create feature names
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y.astype(int)
    df['sample_weight'] = sample_weights

    # Add some categorical features
    df['category_A'] = np.random.choice(['cat_1', 'cat_2', 'cat_3'], n_samples, p=[0.5, 0.3, 0.2])
    df['category_B'] = np.random.choice(['type_X', 'type_Y'], n_samples, p=[0.3, 0.7])

    # Add some missing values randomly
    missing_cols = np.random.choice(feature_names, size=min(5, len(feature_names)), replace=False)
    for col in missing_cols:
        missing_mask = np.random.random(n_samples) < 0.05
        df.loc[missing_mask, col] = np.nan

    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    logger.info(f"Sample dataset saved to {save_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    logger.info(f"Sample weights range: [{df['sample_weight'].min():.2f}, {df['sample_weight'].max():.2f}]")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")

    return save_path


def main():
    """Main comprehensive experiment pipeline."""
    parser = argparse.ArgumentParser(description="Comprehensive ML experiment with all features")
    parser.add_argument("--data_path", type=str, help="Path to dataset CSV file")
    parser.add_argument("--target_col", type=str, default="target", help="Target column name")
    parser.add_argument("--weight_col", type=str, default="sample_weight", help="Sample weight column name")
    parser.add_argument("--output_dir", type=str, default="output/comprehensive_experiment", 
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="comprehensive_ml_experiment",
                       help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, help="MLflow run name")
    parser.add_argument("--model_type", type=str, default="xgboost", choices=["xgboost", "lightgbm", "catboost"],
                       help="Model type to use")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of hyperparameter tuning trials")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create sample dataset if no data_path provided")
    parser.add_argument("--skip_feature_selection", action="store_true",
                       help="Skip feature selection step")
    parser.add_argument("--skip_tuning", action="store_true",
                       help="Skip hyperparameter tuning step")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting Comprehensive ML Experiment Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Tuning trials: {args.n_trials}")
    logger.info("=" * 60)

    # Handle data loading
    if args.data_path:
        data_path = args.data_path
        logger.info(f"Using provided dataset: {data_path}")
    elif args.create_sample:
        data_path = create_sample_data_with_weights(
            n_samples=15000,
            n_features=30,
            save_path=str(output_dir / "sample_data_weighted.csv")
        )
    else:
        logger.error("No data path provided and --create_sample not specified")
        sys.exit(1)

    # Start MLflow tracking
    with MLflowLogger(experiment_name=args.experiment_name, run_name=args.run_name) as mlf:

        # Log run parameters
        mlf.log_params({
            "data_path": data_path,
            "target_col": args.target_col,
            "weight_col": args.weight_col,
            "model_type": args.model_type,
            "n_trials": args.n_trials,
            "output_dir": str(output_dir),
            "skip_feature_selection": args.skip_feature_selection,
            "skip_tuning": args.skip_tuning
        })

        try:
            # Step 1: Load data
            logger.info("📂 Step 1: Loading data")
            df = load_csv(data_path)

            # Separate features, target, and sample weights
            feature_cols = [col for col in df.columns 
                          if col not in [args.target_col, args.weight_col]]

            X = df[feature_cols]
            y = df[args.target_col]

            # Handle sample weights
            if args.weight_col in df.columns:
                sample_weights = df[args.weight_col].values
                logger.info(f"✅ Sample weights loaded: range [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
            else:
                sample_weights = None
                logger.info("⚠️  No sample weights found, using uniform weights")

            # Log data statistics
            data_stats = {
                "n_samples": len(df),
                "n_features": len(feature_cols),
                "target_rate": y.mean(),
                "missing_values": X.isnull().sum().sum(),
                "use_sample_weights": sample_weights is not None
            }

            if sample_weights is not None:
                data_stats.update({
                    "min_weight": float(sample_weights.min()),
                    "max_weight": float(sample_weights.max()),
                    "mean_weight": float(sample_weights.mean()),
                    "total_weight": float(sample_weights.sum())
                })

            mlf.log_params(data_stats)

            # Step 2: Split data
            logger.info("✂️  Step 2: Splitting data")
            X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
                X, y, test_size=0.2, valid_size=0.2, stratify=True
            )

            # Split sample weights accordingly
            if sample_weights is not None:
                train_idx = X_train.index
                valid_idx = X_valid.index
                test_idx = X_test.index

                weights_train = sample_weights[train_idx]
                weights_valid = sample_weights[valid_idx]
                weights_test = sample_weights[test_idx]
            else:
                weights_train = weights_valid = weights_test = None

            # Log split information
            split_info = {
                "train_samples": len(X_train),
                "valid_samples": len(X_valid),
                "test_samples": len(X_test),
                "train_target_rate": y_train.mean(),
                "valid_target_rate": y_valid.mean(),
                "test_target_rate": y_test.mean()
            }
            mlf.log_params(split_info)

            # Step 3: Preprocessing
            logger.info("🔧 Step 3: Preprocessing data")
            preprocessor = DataPreprocessor(
                missing_strategy="median",
                encoding_strategy="label",
                scaling_strategy=None  # Tree models don't need scaling
            )

            X_train_processed = preprocessor.fit_transform(X_train, y_train)
            X_valid_processed = preprocessor.transform(X_valid)
            X_test_processed = preprocessor.transform(X_test)

            # Step 4: Feature Selection (Optional)
            if not args.skip_feature_selection:
                logger.info("🎯 Step 4: Feature Selection")

                feature_selector = ComprehensiveFeatureSelector(
                    variance_threshold=0.01,
                    rfecv_params={"cv": 3, "scoring": "roc_auc"},  # Reduced CV for speed
                    use_boruta=True  # Will use Boruta if available
                )

                # Run comprehensive feature selection
                selection_results = feature_selector.fit_transform_all(
                    X_train_processed, y_train, weights_train
                )

                # Get consensus features
                consensus_features = feature_selector.get_consensus_features(min_agreement=2)

                # Use RFECV selected features for final model
                X_train_selected = selection_results['rfecv']
                X_valid_selected = feature_selector.rfecv_selector.transform(X_valid_processed)
                X_test_selected = feature_selector.rfecv_selector.transform(X_test_processed)

                # Save feature selection results
                feature_selection_dir = output_dir / "feature_selection"
                feature_selector.save_results(str(feature_selection_dir))
                mlf.log_artifacts(feature_selection_dir, "feature_selection")

                # Log feature selection info
                feature_info = {
                    "original_features": len(X_train_processed.columns),
                    "variance_selected": len(selection_results['variance'].columns),
                    "rfecv_selected": len(X_train_selected.columns),
                    "consensus_features": len(consensus_features)
                }

                if 'boruta' in selection_results:
                    feature_info["boruta_selected"] = len(selection_results['boruta'].columns)

                mlf.log_params(feature_info)

                logger.info(f"✅ Feature selection completed:")
                logger.info(f"   Original: {feature_info['original_features']} features")
                logger.info(f"   RFECV selected: {feature_info['rfecv_selected']} features")
                logger.info(f"   Consensus: {len(consensus_features)} features")

            else:
                logger.info("⏭️  Step 4: Skipping feature selection")
                X_train_selected = X_train_processed
                X_valid_selected = X_valid_processed
                X_test_selected = X_test_processed

            # Step 5: Hyperparameter Tuning (Optional)
            if not args.skip_tuning:
                logger.info("🎛️  Step 5: Hyperparameter Tuning")

                tuner = OptunaHyperparameterTuner(
                    model_type=args.model_type,
                    n_trials=args.n_trials,
                    cv_folds=3,  # Reduced for speed
                    scoring="roc_auc",
                    mlflow_logger=mlf
                )

                # Run optimization
                best_params, best_score = tuner.optimize(
                    X_train_selected, y_train, weights_train
                )

                # Save tuning results
                tuning_dir = output_dir / "hyperparameter_tuning"
                tuning_dir.mkdir(exist_ok=True)
                tuner.save_study(str(tuning_dir / "optimization_study.pkl"))

                # Plot optimization history if matplotlib available
                try:
                    tuner.plot_optimization_history(str(tuning_dir / "optimization_history.png"))
                    mlf.log_artifact(tuning_dir / "optimization_history.png", "tuning")
                except Exception as e:
                    logger.warning(f"Could not plot optimization history: {e}")

                # Get parameter importance
                try:
                    param_importance = tuner.get_feature_importance()
                    if not param_importance.empty:
                        param_importance.to_csv(tuning_dir / "parameter_importance.csv", index=False)
                        mlf.log_artifact(tuning_dir / "parameter_importance.csv", "tuning")
                except Exception as e:
                    logger.warning(f"Could not get parameter importance: {e}")

                logger.info(f"✅ Hyperparameter tuning completed:")
                logger.info(f"   Best CV score: {best_score:.4f}")
                logger.info(f"   Best parameters: {best_params}")

            else:
                logger.info("⏭️  Step 5: Skipping hyperparameter tuning")
                # Use default parameters
                from tree_model_helper.models.trainer import ModelTrainer
                trainer = ModelTrainer()
                best_params = trainer.get_default_params(args.model_type)
                best_score = None

            mlf.log_params({"best_" + k: v for k, v in best_params.items()})
            if best_score:
                mlf.log_metrics({"best_cv_score": best_score})

            # Step 6: Train Final Model
            logger.info("🏋️  Step 6: Training final model")

            final_model = train_model(
                model_type=args.model_type,
                X_train=X_train_selected,
                y_train=y_train,
                X_valid=X_valid_selected,
                y_valid=y_valid,
                sample_weight_train=weights_train,
                sample_weight_valid=weights_valid,
                params=best_params,
                early_stopping_rounds=50,
                verbose=True,
                save_path=output_dir / f"{args.model_type}_final_model.pkl"
            )

            # Log model artifact
            mlf.log_model(final_model.model, f"{args.model_type}_final_model", model_type=args.model_type)

            # Step 7: Comprehensive Evaluation
            logger.info("📊 Step 7: Model evaluation")

            # Training set evaluation
            logger.info("   Evaluating on training set...")
            y_train_pred = final_model.predict_proba(X_train_selected)[:, 1]
            train_results = evaluate_model(
                y_train, y_train_pred,
                sample_weight=weights_train,
                output_dir=output_dir / "train_evaluation",
                generate_plots=True
            )

            # Validation set evaluation
            logger.info("   Evaluating on validation set...")
            y_valid_pred = final_model.predict_proba(X_valid_selected)[:, 1]
            valid_results = evaluate_model(
                y_valid, y_valid_pred,
                sample_weight=weights_valid,
                output_dir=output_dir / "valid_evaluation",
                generate_plots=True
            )

            # Test set evaluation
            logger.info("   Evaluating on test set...")
            y_test_pred = final_model.predict_proba(X_test_selected)[:, 1]
            test_results = evaluate_model(
                y_test, y_test_pred,
                sample_weight=weights_test,
                output_dir=output_dir / "test_evaluation",
                generate_plots=True
            )

            # Log metrics to MLflow
            train_metrics = {f"train_{k}": v for k, v in train_results['metrics'].items()}
            valid_metrics = {f"valid_{k}": v for k, v in valid_results['metrics'].items()}
            test_metrics = {f"test_{k}": v for k, v in test_results['metrics'].items()}

            mlf.log_metrics(train_metrics)
            mlf.log_metrics(valid_metrics)
            mlf.log_metrics(test_metrics)

            # Log evaluation artifacts
            mlf.log_artifacts(output_dir / "train_evaluation", "train_plots")
            mlf.log_artifacts(output_dir / "valid_evaluation", "valid_plots")
            mlf.log_artifacts(output_dir / "test_evaluation", "test_plots")

            # Step 8: Feature Importance Analysis
            logger.info("🔍 Step 8: Feature importance analysis")

            feature_importance = pd.DataFrame({
                'feature': X_train_selected.columns,
                'importance': final_model.get_feature_importance()
            }).sort_values('importance', ascending=False)

            # Save feature importance
            importance_path = output_dir / "feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlf.log_artifact(importance_path, "analysis")

            # Log top features
            top_features = feature_importance.head(10)['feature'].tolist()
            mlf.log_params({"top_5_features": top_features[:5]})

            # Step 9: Generate Final Summary
            logger.info("📋 Step 9: Generating experiment summary")

            summary = {
                "experiment_info": {
                    "model_type": args.model_type,
                    "used_sample_weights": sample_weights is not None,
                    "used_feature_selection": not args.skip_feature_selection,
                    "used_hyperparameter_tuning": not args.skip_tuning,
                    "n_tuning_trials": args.n_trials if not args.skip_tuning else 0
                },
                "dataset_info": data_stats,
                "split_info": split_info,
                "final_params": best_params,
                "performance": {
                    "train_auc": train_results['metrics']['auc_roc'],
                    "valid_auc": valid_results['metrics']['auc_roc'],
                    "test_auc": test_results['metrics']['auc_roc'],
                    "train_gini": train_results['metrics']['gini'],
                    "valid_gini": valid_results['metrics']['gini'],
                    "test_gini": test_results['metrics']['gini'],
                    "train_ks": train_results['metrics']['ks_statistic'],
                    "valid_ks": valid_results['metrics']['ks_statistic'],
                    "test_ks": test_results['metrics']['ks_statistic']
                },
                "model_info": {
                    "best_iteration": final_model.best_iteration,
                    "n_features_final": len(X_train_selected.columns),
                    "top_5_features": top_features[:5]
                }
            }

            if not args.skip_tuning:
                summary["tuning_info"] = {
                    "best_cv_score": best_score,
                    "n_trials": args.n_trials
                }

            if not args.skip_feature_selection:
                summary["feature_selection_info"] = feature_info

            # Save summary
            mlf.log_dict(summary, "experiment_summary.json")

            # Print comprehensive summary
            logger.info("=" * 70)
            logger.info("🎉 COMPREHENSIVE EXPERIMENT SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Model Type: {summary['experiment_info']['model_type']}")
            logger.info(f"Dataset: {summary['dataset_info']['n_samples']} samples, "
                       f"{summary['dataset_info']['n_features']} original features")
            logger.info(f"Final Features: {summary['model_info']['n_features_final']}")
            logger.info(f"Sample Weights: {'✅ Used' if summary['experiment_info']['used_sample_weights'] else '❌ Not used'}")
            logger.info(f"Feature Selection: {'✅ Used' if summary['experiment_info']['used_feature_selection'] else '❌ Skipped'}")
            logger.info(f"Hyperparameter Tuning: {'✅ Used' if summary['experiment_info']['used_hyperparameter_tuning'] else '❌ Skipped'}")
            logger.info("")
            logger.info("📈 Performance Metrics (with sample weights):")
            logger.info(f"   Train AUC: {summary['performance']['train_auc']:.4f}")
            logger.info(f"   Valid AUC: {summary['performance']['valid_auc']:.4f}")
            logger.info(f"   Test AUC:  {summary['performance']['test_auc']:.4f}")
            logger.info(f"   Train Gini: {summary['performance']['train_gini']:.4f}")
            logger.info(f"   Valid Gini: {summary['performance']['valid_gini']:.4f}")
            logger.info(f"   Test Gini:  {summary['performance']['test_gini']:.4f}")
            logger.info(f"   Train KS: {summary['performance']['train_ks']:.4f}")
            logger.info(f"   Valid KS: {summary['performance']['valid_ks']:.4f}")
            logger.info(f"   Test KS:  {summary['performance']['test_ks']:.4f}")
            logger.info("")
            logger.info(f"🔝 Top 5 Features: {', '.join(summary['model_info']['top_5_features'])}")
            logger.info(f"🎯 Best Iteration: {summary['model_info']['best_iteration']}")

            if not args.skip_tuning:
                logger.info(f"⚡ Best CV Score: {summary['tuning_info']['best_cv_score']:.4f}")

            logger.info("=" * 70)

            # Set success tag
            mlf.set_tags({
                "status": "success", 
                "model_type": args.model_type,
                "experiment_type": "comprehensive",
                "used_sample_weights": str(summary['experiment_info']['used_sample_weights']),
                "used_feature_selection": str(summary['experiment_info']['used_feature_selection']),
                "used_tuning": str(summary['experiment_info']['used_hyperparameter_tuning'])
            })

            logger.info("🚀 Comprehensive experiment completed successfully!")
            logger.info(f"📁 All results saved to: {output_dir}")
            logger.info(f"📊 MLflow experiment: {args.experiment_name}")

        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            mlf.set_tags({"status": "failed", "error": str(e)})
            raise


if __name__ == "__main__":
    main()
