"""Baseline XGBoost training script demonstrating end-to-end pipeline.

This script demonstrates the complete workflow from data loading to model
evaluation using the tree_model_helper package.
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
from tree_model_helper.tracking.mlflow_logger import MLflowLogger
from tree_model_helper.utils.logger import get_logger
from tree_model_helper.config.model_config import XGB_DEFAULT_PARAMS, SEED

# Setup logging
logger = get_logger(__name__)


def create_sample_data(n_samples: int = 10000, n_features: int = 20, 
                      save_path: str = "data/sample_data.csv") -> str:
    """Create sample binary classification dataset.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        save_path: Path to save the dataset

    Returns:
        Path to the saved dataset
    """
    logger.info(f"Creating sample dataset: {n_samples} samples, {n_features} features")

    # Set random seed for reproducibility
    np.random.seed(SEED)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Add some correlation structure
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = -X[:, 0] + 0.3 * np.random.randn(n_samples)

    # Generate target with some signal
    linear_combination = (
        0.5 * X[:, 0] + 
        0.3 * X[:, 1] - 
        0.2 * X[:, 2] + 
        0.1 * X[:, 3] * X[:, 4] +
        0.15 * np.sin(X[:, 5])
    )

    # Add noise and convert to probabilities
    probabilities = 1 / (1 + np.exp(-(linear_combination + 0.2 * np.random.randn(n_samples))))
    y = np.random.binomial(1, probabilities)

    # Create feature names
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Add some categorical features
    df['category_A'] = np.random.choice(['cat_1', 'cat_2', 'cat_3'], n_samples)
    df['category_B'] = np.random.choice(['type_X', 'type_Y'], n_samples, p=[0.3, 0.7])

    # Add some missing values
    missing_mask = np.random.random((n_samples, n_features)) < 0.05
    df.iloc[:, :n_features] = df.iloc[:, :n_features].mask(missing_mask)

    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    logger.info(f"Sample dataset saved to {save_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")

    return save_path


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train XGBoost baseline model")
    parser.add_argument("--data_path", type=str, help="Path to dataset CSV file")
    parser.add_argument("--target_col", type=str, default="target", help="Target column name")
    parser.add_argument("--output_dir", type=str, default="output/xgb_baseline", 
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="xgb_baseline_experiment",
                       help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, help="MLflow run name")
    parser.add_argument("--sample_size", type=int, help="Sample size for training")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create sample dataset if no data_path provided")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting XGBoost baseline training pipeline")
    logger.info(f"Output directory: {output_dir}")

    # Handle data loading
    if args.data_path:
        data_path = args.data_path
        logger.info(f"Using provided dataset: {data_path}")
    elif args.create_sample:
        data_path = create_sample_data(save_path=str(output_dir / "sample_data.csv"))
    else:
        logger.error("No data path provided and --create_sample not specified")
        sys.exit(1)

    # Start MLflow tracking
    with MLflowLogger(experiment_name=args.experiment_name, run_name=args.run_name) as mlf:

        # Log run parameters
        mlf.log_params({
            "data_path": data_path,
            "target_col": args.target_col,
            "output_dir": str(output_dir),
            "sample_size": args.sample_size
        })

        try:
            # Step 1: Load data
            logger.info("Step 1: Loading data")
            df = load_csv(data_path, sample_size=args.sample_size)

            # Log data statistics
            data_stats = {
                "n_samples": len(df),
                "n_features": len(df.columns) - 1,
                "target_rate": df[args.target_col].mean(),
                "missing_values": df.isnull().sum().sum()
            }
            mlf.log_params(data_stats)

            # Step 2: Prepare features and target
            logger.info("Step 2: Preparing features and target")
            X = df.drop(args.target_col, axis=1)
            y = df[args.target_col]

            # Step 3: Split data
            logger.info("Step 3: Splitting data")
            X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
                X, y, test_size=0.2, valid_size=0.2, stratify=True
            )

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

            # Step 4: Preprocessing
            logger.info("Step 4: Preprocessing data")
            preprocessor = DataPreprocessor(
                missing_strategy="median",
                encoding_strategy="label",
                scaling_strategy=None  # Tree models don't need scaling
            )

            X_train_processed = preprocessor.fit_transform(X_train, y_train)
            X_valid_processed = preprocessor.transform(X_valid)
            X_test_processed = preprocessor.transform(X_test)

            # Log preprocessing info
            preprocessing_info = {
                "missing_strategy": "median",
                "encoding_strategy": "label",
                "scaling_strategy": None,
                "n_features_after_preprocessing": X_train_processed.shape[1]
            }
            mlf.log_params(preprocessing_info)

            # Step 5: Train model
            logger.info("Step 5: Training XGBoost model")

            # Use default parameters with some modifications
            model_params = XGB_DEFAULT_PARAMS.copy()
            model_params.update({
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            })

            # Log model parameters
            mlf.log_params(model_params)

            # Train model
            model = train_model(
                model_type="xgboost",
                X_train=X_train_processed,
                y_train=y_train,
                X_valid=X_valid_processed,
                y_valid=y_valid,
                params=model_params,
                early_stopping_rounds=50,
                verbose=True,
                save_path=output_dir / "xgb_baseline_model.pkl"
            )

            # Log model artifact
            mlf.log_model(model.model, "xgboost_model", model_type="xgboost")

            # Step 6: Evaluate model
            logger.info("Step 6: Evaluating model")

            # Training set evaluation
            y_train_pred = model.predict_proba(X_train_processed)[:, 1]
            train_results = evaluate_model(
                y_train, y_train_pred, 
                output_dir=output_dir / "train_evaluation",
                generate_plots=True
            )

            # Validation set evaluation
            y_valid_pred = model.predict_proba(X_valid_processed)[:, 1]
            valid_results = evaluate_model(
                y_valid, y_valid_pred,
                output_dir=output_dir / "valid_evaluation", 
                generate_plots=True
            )

            # Test set evaluation
            y_test_pred = model.predict_proba(X_test_processed)[:, 1]
            test_results = evaluate_model(
                y_test, y_test_pred,
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

            # Step 7: Feature importance analysis
            logger.info("Step 7: Feature importance analysis")

            feature_importance = pd.DataFrame({
                'feature': X_train_processed.columns,
                'importance': model.get_feature_importance()
            }).sort_values('importance', ascending=False)

            # Save feature importance
            importance_path = output_dir / "feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlf.log_artifact(importance_path, "analysis")

            # Log top features
            top_features = feature_importance.head(10)['feature'].tolist()
            mlf.log_params({"top_10_features": top_features[:5]})  # MLflow param length limit

            # Step 8: Generate summary
            logger.info("Step 8: Generating summary")

            summary = {
                "model_type": "xgboost",
                "dataset_info": data_stats,
                "split_info": split_info,
                "model_params": model_params,
                "train_auc": train_results['metrics']['auc_roc'],
                "valid_auc": valid_results['metrics']['auc_roc'],
                "test_auc": test_results['metrics']['auc_roc'],
                "best_iteration": model.best_iteration,
                "top_5_features": top_features[:5]
            }

            # Save summary
            mlf.log_dict(summary, "experiment_summary.json")

            # Print summary
            logger.info("=" * 50)
            logger.info("EXPERIMENT SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Model Type: {summary['model_type']}")
            logger.info(f"Dataset: {data_stats['n_samples']} samples, {data_stats['n_features']} features")
            logger.info(f"Train AUC: {summary['train_auc']:.4f}")
            logger.info(f"Valid AUC: {summary['valid_auc']:.4f}")
            logger.info(f"Test AUC: {summary['test_auc']:.4f}")
            logger.info(f"Best Iteration: {summary['best_iteration']}")
            logger.info(f"Top Features: {', '.join(summary['top_5_features'])}")
            logger.info("=" * 50)

            # Set success tag
            mlf.set_tags({"status": "success", "model_type": "xgboost"})

            logger.info("Training pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            mlf.set_tags({"status": "failed", "error": str(e)})
            raise


if __name__ == "__main__":
    main()
