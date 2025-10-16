"""Feature selection utilities with variance filtering, RFECV, and Boruta.

This module provides comprehensive feature selection capabilities including
variance filtering, recursive feature elimination with cross-validation,
and Boruta feature selection algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
from sklearn.feature_selection import (
    VarianceThreshold, RFECV, SelectKBest, f_classif, chi2, mutual_info_classif
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import get_logger
from ..utils.timer import timer
from ..utils.io_utils import save_json, save_dataframe

logger = get_logger(__name__)


# Try to import Boruta (optional dependency)
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    logger.warning("Boruta not available. Install with: pip install boruta")


class VarianceFeatureSelector:
    """Variance-based feature selection.

    Removes features with low variance across samples.

    Example:
        >>> selector = VarianceFeatureSelector(threshold=0.01)
        >>> X_selected = selector.fit_transform(X_train)
        >>> X_test_selected = selector.transform(X_test)
    """

    def __init__(self, threshold: float = 0.0):
        """Initialize variance selector.

        Args:
            threshold: Variance threshold below which features are removed
        """
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=threshold)
        self.selected_features_ = None
        self.feature_variances_ = None
        self.fitted_ = False

    @timer
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VarianceFeatureSelector':
        """Fit the variance selector.

        Args:
            X: Training features
            y: Target (not used, for API consistency)

        Returns:
            Fitted selector
        """
        logger.info(f"Fitting variance selector with threshold {self.threshold}")

        # Calculate variances
        self.feature_variances_ = X.var()

        # Fit selector
        self.selector.fit(X)

        # Get selected features
        selected_mask = self.selector.get_support()
        self.selected_features_ = X.columns[selected_mask].tolist()

        removed_count = len(X.columns) - len(self.selected_features_)
        logger.info(f"Removed {removed_count} low-variance features")
        logger.info(f"Selected {len(self.selected_features_)} features")

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features by removing low-variance ones.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted before transform")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit selector and transform features.

        Args:
            X: Training features
            y: Target (not used)

        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)

    def get_feature_info(self) -> pd.DataFrame:
        """Get information about feature selection.

        Returns:
            DataFrame with feature variances and selection status
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted first")

        info_df = pd.DataFrame({
            'feature': self.feature_variances_.index,
            'variance': self.feature_variances_.values,
            'selected': [f in self.selected_features_ for f in self.feature_variances_.index]
        }).sort_values('variance', ascending=False)

        return info_df


class RFECVFeatureSelector:
    """Recursive Feature Elimination with Cross-Validation.

    Uses cross-validation to select optimal number of features by
    recursively eliminating features.

    Example:
        >>> selector = RFECVFeatureSelector(estimator=RandomForestClassifier())
        >>> X_selected = selector.fit_transform(X_train, y_train)
    """

    def __init__(self,
                 estimator=None,
                 step: int = 1,
                 min_features_to_select: int = 1,
                 cv: int = 5,
                 scoring: str = 'roc_auc',
                 n_jobs: int = 1,
                 random_state: int = 42):
        """Initialize RFECV selector.

        Args:
            estimator: Estimator with feature_importances_ or coef_ attribute
            step: Number of features to remove at each iteration
            min_features_to_select: Minimum number of features to select
            cv: Cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
        """
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=n_jobs
            )

        self.estimator = estimator
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.selector = None
        self.selected_features_ = None
        self.feature_rankings_ = None
        self.cv_scores_ = None
        self.fitted_ = False

    @timer
    def fit(self, X: pd.DataFrame, y: pd.Series, 
           sample_weight: Optional[np.ndarray] = None) -> 'RFECVFeatureSelector':
        """Fit the RFECV selector.

        Args:
            X: Training features
            y: Target variable
            sample_weight: Optional sample weights

        Returns:
            Fitted selector
        """
        logger.info(f"Fitting RFECV selector with {len(X.columns)} features")

        # Create custom scorer that handles sample weights
        if sample_weight is not None:
            def custom_scorer(estimator, X_val, y_val):
                y_pred = estimator.predict_proba(X_val)[:, 1]
                # Note: sample weights would need to be handled in CV splits
                return roc_auc_score(y_val, y_pred)
            scorer = make_scorer(custom_scorer, greater_is_better=True, needs_proba=True)
        else:
            scorer = self.scoring

        # Set up cross-validation
        cv_splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        # Create RFECV selector
        self.selector = RFECV(
            estimator=self.estimator,
            step=self.step,
            min_features_to_select=self.min_features_to_select,
            cv=cv_splitter,
            scoring=scorer,
            n_jobs=self.n_jobs
        )

        # Fit selector
        self.selector.fit(X, y)

        # Get results
        selected_mask = self.selector.get_support()
        self.selected_features_ = X.columns[selected_mask].tolist()
        self.feature_rankings_ = pd.Series(
            self.selector.ranking_,
            index=X.columns
        ).sort_values()
        self.cv_scores_ = self.selector.cv_results_

        optimal_features = self.selector.n_features_
        removed_count = len(X.columns) - len(self.selected_features_)

        logger.info(f"Optimal number of features: {optimal_features}")
        logger.info(f"Removed {removed_count} features")
        logger.info(f"Best CV score: {np.max(self.cv_scores_):.4f}")

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting optimal subset.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted before transform")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fit selector and transform features.

        Args:
            X: Training features
            y: Target variable
            sample_weight: Optional sample weights

        Returns:
            Transformed features
        """
        return self.fit(X, y, sample_weight).transform(X)

    def get_feature_rankings(self) -> pd.DataFrame:
        """Get feature rankings from RFECV.

        Returns:
            DataFrame with feature rankings
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted first")

        rankings_df = pd.DataFrame({
            'feature': self.feature_rankings_.index,
            'ranking': self.feature_rankings_.values,
            'selected': [f in self.selected_features_ for f in self.feature_rankings_.index]
        }).sort_values('ranking')

        return rankings_df

    def plot_cv_scores(self, save_path: Optional[str] = None):
        """Plot cross-validation scores vs number of features.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted first")

        try:
            import matplotlib.pyplot as plt

            n_features_range = range(self.min_features_to_select, 
                                   len(self.cv_scores_) + self.min_features_to_select)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(n_features_range, self.cv_scores_, 'o-')
            ax.axvline(x=self.selector.n_features_, color='red', linestyle='--',
                      label=f'Optimal: {self.selector.n_features_} features')

            ax.set_xlabel('Number of Features')
            ax.set_ylabel(f'Cross-Validation Score ({self.scoring})')
            ax.set_title('RFECV: Feature Selection Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"RFECV plot saved to {save_path}")

            plt.tight_layout()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")


class BorutaFeatureSelector:
    """Boruta feature selection algorithm.

    All-relevant feature selection method that finds features that are
    statistically relevant for prediction.

    Example:
        >>> selector = BorutaFeatureSelector(max_iter=100)
        >>> X_selected = selector.fit_transform(X_train, y_train)
    """

    def __init__(self,
                 estimator=None,
                 n_estimators: int = 1000,
                 max_iter: int = 100,
                 alpha: float = 0.05,
                 two_step: bool = True,
                 random_state: int = 42):
        """Initialize Boruta selector.

        Args:
            estimator: Estimator with feature_importances_ attribute
            n_estimators: Number of estimators in ensemble methods
            max_iter: Maximum number of iterations
            alpha: Significance level for statistical tests
            two_step: Whether to use two-step correction
            random_state: Random state for reproducibility
        """
        if not BORUTA_AVAILABLE:
            raise ImportError("Boruta not available. Install with: pip install boruta")

        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )

        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.two_step = two_step
        self.random_state = random_state

        self.selector = None
        self.selected_features_ = None
        self.tentative_features_ = None
        self.rejected_features_ = None
        self.feature_rankings_ = None
        self.fitted_ = False

    @timer
    def fit(self, X: pd.DataFrame, y: pd.Series,
           sample_weight: Optional[np.ndarray] = None) -> 'BorutaFeatureSelector':
        """Fit the Boruta selector.

        Args:
            X: Training features
            y: Target variable
            sample_weight: Optional sample weights (not directly supported by Boruta)

        Returns:
            Fitted selector
        """
        logger.info(f"Fitting Boruta selector with {len(X.columns)} features")
        logger.info(f"Max iterations: {self.max_iter}, Alpha: {self.alpha}")

        if sample_weight is not None:
            logger.warning("Boruta does not directly support sample weights")

        # Create Boruta selector
        self.selector = BorutaPy(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            max_iter=self.max_iter,
            alpha=self.alpha,
            two_step=self.two_step,
            random_state=self.random_state,
            verbose=1
        )

        # Fit selector
        self.selector.fit(X.values, y.values)

        # Get results
        selected_mask = self.selector.support_
        tentative_mask = self.selector.support_weak_

        self.selected_features_ = X.columns[selected_mask].tolist()
        self.tentative_features_ = X.columns[tentative_mask].tolist()
        self.rejected_features_ = X.columns[
            ~(selected_mask | tentative_mask)
        ].tolist()

        self.feature_rankings_ = pd.Series(
            self.selector.ranking_,
            index=X.columns
        ).sort_values()

        logger.info(f"Selected {len(self.selected_features_)} confirmed features")
        logger.info(f"Tentative features: {len(self.tentative_features_)}")
        logger.info(f"Rejected features: {len(self.rejected_features_)}")

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, include_tentative: bool = False) -> pd.DataFrame:
        """Transform features by selecting confirmed features.

        Args:
            X: Features to transform
            include_tentative: Whether to include tentative features

        Returns:
            Transformed features
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted before transform")

        features_to_select = self.selected_features_.copy()

        if include_tentative:
            features_to_select.extend(self.tentative_features_)

        return X[features_to_select]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     sample_weight: Optional[np.ndarray] = None,
                     include_tentative: bool = False) -> pd.DataFrame:
        """Fit selector and transform features.

        Args:
            X: Training features
            y: Target variable
            sample_weight: Optional sample weights
            include_tentative: Whether to include tentative features

        Returns:
            Transformed features
        """
        return self.fit(X, y, sample_weight).transform(X, include_tentative)

    def get_feature_decision(self) -> pd.DataFrame:
        """Get Boruta feature selection decisions.

        Returns:
            DataFrame with feature decisions
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted first")

        decisions = []
        for feature in self.feature_rankings_.index:
            if feature in self.selected_features_:
                decision = 'Confirmed'
            elif feature in self.tentative_features_:
                decision = 'Tentative'
            else:
                decision = 'Rejected'
            decisions.append(decision)

        decision_df = pd.DataFrame({
            'feature': self.feature_rankings_.index,
            'ranking': self.feature_rankings_.values,
            'decision': decisions
        }).sort_values('ranking')

        return decision_df


class ComprehensiveFeatureSelector:
    """Comprehensive feature selection combining multiple methods.

    Applies multiple feature selection techniques in sequence and
    provides comparison of results.

    Example:
        >>> selector = ComprehensiveFeatureSelector()
        >>> results = selector.fit_transform_all(X_train, y_train)
        >>> best_features = selector.get_consensus_features()
    """

    def __init__(self,
                 variance_threshold: float = 0.01,
                 rfecv_params: Optional[Dict] = None,
                 boruta_params: Optional[Dict] = None,
                 use_boruta: bool = True):
        """Initialize comprehensive selector.

        Args:
            variance_threshold: Threshold for variance filtering
            rfecv_params: Parameters for RFECV selector
            boruta_params: Parameters for Boruta selector
            use_boruta: Whether to include Boruta (if available)
        """
        self.variance_threshold = variance_threshold
        self.rfecv_params = rfecv_params or {}
        self.boruta_params = boruta_params or {}
        self.use_boruta = use_boruta and BORUTA_AVAILABLE

        # Initialize selectors
        self.variance_selector = VarianceFeatureSelector(variance_threshold)
        self.rfecv_selector = RFECVFeatureSelector(**self.rfecv_params)

        if self.use_boruta:
            self.boruta_selector = BorutaFeatureSelector(**self.boruta_params)
        else:
            self.boruta_selector = None

        self.results_ = {}
        self.fitted_ = False

    @timer
    def fit_transform_all(self, X: pd.DataFrame, y: pd.Series,
                         sample_weight: Optional[np.ndarray] = None) -> Dict[str, pd.DataFrame]:
        """Apply all feature selection methods.

        Args:
            X: Training features
            y: Target variable
            sample_weight: Optional sample weights

        Returns:
            Dictionary with results from each method
        """
        logger.info("Running comprehensive feature selection")
        logger.info(f"Initial features: {len(X.columns)}")

        results = {}

        # 1. Variance filtering
        logger.info("Step 1: Variance filtering")
        X_variance = self.variance_selector.fit_transform(X)
        results['variance'] = X_variance

        # 2. RFECV on variance-filtered features
        logger.info("Step 2: RFECV selection")
        X_rfecv = self.rfecv_selector.fit_transform(X_variance, y, sample_weight)
        results['rfecv'] = X_rfecv

        # 3. Boruta on variance-filtered features (if available)
        if self.use_boruta:
            logger.info("Step 3: Boruta selection")
            try:
                X_boruta = self.boruta_selector.fit_transform(X_variance, y, sample_weight)
                results['boruta'] = X_boruta
                results['boruta_with_tentative'] = self.boruta_selector.transform(
                    X_variance, include_tentative=True
                )
            except Exception as e:
                logger.warning(f"Boruta selection failed: {e}")
                results['boruta'] = X_variance

        self.results_ = results
        self.fitted_ = True

        # Log summary
        for method, X_selected in results.items():
            logger.info(f"{method}: {len(X_selected.columns)} features")

        return results

    def get_consensus_features(self, min_agreement: int = 2) -> List[str]:
        """Get features selected by multiple methods.

        Args:
            min_agreement: Minimum number of methods that must agree

        Returns:
            List of consensus features
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted first")

        # Count how many methods selected each feature
        feature_counts = {}

        for method, X_selected in self.results_.items():
            if method == 'boruta_with_tentative':
                continue  # Skip tentative for consensus

            for feature in X_selected.columns:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Get features with sufficient agreement
        consensus_features = [
            feature for feature, count in feature_counts.items()
            if count >= min_agreement
        ]

        logger.info(f"Consensus features (min_agreement={min_agreement}): {len(consensus_features)}")

        return consensus_features

    def get_selection_summary(self) -> pd.DataFrame:
        """Get summary of feature selection results.

        Returns:
            DataFrame comparing selection methods
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted first")

        summary_data = []

        for method, X_selected in self.results_.items():
            summary_data.append({
                'method': method,
                'n_features_selected': len(X_selected.columns),
                'features': sorted(X_selected.columns.tolist())
            })

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def save_results(self, output_dir: str) -> None:
        """Save feature selection results to files.

        Args:
            output_dir: Directory to save results
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted first")

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save selected features for each method
        for method, X_selected in self.results_.items():
            filename = f"{method}_selected_features.csv"
            filepath = os.path.join(output_dir, filename)

            feature_df = pd.DataFrame({
                'feature': X_selected.columns,
                'selected': True
            })
            save_dataframe(feature_df, filepath)

        # Save summary
        summary_df = self.get_selection_summary()
        save_dataframe(summary_df, os.path.join(output_dir, "selection_summary.csv"))

        # Save consensus features
        consensus_features = self.get_consensus_features()
        consensus_df = pd.DataFrame({
            'feature': consensus_features,
            'consensus': True
        })
        save_dataframe(consensus_df, os.path.join(output_dir, "consensus_features.csv"))

        logger.info(f"Feature selection results saved to {output_dir}")


# Convenience functions
def select_features_variance(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Quick variance-based feature selection.

    Args:
        X: Feature matrix
        threshold: Variance threshold

    Returns:
        Selected features
    """
    selector = VarianceFeatureSelector(threshold=threshold)
    return selector.fit_transform(X)


def select_features_rfecv(X: pd.DataFrame, y: pd.Series,
                         estimator=None, cv: int = 5,
                         sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Quick RFECV feature selection.

    Args:
        X: Feature matrix
        y: Target vector
        estimator: Estimator to use
        cv: Cross-validation folds
        sample_weight: Optional sample weights

    Returns:
        Selected features
    """
    selector = RFECVFeatureSelector(estimator=estimator, cv=cv)
    return selector.fit_transform(X, y, sample_weight)


def select_features_boruta(X: pd.DataFrame, y: pd.Series,
                          max_iter: int = 100,
                          sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Quick Boruta feature selection.

    Args:
        X: Feature matrix
        y: Target vector
        max_iter: Maximum iterations
        sample_weight: Optional sample weights

    Returns:
        Selected features
    """
    if not BORUTA_AVAILABLE:
        raise ImportError("Boruta not available")

    selector = BorutaFeatureSelector(max_iter=max_iter)
    return selector.fit_transform(X, y, sample_weight)


def comprehensive_feature_selection(X: pd.DataFrame, y: pd.Series,
                                   sample_weight: Optional[np.ndarray] = None,
                                   output_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Run comprehensive feature selection with all methods.

    Args:
        X: Feature matrix
        y: Target vector
        sample_weight: Optional sample weights
        output_dir: Optional directory to save results

    Returns:
        Dictionary with results from all methods
    """
    selector = ComprehensiveFeatureSelector()
    results = selector.fit_transform_all(X, y, sample_weight)

    if output_dir:
        selector.save_results(output_dir)

    return results
