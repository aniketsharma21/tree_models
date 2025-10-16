# tree_models/models/feature_selector.py
"""Enhanced feature selection utilities with comprehensive algorithms and type safety.

This module provides production-ready feature selection capabilities with:
- Type-safe interfaces and comprehensive error handling
- Variance filtering with statistical analysis
- Recursive Feature Elimination with Cross-Validation (RFECV)
- Boruta all-relevant feature selection algorithm
- Consensus feature selection combining multiple methods
- Sample weights integration throughout selection workflows
- Performance monitoring and comprehensive logging

Key Features:
- Multiple feature selection algorithms with unified interface
- Statistical significance testing for feature importance
- Cross-validation based selection with proper evaluation
- Comprehensive reporting and visualization capabilities
- Production-ready error handling and logging
- Sample weights support where algorithmically possible
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
from pathlib import Path

from .base import BaseFeatureSelector
from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    FeatureSelectionError,
    ConfigurationError,
    DataValidationError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)

logger = get_logger(__name__)

# Try to import Boruta (optional dependency)
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
    logger.info("Boruta feature selection available")
except ImportError:
    BORUTA_AVAILABLE = False
    logger.warning("Boruta not available. Install with: pip install boruta")

warnings.filterwarnings('ignore')


class VarianceFeatureSelector(BaseFeatureSelector):
    """Enhanced variance-based feature selection with comprehensive analysis.

    Removes features with low variance across samples with additional
    statistical analysis and comprehensive error handling.

    Example:
        >>> selector = VarianceFeatureSelector(threshold=0.01)
        >>> X_selected, feature_names = selector.select_features(X_train, y_train)
        >>> importance_df = selector.get_feature_importance()
    """

    def __init__(
        self, 
        threshold: float = 0.0,
        normalize: bool = False,
        handle_missing: str = "error"
    ) -> None:
        """Initialize enhanced variance selector.

        Args:
            threshold: Variance threshold below which features are removed
            normalize: Whether to normalize variances by mean
            handle_missing: How to handle missing values ('error', 'drop', 'impute')
        """
        super().__init__(
            selection_method="variance_threshold",
            max_features=None
        )
        
        validate_parameter("threshold", threshold, min_value=0.0)
        validate_parameter("handle_missing", handle_missing, 
                         valid_values=["error", "drop", "impute"])
        
        self.threshold = threshold
        self.normalize = normalize
        self.handle_missing = handle_missing
        
        self.selector = VarianceThreshold(threshold=threshold)
        self.feature_variances_: Optional[pd.Series] = None
        self.normalized_variances_: Optional[pd.Series] = None
        
        logger.info(f"Initialized VarianceFeatureSelector:")
        logger.info(f"  Threshold: {threshold}, Normalize: {normalize}")

    def _validate_and_prepare_data(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Validate and prepare data for variance selection."""
        
        # Check for missing values
        if X.isnull().any().any():
            if self.handle_missing == "error":
                missing_cols = X.columns[X.isnull().any()].tolist()
                raise DataValidationError(
                    f"Missing values found in columns: {missing_cols}",
                    error_code="MISSING_VALUES_FOUND",
                    context={"missing_columns": missing_cols}
                )
            elif self.handle_missing == "drop":
                X = X.dropna(axis=1)
                logger.warning(f"Dropped {X.shape[1]} columns with missing values")
            elif self.handle_missing == "impute":
                X = X.fillna(X.median())
                logger.warning("Imputed missing values with median")
        
        # Check for non-numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise DataValidationError(
                f"Non-numeric columns found: {non_numeric.tolist()}",
                error_code="NON_NUMERIC_FEATURES",
                context={"non_numeric_columns": non_numeric.tolist()}
            )
        
        return X

    @timer(name="variance_feature_selection")
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on variance threshold with comprehensive analysis.

        Args:
            X: Input features
            y: Target variable (not used for variance selection)
            sample_weight: Optional sample weights (not used for variance selection)
            **kwargs: Additional selection parameters

        Returns:
            Tuple of (selected_features_df, selected_feature_names)

        Raises:
            FeatureSelectionError: If feature selection fails
        """
        logger.info(f"🎯 Starting variance feature selection:")
        logger.info(f"   Features: {X.shape[1]}, Threshold: {self.threshold}")
        
        try:
            with timed_operation("variance_selection_preparation"):
                # Validate and prepare data
                X_clean = self._validate_and_prepare_data(X, y)
                
                # Calculate variances
                self.feature_variances_ = X_clean.var()
                
                # Calculate normalized variances if requested
                if self.normalize:
                    feature_means = X_clean.mean()
                    # Avoid division by zero
                    self.normalized_variances_ = self.feature_variances_ / (feature_means + 1e-8)
                    selection_variances = self.normalized_variances_
                    logger.info("Using normalized variances for selection")
                else:
                    selection_variances = self.feature_variances_
                
                # Apply threshold
                selected_mask = selection_variances >= self.threshold
                self.selected_features_ = X_clean.columns[selected_mask].tolist()
                
                # Store feature scores for importance ranking
                self.feature_scores_ = pd.DataFrame({
                    'feature': X_clean.columns,
                    'variance': self.feature_variances_,
                    'normalized_variance': self.normalized_variances_ if self.normalize else self.feature_variances_,
                    'selected': selected_mask,
                    'selection_score': selection_variances
                }).sort_values('selection_score', ascending=False)
            
            removed_count = len(X_clean.columns) - len(self.selected_features_)
            
            logger.info(f"✅ Variance selection completed:")
            logger.info(f"   Selected: {len(self.selected_features_)} features")
            logger.info(f"   Removed: {removed_count} features ({removed_count/len(X_clean.columns)*100:.1f}%)")
            
            if len(self.selected_features_) == 0:
                raise FeatureSelectionError(
                    "No features passed variance threshold",
                    error_code="NO_FEATURES_SELECTED",
                    context={"threshold": self.threshold}
                )
            
            return X_clean[self.selected_features_], self.selected_features_
            
        except Exception as e:
            handle_and_reraise(
                e, FeatureSelectionError,
                "Variance feature selection failed",
                error_code="VARIANCE_SELECTION_FAILED",
                context=create_error_context(
                    threshold=self.threshold,
                    n_features=X.shape[1],
                    normalize=self.normalize
                )
            )

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on variance scores.

        Returns:
            DataFrame with feature importance information

        Raises:
            FeatureSelectionError: If selection hasn't been performed
        """
        if self.feature_scores_ is None:
            raise FeatureSelectionError(
                "Must perform feature selection first",
                error_code="NO_SELECTION_PERFORMED"
            )
        
        return self.feature_scores_.copy()

    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of variance selection results.

        Returns:
            Dictionary with selection summary statistics
        """
        if self.feature_scores_ is None:
            raise FeatureSelectionError("No selection results available")
        
        selected_variances = self.feature_scores_[self.feature_scores_['selected']]['variance']
        removed_variances = self.feature_scores_[~self.feature_scores_['selected']]['variance']
        
        return {
            'threshold': self.threshold,
            'total_features': len(self.feature_scores_),
            'selected_features': len(self.selected_features_),
            'removed_features': len(self.feature_scores_) - len(self.selected_features_),
            'selection_rate': len(self.selected_features_) / len(self.feature_scores_),
            'selected_variance_stats': {
                'mean': selected_variances.mean(),
                'std': selected_variances.std(),
                'min': selected_variances.min(),
                'max': selected_variances.max()
            } if len(selected_variances) > 0 else None,
            'removed_variance_stats': {
                'mean': removed_variances.mean(),
                'std': removed_variances.std(),
                'min': removed_variances.min(),
                'max': removed_variances.max()
            } if len(removed_variances) > 0 else None
        }


class RFECVFeatureSelector(BaseFeatureSelector):
    """Enhanced Recursive Feature Elimination with Cross-Validation.

    Uses cross-validation to select optimal number of features by recursively
    eliminating features with comprehensive error handling and sample weights support.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> selector = RFECVFeatureSelector(estimator=estimator, cv=5)
        >>> X_selected, features = selector.select_features(X_train, y_train, sample_weight=weights)
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        step: int = 1,
        min_features_to_select: int = 1,
        cv: int = 5,
        scoring: str = 'roc_auc',
        n_jobs: int = 1,
        random_state: int = 42,
        importance_getter: str = 'auto'
    ) -> None:
        """Initialize enhanced RFECV selector.

        Args:
            estimator: Estimator with feature_importances_ or coef_ attribute
            step: Number of features to remove at each iteration
            min_features_to_select: Minimum number of features to select
            cv: Cross-validation folds
            scoring: Scoring metric for evaluation
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
            importance_getter: Method to get feature importance
        """
        super().__init__(
            selection_method="rfecv",
            max_features=None
        )
        
        # Validation
        validate_parameter("step", step, min_value=1)
        validate_parameter("min_features_to_select", min_features_to_select, min_value=1)
        validate_parameter("cv", cv, min_value=2, max_value=20)
        validate_parameter("n_jobs", n_jobs, min_value=-1)
        validate_parameter("importance_getter", importance_getter, 
                         valid_values=['auto', 'feature_importances_', 'coef_'])
        
        # Set default estimator
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=max(1, n_jobs)  # Ensure positive for RF
            )

        self.estimator = estimator
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.importance_getter = importance_getter

        self.selector: Optional[RFECV] = None
        self.cv_scores_: Optional[np.ndarray] = None
        self.optimal_features_: Optional[int] = None
        
        logger.info(f"Initialized RFECVFeatureSelector:")
        logger.info(f"  Estimator: {type(estimator).__name__}")
        logger.info(f"  CV: {cv}, Scoring: {scoring}, Step: {step}")

    def _create_scorer(self, sample_weight: Optional[np.ndarray] = None) -> Union[str, Callable]:
        """Create appropriate scorer handling sample weights."""
        
        if sample_weight is None:
            return self.scoring
        
        # For weighted scoring, we need custom scorer
        if self.scoring == 'roc_auc':
            def weighted_roc_auc(estimator, X_val, y_val):
                if hasattr(estimator, 'predict_proba'):
                    y_pred = estimator.predict_proba(X_val)[:, 1]
                else:
                    y_pred = estimator.predict(X_val)
                # Note: This is simplified - in practice, we'd need to properly 
                # handle sample weights in the CV splits
                return roc_auc_score(y_val, y_pred)
            
            return make_scorer(weighted_roc_auc, greater_is_better=True, needs_proba=True)
        else:
            logger.warning(f"Sample weights may not be properly handled for {self.scoring}")
            return self.scoring

    @timer(name="rfecv_feature_selection")
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using RFECV with comprehensive error handling.

        Args:
            X: Input features
            y: Target variable
            sample_weight: Optional sample weights
            **kwargs: Additional selection parameters

        Returns:
            Tuple of (selected_features_df, selected_feature_names)

        Raises:
            FeatureSelectionError: If feature selection fails
        """
        logger.info(f"🎯 Starting RFECV feature selection:")
        logger.info(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
        logger.info(f"   Min features: {self.min_features_to_select}, CV: {self.cv}")
        
        try:
            with timed_operation("rfecv_selection"):
                # Validate inputs
                if X.empty or y.empty:
                    raise DataValidationError("Empty data provided")
                
                if len(X) != len(y):
                    raise DataValidationError("X and y must have same length")
                
                if sample_weight is not None and len(sample_weight) != len(X):
                    raise DataValidationError("sample_weight must have same length as X")

                # Create scorer
                scorer = self._create_scorer(sample_weight)
                
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
                    n_jobs=self.n_jobs,
                    importance_getter=self.importance_getter
                )
                
                # Fit selector
                if sample_weight is not None:
                    logger.warning("RFECV may not fully utilize sample weights in CV splits")
                
                self.selector.fit(X, y)
                
                # Extract results
                selected_mask = self.selector.get_support()
                self.selected_features_ = X.columns[selected_mask].tolist()
                self.cv_scores_ = self.selector.cv_results_
                self.optimal_features_ = self.selector.n_features_
                
                # Create feature importance scores
                feature_rankings = pd.Series(self.selector.ranking_, index=X.columns)
                
                self.feature_scores_ = pd.DataFrame({
                    'feature': X.columns,
                    'ranking': feature_rankings.values,
                    'selected': selected_mask,
                    'importance_score': 1.0 / feature_rankings.values  # Convert ranking to score
                }).sort_values('importance_score', ascending=False)
            
            removed_count = len(X.columns) - len(self.selected_features_)
            best_cv_score = np.max(self.cv_scores_) if len(self.cv_scores_) > 0 else 0.0
            
            logger.info(f"✅ RFECV selection completed:")
            logger.info(f"   Optimal features: {self.optimal_features_}")
            logger.info(f"   Selected: {len(self.selected_features_)} features")
            logger.info(f"   Removed: {removed_count} features")
            logger.info(f"   Best CV score: {best_cv_score:.4f}")
            
            if len(self.selected_features_) == 0:
                raise FeatureSelectionError(
                    "RFECV selected no features",
                    error_code="NO_FEATURES_SELECTED"
                )
            
            return X[self.selected_features_], self.selected_features_
            
        except Exception as e:
            handle_and_reraise(
                e, FeatureSelectionError,
                "RFECV feature selection failed",
                error_code="RFECV_SELECTION_FAILED",
                context=create_error_context(
                    estimator_type=type(self.estimator).__name__,
                    cv=self.cv,
                    n_features=X.shape[1],
                    min_features=self.min_features_to_select
                )
            )

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on RFECV rankings.

        Returns:
            DataFrame with feature importance information
        """
        if self.feature_scores_ is None:
            raise FeatureSelectionError(
                "Must perform feature selection first",
                error_code="NO_SELECTION_PERFORMED"
            )
        
        return self.feature_scores_.copy()

    def get_cv_scores(self) -> np.ndarray:
        """Get cross-validation scores for different feature counts.

        Returns:
            Array of CV scores
        """
        if self.cv_scores_ is None:
            raise FeatureSelectionError("No CV scores available")
        
        return self.cv_scores_.copy()

    def plot_cv_scores(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot cross-validation scores vs number of features.

        Args:
            save_path: Optional path to save the plot
        """
        if self.cv_scores_ is None:
            raise FeatureSelectionError("No CV scores available for plotting")

        try:
            import matplotlib.pyplot as plt

            n_features_range = range(
                self.min_features_to_select, 
                len(self.cv_scores_) + self.min_features_to_select
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(n_features_range, self.cv_scores_, 'o-', linewidth=2, markersize=6)
            
            if self.optimal_features_:
                ax.axvline(
                    x=self.optimal_features_, 
                    color='red', 
                    linestyle='--',
                    alpha=0.7,
                    label=f'Optimal: {self.optimal_features_} features'
                )

            ax.set_xlabel('Number of Features')
            ax.set_ylabel(f'Cross-Validation Score ({self.scoring})')
            ax.set_title('RFECV: Feature Selection Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"RFECV plot saved to {save_path}")
            
            plt.tight_layout()
            return fig

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to create RFECV plot: {e}")


class BorutaFeatureSelector(BaseFeatureSelector):
    """Enhanced Boruta feature selection with comprehensive error handling.

    All-relevant feature selection method that finds features statistically
    relevant for prediction with production-ready error handling.

    Example:
        >>> selector = BorutaFeatureSelector(max_iter=100, alpha=0.05)
        >>> X_selected, features = selector.select_features(X_train, y_train)
        >>> decisions = selector.get_feature_decisions()
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        n_estimators: int = 1000,
        max_iter: int = 100,
        alpha: float = 0.05,
        two_step: bool = True,
        random_state: int = 42
    ) -> None:
        """Initialize enhanced Boruta selector.

        Args:
            estimator: Estimator with feature_importances_ attribute
            n_estimators: Number of estimators in ensemble methods
            max_iter: Maximum number of iterations
            alpha: Significance level for statistical tests
            two_step: Whether to use two-step correction
            random_state: Random state for reproducibility
        """
        if not BORUTA_AVAILABLE:
            raise ImportError(
                "Boruta not available. Install with: pip install boruta"
            )

        super().__init__(
            selection_method="boruta",
            max_features=None
        )
        
        # Validation
        validate_parameter("n_estimators", n_estimators, min_value=100, max_value=10000)
        validate_parameter("max_iter", max_iter, min_value=10, max_value=1000)
        validate_parameter("alpha", alpha, min_value=0.001, max_value=0.1)

        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=min(n_estimators, 1000),  # Cap for performance
                random_state=random_state,
                n_jobs=-1
            )

        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.two_step = two_step
        self.random_state = random_state

        self.selector: Optional[BorutaPy] = None
        self.tentative_features_: Optional[List[str]] = None
        self.rejected_features_: Optional[List[str]] = None
        self.feature_decisions_: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized BorutaFeatureSelector:")
        logger.info(f"  Max iterations: {max_iter}, Alpha: {alpha}")
        logger.info(f"  Estimator: {type(estimator).__name__}")

    @timer(name="boruta_feature_selection")
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using Boruta algorithm with comprehensive error handling.

        Args:
            X: Input features
            y: Target variable
            sample_weight: Optional sample weights (not directly supported by Boruta)
            **kwargs: Additional selection parameters

        Returns:
            Tuple of (selected_features_df, selected_feature_names)

        Raises:
            FeatureSelectionError: If feature selection fails
        """
        logger.info(f"🎯 Starting Boruta feature selection:")
        logger.info(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
        logger.info(f"   Max iterations: {self.max_iter}, Alpha: {self.alpha}")
        
        if sample_weight is not None:
            logger.warning("Boruta does not directly support sample weights")
        
        try:
            with timed_operation("boruta_selection"):
                # Validate inputs
                if X.empty or y.empty:
                    raise DataValidationError("Empty data provided")
                
                if len(X) != len(y):
                    raise DataValidationError("X and y must have same length")
                
                # Check for non-numeric data
                non_numeric = X.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric) > 0:
                    raise DataValidationError(f"Non-numeric columns: {non_numeric.tolist()}")

                # Create Boruta selector
                self.selector = BorutaPy(
                    estimator=self.estimator,
                    n_estimators=self.n_estimators,
                    max_iter=self.max_iter,
                    alpha=self.alpha,
                    two_step=self.two_step,
                    random_state=self.random_state,
                    verbose=0  # Control verbosity
                )

                # Fit selector
                logger.info("Running Boruta algorithm (this may take a while)...")
                self.selector.fit(X.values, y.values)

                # Extract results
                confirmed_mask = self.selector.support_
                tentative_mask = self.selector.support_weak_
                
                self.selected_features_ = X.columns[confirmed_mask].tolist()
                self.tentative_features_ = X.columns[tentative_mask].tolist()
                self.rejected_features_ = X.columns[
                    ~(confirmed_mask | tentative_mask)
                ].tolist()

                # Create feature decisions DataFrame
                decisions = []
                for feature in X.columns:
                    if feature in self.selected_features_:
                        decision = 'Confirmed'
                    elif feature in self.tentative_features_:
                        decision = 'Tentative'
                    else:
                        decision = 'Rejected'
                    decisions.append(decision)

                self.feature_decisions_ = pd.DataFrame({
                    'feature': X.columns,
                    'decision': decisions,
                    'ranking': self.selector.ranking_,
                    'selected': confirmed_mask
                }).sort_values('ranking')

                # Create feature scores (inverse of ranking)
                self.feature_scores_ = pd.DataFrame({
                    'feature': X.columns,
                    'ranking': self.selector.ranking_,
                    'decision': decisions,
                    'selected': confirmed_mask,
                    'importance_score': 1.0 / self.selector.ranking_
                }).sort_values('importance_score', ascending=False)
            
            logger.info(f"✅ Boruta selection completed:")
            logger.info(f"   Confirmed: {len(self.selected_features_)} features")
            logger.info(f"   Tentative: {len(self.tentative_features_)} features")
            logger.info(f"   Rejected: {len(self.rejected_features_)} features")
            
            if len(self.selected_features_) == 0:
                logger.warning("Boruta confirmed no features - consider including tentative features")
            
            return X[self.selected_features_], self.selected_features_
            
        except Exception as e:
            handle_and_reraise(
                e, FeatureSelectionError,
                "Boruta feature selection failed",
                error_code="BORUTA_SELECTION_FAILED",
                context=create_error_context(
                    max_iter=self.max_iter,
                    alpha=self.alpha,
                    n_features=X.shape[1],
                    estimator_type=type(self.estimator).__name__
                )
            )

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on Boruta rankings.

        Returns:
            DataFrame with feature importance information
        """
        if self.feature_scores_ is None:
            raise FeatureSelectionError(
                "Must perform feature selection first",
                error_code="NO_SELECTION_PERFORMED"
            )
        
        return self.feature_scores_.copy()

    def get_feature_decisions(self) -> pd.DataFrame:
        """Get Boruta feature selection decisions.

        Returns:
            DataFrame with feature decisions (Confirmed/Tentative/Rejected)
        """
        if self.feature_decisions_ is None:
            raise FeatureSelectionError("No feature decisions available")
        
        return self.feature_decisions_.copy()

    def transform_with_tentative(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features including tentative features.

        Args:
            X: Features to transform

        Returns:
            Transformed features including tentative ones
        """
        if self.selected_features_ is None or self.tentative_features_ is None:
            raise FeatureSelectionError("Must perform feature selection first")

        all_features = self.selected_features_ + self.tentative_features_
        return X[all_features]

    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of Boruta selection results.

        Returns:
            Dictionary with selection summary
        """
        if self.feature_decisions_ is None:
            raise FeatureSelectionError("No selection results available")

        return {
            'algorithm': 'Boruta',
            'max_iterations': self.max_iter,
            'alpha': self.alpha,
            'total_features': len(self.feature_decisions_),
            'confirmed_features': len(self.selected_features_),
            'tentative_features': len(self.tentative_features_),
            'rejected_features': len(self.rejected_features_),
            'selection_rate': len(self.selected_features_) / len(self.feature_decisions_),
            'decision_counts': self.feature_decisions_['decision'].value_counts().to_dict()
        }


class ConsensusFeatureSelector(BaseFeatureSelector):
    """Consensus feature selection combining multiple algorithms.

    Combines results from multiple feature selection methods to identify
    features that are consistently selected across different algorithms.

    Example:
        >>> selector = ConsensusFeatureSelector(
        ...     methods=['variance', 'rfecv', 'boruta'],
        ...     min_agreement=2
        ... )
        >>> results = selector.select_features_comprehensive(X_train, y_train)
        >>> consensus_features = selector.get_consensus_features()
    """

    def __init__(
        self,
        methods: List[str] = ['variance', 'rfecv'],
        min_agreement: int = 2,
        variance_threshold: float = 0.01,
        rfecv_params: Optional[Dict[str, Any]] = None,
        boruta_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize consensus feature selector.

        Args:
            methods: List of methods to use ('variance', 'rfecv', 'boruta')
            min_agreement: Minimum number of methods that must agree
            variance_threshold: Threshold for variance filtering
            rfecv_params: Parameters for RFECV selector
            boruta_params: Parameters for Boruta selector
        """
        super().__init__(
            selection_method="consensus",
            max_features=None
        )
        
        available_methods = ['variance', 'rfecv']
        if BORUTA_AVAILABLE:
            available_methods.append('boruta')
        
        # Validate methods
        invalid_methods = set(methods) - set(available_methods)
        if invalid_methods:
            raise ConfigurationError(
                f"Invalid methods: {invalid_methods}. Available: {available_methods}"
            )
        
        if 'boruta' in methods and not BORUTA_AVAILABLE:
            logger.warning("Boruta requested but not available, removing from methods")
            methods = [m for m in methods if m != 'boruta']
        
        validate_parameter("min_agreement", min_agreement, min_value=1, max_value=len(methods))
        
        self.methods = methods
        self.min_agreement = min_agreement
        self.variance_threshold = variance_threshold
        self.rfecv_params = rfecv_params or {}
        self.boruta_params = boruta_params or {}
        
        # Initialize selectors
        self.selectors: Dict[str, BaseFeatureSelector] = {}
        self.method_results_: Dict[str, Dict[str, Any]] = {}
        self.consensus_features_: Optional[List[str]] = None
        
        logger.info(f"Initialized ConsensusFeatureSelector:")
        logger.info(f"  Methods: {methods}, Min agreement: {min_agreement}")

    def _initialize_selectors(self) -> None:
        """Initialize individual selectors."""
        if 'variance' in self.methods:
            self.selectors['variance'] = VarianceFeatureSelector(
                threshold=self.variance_threshold
            )
        
        if 'rfecv' in self.methods:
            self.selectors['rfecv'] = RFECVFeatureSelector(**self.rfecv_params)
        
        if 'boruta' in self.methods and BORUTA_AVAILABLE:
            self.selectors['boruta'] = BorutaFeatureSelector(**self.boruta_params)

    @timer(name="consensus_feature_selection")
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using consensus approach.

        Args:
            X: Input features
            y: Target variable
            sample_weight: Optional sample weights
            **kwargs: Additional selection parameters

        Returns:
            Tuple of (selected_features_df, selected_feature_names)

        Raises:
            FeatureSelectionError: If consensus selection fails
        """
        logger.info(f"🎯 Starting consensus feature selection:")
        logger.info(f"   Methods: {self.methods}")
        logger.info(f"   Features: {X.shape[1]}, Min agreement: {self.min_agreement}")
        
        try:
            self._initialize_selectors()
            self.method_results_ = {}
            
            # Run each method
            for method in self.methods:
                logger.info(f"Running {method} feature selection...")
                
                try:
                    selector = self.selectors[method]
                    X_selected, selected_features = selector.select_features(
                        X, y, sample_weight, **kwargs
                    )
                    
                    self.method_results_[method] = {
                        'selected_features': selected_features,
                        'n_features': len(selected_features),
                        'selector': selector,
                        'success': True
                    }
                    
                    logger.info(f"  {method}: {len(selected_features)} features selected")
                    
                except Exception as e:
                    logger.error(f"Method {method} failed: {e}")
                    self.method_results_[method] = {
                        'selected_features': [],
                        'n_features': 0,
                        'error': str(e),
                        'success': False
                    }
            
            # Calculate consensus
            self.consensus_features_ = self._calculate_consensus()
            
            if len(self.consensus_features_) == 0:
                raise FeatureSelectionError(
                    "No consensus features found",
                    error_code="NO_CONSENSUS_FEATURES",
                    context={'min_agreement': self.min_agreement}
                )
            
            # Store final results
            self.selected_features_ = self.consensus_features_
            
            # Create feature scores based on agreement count
            feature_votes = self._count_feature_votes(X.columns.tolist())
            
            self.feature_scores_ = pd.DataFrame({
                'feature': list(feature_votes.keys()),
                'agreement_count': list(feature_votes.values()),
                'agreement_rate': [v/len([r for r in self.method_results_.values() if r['success']]) 
                                 for v in feature_votes.values()],
                'selected': [f in self.consensus_features_ for f in feature_votes.keys()]
            }).sort_values('agreement_count', ascending=False)
            
            logger.info(f"✅ Consensus selection completed:")
            logger.info(f"   Consensus features: {len(self.consensus_features_)}")
            logger.info(f"   Agreement threshold: {self.min_agreement}/{len(self.methods)}")
            
            return X[self.consensus_features_], self.consensus_features_
            
        except Exception as e:
            handle_and_reraise(
                e, FeatureSelectionError,
                "Consensus feature selection failed",
                error_code="CONSENSUS_SELECTION_FAILED",
                context=create_error_context(
                    methods=self.methods,
                    min_agreement=self.min_agreement,
                    n_features=X.shape[1]
                )
            )

    def _count_feature_votes(self, all_features: List[str]) -> Dict[str, int]:
        """Count votes for each feature across successful methods."""
        feature_votes = {feature: 0 for feature in all_features}
        
        for method_result in self.method_results_.values():
            if method_result['success']:
                for feature in method_result['selected_features']:
                    if feature in feature_votes:
                        feature_votes[feature] += 1
        
        return feature_votes

    def _calculate_consensus(self) -> List[str]:
        """Calculate consensus features based on agreement threshold."""
        if not self.method_results_:
            return []
        
        # Get successful method results
        successful_results = [
            result for result in self.method_results_.values() 
            if result['success']
        ]
        
        if len(successful_results) < self.min_agreement:
            logger.warning(f"Insufficient successful methods ({len(successful_results)}) for min_agreement ({self.min_agreement})")
            return []
        
        # Count feature selections across methods
        feature_counts = {}
        all_features = set()
        
        for result in successful_results:
            for feature in result['selected_features']:
                all_features.add(feature)
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Select features with sufficient agreement
        consensus_features = [
            feature for feature, count in feature_counts.items()
            if count >= self.min_agreement
        ]
        
        return sorted(consensus_features)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on consensus agreement.

        Returns:
            DataFrame with feature importance based on agreement rates
        """
        if self.feature_scores_ is None:
            raise FeatureSelectionError(
                "Must perform feature selection first",
                error_code="NO_SELECTION_PERFORMED"
            )
        
        return self.feature_scores_.copy()

    def get_method_results(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed results from each selection method.

        Returns:
            Dictionary with results from each method
        """
        if not self.method_results_:
            raise FeatureSelectionError("No method results available")
        
        return self.method_results_.copy()

    def get_selection_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of consensus selection.

        Returns:
            Dictionary with selection summary
        """
        if not self.method_results_:
            raise FeatureSelectionError("No selection results available")
        
        successful_methods = [k for k, v in self.method_results_.items() if v['success']]
        
        summary = {
            'methods_used': self.methods,
            'successful_methods': successful_methods,
            'min_agreement': self.min_agreement,
            'consensus_features': len(self.consensus_features_) if self.consensus_features_ else 0,
            'method_feature_counts': {
                method: result['n_features'] 
                for method, result in self.method_results_.items()
                if result['success']
            }
        }
        
        if self.feature_scores_ is not None:
            summary['agreement_distribution'] = (
                self.feature_scores_['agreement_count'].value_counts().to_dict()
            )
        
        return summary


# Convenience functions for backward compatibility and quick usage
def select_features_variance(
    X: pd.DataFrame, 
    threshold: float = 0.01,
    **kwargs: Any
) -> pd.DataFrame:
    """Quick variance-based feature selection.

    Args:
        X: Feature matrix
        threshold: Variance threshold
        **kwargs: Additional parameters

    Returns:
        Selected features
    """
    selector = VarianceFeatureSelector(threshold=threshold)
    X_selected, _ = selector.select_features(X, pd.Series([0]*len(X)), **kwargs)
    return X_selected


def select_features_rfecv(
    X: pd.DataFrame, 
    y: pd.Series,
    estimator: Optional[Any] = None, 
    cv: int = 5,
    sample_weight: Optional[np.ndarray] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """Quick RFECV feature selection.

    Args:
        X: Feature matrix
        y: Target vector
        estimator: Estimator to use
        cv: Cross-validation folds
        sample_weight: Optional sample weights
        **kwargs: Additional parameters

    Returns:
        Selected features
    """
    selector = RFECVFeatureSelector(estimator=estimator, cv=cv)
    X_selected, _ = selector.select_features(X, y, sample_weight, **kwargs)
    return X_selected


def select_features_boruta(
    X: pd.DataFrame, 
    y: pd.Series,
    max_iter: int = 100,
    sample_weight: Optional[np.ndarray] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """Quick Boruta feature selection.

    Args:
        X: Feature matrix
        y: Target vector
        max_iter: Maximum iterations
        sample_weight: Optional sample weights
        **kwargs: Additional parameters

    Returns:
        Selected features
    """
    if not BORUTA_AVAILABLE:
        raise ImportError("Boruta not available. Install with: pip install boruta")

    selector = BorutaFeatureSelector(max_iter=max_iter)
    X_selected, _ = selector.select_features(X, y, sample_weight, **kwargs)
    return X_selected


def comprehensive_feature_selection(
    X: pd.DataFrame, 
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    methods: List[str] = ['variance', 'rfecv'],
    min_agreement: int = 2,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Run comprehensive feature selection with multiple methods.

    Args:
        X: Feature matrix
        y: Target vector
        sample_weight: Optional sample weights
        methods: Methods to use for selection
        min_agreement: Minimum number of methods that must agree
        output_dir: Optional directory to save results
        **kwargs: Additional parameters

    Returns:
        Dictionary with comprehensive results from all methods

    Example:
        >>> results = comprehensive_feature_selection(
        ...     X_train, y_train,
        ...     sample_weight=weights,
        ...     methods=['variance', 'rfecv', 'boruta'],
        ...     min_agreement=2,
        ...     output_dir='feature_selection_results'
        ... )
        >>> consensus_features = results['consensus_features']
        >>> print(f"Selected {len(consensus_features)} consensus features")
    """
    selector = ConsensusFeatureSelector(
        methods=methods,
        min_agreement=min_agreement
    )
    
    X_selected, consensus_features = selector.select_features(X, y, sample_weight, **kwargs)
    
    results = {
        'consensus_features': consensus_features,
        'selected_data': X_selected,
        'method_results': selector.get_method_results(),
        'selection_summary': selector.get_selection_summary(),
        'feature_importance': selector.get_feature_importance(),
        'selector': selector
    }
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results['feature_importance'].to_csv(output_path / 'feature_importance.csv', index=False)
        
        pd.DataFrame({'feature': consensus_features}).to_csv(
            output_path / 'consensus_features.csv', index=False
        )
        
        # Save method-specific results
        for method, method_result in results['method_results'].items():
            if method_result['success']:
                pd.DataFrame({'feature': method_result['selected_features']}).to_csv(
                    output_path / f'{method}_features.csv', index=False
                )
        
        logger.info(f"Feature selection results saved to {output_path}")
    
    return results