# tree_models/data/validator.py
"""Enhanced data validation system with comprehensive quality assurance.

This module provides comprehensive data validation capabilities with:
- Type-safe data quality assessment and anomaly detection
- Schema validation and data type consistency checking
- Statistical validation with distribution analysis and drift detection
- Missing value analysis and data completeness assessment
- Outlier detection using multiple statistical methods
- Feature correlation analysis and multicollinearity detection
- Sample weights validation and distribution analysis
- Comprehensive reporting and visualization of validation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from datetime import datetime
import json

from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    DataValidationError,
    ValidationError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

# Optional dependencies with fallbacks
try:
    from scipy import stats
    from scipy.stats import jarque_bera, shapiro, kstest, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available for advanced statistical tests")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available for outlier detection")


@dataclass
class ValidationConfig:
    """Type-safe configuration for data validation with comprehensive options."""
    
    # Basic validation settings
    check_missing_values: bool = True
    check_data_types: bool = True
    check_duplicates: bool = True
    check_outliers: bool = True
    
    # Statistical validation
    check_distributions: bool = True
    check_correlation: bool = True
    correlation_threshold: float = 0.95
    
    # Outlier detection settings
    outlier_methods: List[str] = field(default_factory=lambda: ['iqr', 'zscore', 'isolation'])
    outlier_threshold: float = 3.0
    isolation_contamination: float = 0.1
    
    # Missing value analysis
    missing_threshold: float = 0.5  # Flag if >50% missing
    missing_pattern_analysis: bool = True
    
    # Sample weights validation
    validate_sample_weights: bool = True
    weights_distribution_check: bool = True
    
    # Schema validation
    enforce_schema: bool = False
    expected_schema: Optional[Dict[str, str]] = None
    
    # Distribution analysis
    normality_tests: bool = True
    distribution_comparison: bool = False  # Compare train vs test
    
    # Reporting settings
    generate_report: bool = True
    include_visualizations: bool = True
    save_plots: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        validate_parameter("correlation_threshold", self.correlation_threshold, min_value=0.0, max_value=1.0)
        validate_parameter("outlier_threshold", self.outlier_threshold, min_value=0.5, max_value=10.0)
        validate_parameter("missing_threshold", self.missing_threshold, min_value=0.0, max_value=1.0)
        validate_parameter("isolation_contamination", self.isolation_contamination, min_value=0.01, max_value=0.5)
        
        valid_outlier_methods = ['iqr', 'zscore', 'isolation', 'elliptic', 'modified_zscore']
        for method in self.outlier_methods:
            if method not in valid_outlier_methods:
                raise ValidationError(f"Unknown outlier method: {method}")


@dataclass
class ValidationResults:
    """Comprehensive validation results with detailed findings and recommendations."""
    
    # Basic info
    dataset_name: str
    n_samples: int
    n_features: int
    validation_timestamp: str
    
    # Validation status
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Data quality metrics
    data_quality_score: Optional[float] = None
    missing_value_analysis: Dict[str, Any] = field(default_factory=dict)
    outlier_analysis: Dict[str, Any] = field(default_factory=dict)
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    distribution_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Schema validation results
    schema_validation: Dict[str, Any] = field(default_factory=dict)
    data_type_issues: List[str] = field(default_factory=list)
    
    # Statistical tests
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    # Sample weights validation
    weights_validation: Optional[Dict[str, Any]] = None
    
    # Feature-level analysis
    feature_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Validation config used
    config: Optional[ValidationConfig] = None


class DataValidator:
    """Enhanced data validator with comprehensive quality assurance capabilities.
    
    Provides systematic data validation with statistical analysis, quality
    assessment, and detailed reporting for machine learning workflows.
    
    Example:
        >>> validator = DataValidator()
        >>> 
        >>> # Configure validation
        >>> config = ValidationConfig(
        ...     check_outliers=True,
        ...     check_distributions=True,
        ...     generate_report=True
        ... )
        >>> 
        >>> # Validate dataset
        >>> results = validator.validate_dataset(
        ...     df, target_column='target',
        ...     sample_weight=weights,
        ...     config=config
        ... )
        >>> 
        >>> print(f"Data quality score: {results.data_quality_score:.2f}")
        >>> print(f"Validation errors: {len(results.validation_errors)}")
        >>> 
        >>> # Generate detailed report
        >>> validator.generate_validation_report(results, "validation_report")
    """
    
    def __init__(
        self,
        random_state: int = 42,
        enable_logging: bool = True
    ) -> None:
        """Initialize enhanced data validator.
        
        Args:
            random_state: Random state for reproducibility
            enable_logging: Whether to enable detailed logging
        """
        self.random_state = random_state
        self.enable_logging = enable_logging
        
        # Set random seeds
        np.random.seed(random_state)
        
        logger.info(f"Initialized DataValidator with random_state={random_state}")

    @timer(name="data_validation")
    def validate_dataset(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        sample_weight: Optional[np.ndarray] = None,
        reference_data: Optional[pd.DataFrame] = None,
        config: Optional[ValidationConfig] = None,
        dataset_name: str = "dataset"
    ) -> ValidationResults:
        """Comprehensive dataset validation with quality assessment.
        
        Args:
            data: DataFrame to validate
            target_column: Name of target column (if present)
            sample_weight: Optional sample weights for validation
            reference_data: Optional reference dataset for comparison
            config: Validation configuration
            dataset_name: Name for the dataset (used in reports)
            
        Returns:
            Comprehensive validation results
            
        Raises:
            ValidationError: If validation configuration is invalid
        """
        logger.info(f"🔍 Starting comprehensive data validation:")
        logger.info(f"   Dataset: {dataset_name}")
        logger.info(f"   Shape: {data.shape}")
        logger.info(f"   Target column: {target_column}")
        logger.info(f"   Has sample weights: {sample_weight is not None}")
        
        if config is None:
            config = ValidationConfig()
        
        start_time = datetime.now()
        
        try:
            with timed_operation("data_validation") as timing:
                # Initialize results
                results = ValidationResults(
                    dataset_name=dataset_name,
                    n_samples=len(data),
                    n_features=data.shape[1],
                    validation_timestamp=start_time.isoformat(),
                    config=config
                )
                
                # Basic data validation
                self._validate_basic_properties(data, results, config)
                
                # Schema validation
                if config.check_data_types or config.enforce_schema:
                    self._validate_schema(data, results, config)
                
                # Missing value analysis
                if config.check_missing_values:
                    self._analyze_missing_values(data, results, config)
                
                # Outlier detection
                if config.check_outliers:
                    self._detect_outliers(data, results, config, target_column)
                
                # Correlation analysis
                if config.check_correlation:
                    self._analyze_correlations(data, results, config, target_column)
                
                # Distribution analysis
                if config.check_distributions:
                    self._analyze_distributions(data, results, config, target_column)
                
                # Sample weights validation
                if sample_weight is not None and config.validate_sample_weights:
                    self._validate_sample_weights(sample_weight, data, results, config)
                
                # Feature profiling
                self._profile_features(data, results, config, target_column)
                
                # Data drift detection (if reference data provided)
                if reference_data is not None and config.distribution_comparison:
                    self._detect_data_drift(data, reference_data, results, config)
                
                # Compute overall data quality score
                self._compute_data_quality_score(results)
                
                # Generate recommendations
                self._generate_recommendations(results, config)
                
                # Set final validation status
                results.is_valid = len(results.validation_errors) == 0
            
            logger.info(f"✅ Data validation completed:")
            logger.info(f"   Duration: {timing['duration']:.2f}s")
            logger.info(f"   Quality score: {results.data_quality_score:.2f}")
            logger.info(f"   Errors: {len(results.validation_errors)}")
            logger.info(f"   Warnings: {len(results.validation_warnings)}")
            
            return results
            
        except Exception as e:
            handle_and_reraise(
                e, ValidationError,
                f"Data validation failed for {dataset_name}",
                error_code="DATA_VALIDATION_FAILED",
                context=create_error_context(
                    dataset_name=dataset_name,
                    data_shape=data.shape,
                    has_target=target_column is not None,
                    has_weights=sample_weight is not None
                )
            )

    def _validate_basic_properties(
        self,
        data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig
    ) -> None:
        """Validate basic data properties."""
        
        logger.debug("Validating basic data properties")
        
        # Check if data is empty
        if data.empty:
            results.validation_errors.append("Dataset is empty")
            return
        
        # Check for duplicate rows
        if config.check_duplicates:
            duplicate_count = data.duplicated().sum()
            duplicate_percentage = (duplicate_count / len(data)) * 100
            
            if duplicate_count > 0:
                if duplicate_percentage > 5:  # More than 5% duplicates
                    results.validation_errors.append(
                        f"High number of duplicate rows: {duplicate_count} ({duplicate_percentage:.1f}%)"
                    )
                else:
                    results.validation_warnings.append(
                        f"Duplicate rows found: {duplicate_count} ({duplicate_percentage:.1f}%)"
                    )
        
        # Check for constant columns
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            results.validation_warnings.append(f"Constant columns found: {constant_columns}")
        
        # Check for extremely imbalanced features
        imbalanced_columns = []
        for col in data.select_dtypes(include=['object', 'category']).columns:
            value_counts = data[col].value_counts()
            if len(value_counts) > 1:
                # Check if most frequent value is >95% of data
                max_frequency = value_counts.iloc[0] / len(data)
                if max_frequency > 0.95:
                    imbalanced_columns.append(col)
        
        if imbalanced_columns:
            results.validation_warnings.append(f"Highly imbalanced categorical columns: {imbalanced_columns}")

    def _validate_schema(
        self,
        data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig
    ) -> None:
        """Validate data schema and types."""
        
        logger.debug("Validating data schema and types")
        
        schema_issues = []
        
        # Check for mixed data types in columns
        for col in data.columns:
            try:
                # Try to identify inconsistent data types
                if data[col].dtype == 'object':
                    # For object columns, check if they contain mixed types
                    sample_values = data[col].dropna().head(100)
                    type_counts = {}
                    
                    for value in sample_values:
                        value_type = type(value).__name__
                        type_counts[value_type] = type_counts.get(value_type, 0) + 1
                    
                    if len(type_counts) > 1:
                        most_common_type = max(type_counts, key=type_counts.get)
                        if type_counts[most_common_type] / len(sample_values) < 0.9:
                            schema_issues.append(f"Column '{col}' has mixed data types: {type_counts}")
                
            except Exception as e:
                logger.warning(f"Error checking data types for column '{col}': {e}")
        
        # Schema enforcement
        if config.enforce_schema and config.expected_schema:
            for col, expected_type in config.expected_schema.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if actual_type != expected_type:
                        schema_issues.append(
                            f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}"
                        )
                else:
                    schema_issues.append(f"Expected column '{col}' not found in data")
        
        # Check for suspicious column names
        suspicious_names = []
        for col in data.columns:
            if any(char in col for char in ['/', '\\', '?', '*', '[', ']']):
                suspicious_names.append(col)
        
        if suspicious_names:
            results.validation_warnings.append(f"Columns with suspicious characters: {suspicious_names}")
        
        if schema_issues:
            results.data_type_issues = schema_issues
            results.validation_warnings.extend(schema_issues)
        
        results.schema_validation = {
            'schema_issues': schema_issues,
            'suspicious_column_names': suspicious_names,
            'total_columns': len(data.columns),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns)
        }

    def _analyze_missing_values(
        self,
        data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig
    ) -> None:
        """Comprehensive missing value analysis."""
        
        logger.debug("Analyzing missing values")
        
        # Basic missing value statistics
        missing_stats = {}
        total_missing = 0
        
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_percentage = (missing_count / len(data)) * 100
            
            missing_stats[col] = {
                'count': missing_count,
                'percentage': missing_percentage
            }
            
            total_missing += missing_count
            
            # Flag columns with high missing rates
            if missing_percentage > config.missing_threshold * 100:
                results.validation_errors.append(
                    f"Column '{col}' has high missing rate: {missing_percentage:.1f}%"
                )
            elif missing_percentage > 20:  # Warning for >20% missing
                results.validation_warnings.append(
                    f"Column '{col}' has significant missing values: {missing_percentage:.1f}%"
                )
        
        # Missing pattern analysis
        missing_patterns = {}
        if config.missing_pattern_analysis and total_missing > 0:
            # Analyze missing patterns across columns
            missing_mask = data.isnull()
            
            # Find columns that are often missing together
            if len(data.columns) > 1:
                correlation_matrix = missing_mask.astype(int).corr()
                
                # Find high correlations in missingness
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        corr = correlation_matrix.iloc[i, j]
                        if abs(corr) > 0.5:  # Strong correlation in missingness
                            col1 = correlation_matrix.columns[i]
                            col2 = correlation_matrix.columns[j]
                            high_corr_pairs.append((col1, col2, corr))
                
                if high_corr_pairs:
                    missing_patterns['correlated_missingness'] = high_corr_pairs
        
        # Overall missing data assessment
        total_cells = len(data) * len(data.columns)
        overall_missing_rate = (total_missing / total_cells) * 100
        
        if overall_missing_rate > 30:
            results.validation_errors.append(
                f"Dataset has very high overall missing rate: {overall_missing_rate:.1f}%"
            )
        elif overall_missing_rate > 10:
            results.validation_warnings.append(
                f"Dataset has high overall missing rate: {overall_missing_rate:.1f}%"
            )
        
        results.missing_value_analysis = {
            'column_stats': missing_stats,
            'overall_missing_rate': overall_missing_rate,
            'total_missing_values': total_missing,
            'columns_with_missing': sum(1 for stats in missing_stats.values() if stats['count'] > 0),
            'patterns': missing_patterns
        }

    def _detect_outliers(
        self,
        data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig,
        target_column: Optional[str] = None
    ) -> None:
        """Comprehensive outlier detection using multiple methods."""
        
        logger.debug(f"Detecting outliers using methods: {config.outlier_methods}")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(target_column)
        
        if len(numeric_columns) == 0:
            results.outlier_analysis = {'message': 'No numeric columns for outlier detection'}
            return
        
        outlier_results = {}
        
        for method in config.outlier_methods:
            try:
                method_results = {}
                
                if method == 'iqr':
                    method_results = self._detect_outliers_iqr(data[numeric_columns], config.outlier_threshold)
                elif method == 'zscore':
                    method_results = self._detect_outliers_zscore(data[numeric_columns], config.outlier_threshold)
                elif method == 'modified_zscore':
                    method_results = self._detect_outliers_modified_zscore(data[numeric_columns], config.outlier_threshold)
                elif method == 'isolation' and SKLEARN_AVAILABLE:
                    method_results = self._detect_outliers_isolation(data[numeric_columns], config.isolation_contamination)
                elif method == 'elliptic' and SKLEARN_AVAILABLE:
                    method_results = self._detect_outliers_elliptic(data[numeric_columns], config.isolation_contamination)
                
                outlier_results[method] = method_results
                
                # Add warnings for columns with high outlier rates
                for col, outlier_info in method_results.items():
                    if isinstance(outlier_info, dict) and 'percentage' in outlier_info:
                        if outlier_info['percentage'] > 10:
                            results.validation_warnings.append(
                                f"Column '{col}' has high outlier rate ({method}): {outlier_info['percentage']:.1f}%"
                            )
                
            except Exception as e:
                logger.warning(f"Failed to detect outliers using {method}: {e}")
                outlier_results[method] = {'error': str(e)}
        
        results.outlier_analysis = outlier_results

    def _detect_outliers_iqr(self, data: pd.DataFrame, threshold: float = 1.5) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range method."""
        
        results = {}
        
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_percentage = (len(outliers) / len(col_data)) * 100
            
            results[col] = {
                'count': len(outliers),
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': outliers.index.tolist()
            }
        
        return results

    def _detect_outliers_zscore(self, data: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        
        results = {}
        
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            z_scores = np.abs(stats.zscore(col_data)) if SCIPY_AVAILABLE else np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = col_data[z_scores > threshold]
            outlier_percentage = (len(outliers) / len(col_data)) * 100
            
            results[col] = {
                'count': len(outliers),
                'percentage': outlier_percentage,
                'threshold': threshold,
                'outlier_indices': outliers.index.tolist()
            }
        
        return results

    def _detect_outliers_modified_zscore(self, data: pd.DataFrame, threshold: float = 3.5) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score method (using median)."""
        
        results = {}
        
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))  # Median Absolute Deviation
            
            if mad == 0:
                # Use standard deviation as fallback
                mad = col_data.std()
            
            if mad > 0:
                modified_z_scores = 0.6745 * (col_data - median) / mad
                outliers = col_data[np.abs(modified_z_scores) > threshold]
            else:
                outliers = pd.Series([], dtype=col_data.dtype)
            
            outlier_percentage = (len(outliers) / len(col_data)) * 100
            
            results[col] = {
                'count': len(outliers),
                'percentage': outlier_percentage,
                'threshold': threshold,
                'outlier_indices': outliers.index.tolist()
            }
        
        return results

    def _detect_outliers_isolation(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest."""
        
        try:
            # Remove rows with any missing values for isolation forest
            clean_data = data.dropna()
            
            if len(clean_data) == 0:
                return {'error': 'No complete cases for isolation forest'}
            
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=self.random_state
            )
            
            outlier_labels = iso_forest.fit_predict(clean_data)
            outliers = clean_data[outlier_labels == -1]
            
            outlier_percentage = (len(outliers) / len(clean_data)) * 100
            
            return {
                'multivariate_outliers': {
                    'count': len(outliers),
                    'percentage': outlier_percentage,
                    'contamination': contamination,
                    'outlier_indices': outliers.index.tolist()
                }
            }
        
        except Exception as e:
            return {'error': f'Isolation forest failed: {str(e)}'}

    def _detect_outliers_elliptic(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect outliers using Elliptic Envelope."""
        
        try:
            # Remove rows with any missing values
            clean_data = data.dropna()
            
            if len(clean_data) == 0:
                return {'error': 'No complete cases for elliptic envelope'}
            
            elliptic = EllipticEnvelope(
                contamination=contamination,
                random_state=self.random_state
            )
            
            outlier_labels = elliptic.fit_predict(clean_data)
            outliers = clean_data[outlier_labels == -1]
            
            outlier_percentage = (len(outliers) / len(clean_data)) * 100
            
            return {
                'multivariate_outliers': {
                    'count': len(outliers),
                    'percentage': outlier_percentage,
                    'contamination': contamination,
                    'outlier_indices': outliers.index.tolist()
                }
            }
        
        except Exception as e:
            return {'error': f'Elliptic envelope failed: {str(e)}'}

    def _analyze_correlations(
        self,
        data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig,
        target_column: Optional[str] = None
    ) -> None:
        """Analyze feature correlations and multicollinearity."""
        
        logger.debug("Analyzing correlations and multicollinearity")
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            results.correlation_analysis = {'message': 'Insufficient numeric columns for correlation analysis'}
            return
        
        try:
            # Compute correlation matrix
            correlation_matrix = numeric_data.corr()
            
            # Find high correlations
            high_correlations = []
            correlation_warnings = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    
                    if abs(corr) > config.correlation_threshold:
                        high_correlations.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': corr
                        })
                        correlation_warnings.append(
                            f"High correlation between '{col1}' and '{col2}': {corr:.3f}"
                        )
            
            # Add correlation warnings
            if correlation_warnings:
                results.validation_warnings.extend(correlation_warnings)
            
            # Target correlation analysis
            target_correlations = {}
            if target_column and target_column in numeric_data.columns:
                target_corr = correlation_matrix[target_column].drop(target_column)
                
                # Find features with very low correlation to target
                low_target_corr = target_corr[abs(target_corr) < 0.05]
                if len(low_target_corr) > 0:
                    results.validation_warnings.append(
                        f"Features with very low target correlation: {low_target_corr.index.tolist()}"
                    )
                
                target_correlations = {
                    'all_correlations': target_corr.to_dict(),
                    'high_positive': target_corr[target_corr > 0.5].to_dict(),
                    'high_negative': target_corr[target_corr < -0.5].to_dict(),
                    'low_correlation': low_target_corr.to_dict()
                }
            
            results.correlation_analysis = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': high_correlations,
                'target_correlations': target_correlations,
                'max_correlation': float(correlation_matrix.abs().max().max()) if not correlation_matrix.empty else 0,
                'n_high_correlations': len(high_correlations)
            }
        
        except Exception as e:
            logger.warning(f"Error in correlation analysis: {e}")
            results.correlation_analysis = {'error': str(e)}

    def _analyze_distributions(
        self,
        data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig,
        target_column: Optional[str] = None
    ) -> None:
        """Analyze feature distributions and normality."""
        
        logger.debug("Analyzing feature distributions")
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            results.distribution_analysis = {'message': 'No numeric columns for distribution analysis'}
            return
        
        distribution_results = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            col_analysis = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'range': float(col_data.max() - col_data.min()),
                'n_unique': int(col_data.nunique()),
                'zeros_percentage': float((col_data == 0).sum() / len(col_data) * 100)
            }
            
            # Normality tests
            if config.normality_tests and SCIPY_AVAILABLE and len(col_data) > 3:
                try:
                    # Shapiro-Wilk test (for smaller samples)
                    if len(col_data) <= 5000:
                        shapiro_stat, shapiro_p = shapiro(col_data.sample(min(5000, len(col_data))))
                        col_analysis['shapiro_test'] = {
                            'statistic': float(shapiro_stat),
                            'p_value': float(shapiro_p),
                            'is_normal': shapiro_p > 0.05
                        }
                    
                    # Jarque-Bera test
                    if len(col_data) > 7:
                        jb_stat, jb_p = jarque_bera(col_data)
                        col_analysis['jarque_bera_test'] = {
                            'statistic': float(jb_stat),
                            'p_value': float(jb_p),
                            'is_normal': jb_p > 0.05
                        }
                
                except Exception as e:
                    logger.warning(f"Error in normality tests for {col}: {e}")
            
            # Distribution shape warnings
            if abs(col_analysis['skewness']) > 2:
                results.validation_warnings.append(
                    f"Column '{col}' is highly skewed (skewness: {col_analysis['skewness']:.2f})"
                )
            
            if abs(col_analysis['kurtosis']) > 7:
                results.validation_warnings.append(
                    f"Column '{col}' has high kurtosis (kurtosis: {col_analysis['kurtosis']:.2f})"
                )
            
            distribution_results[col] = col_analysis
        
        results.distribution_analysis = distribution_results

    def _validate_sample_weights(
        self,
        sample_weight: np.ndarray,
        data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig
    ) -> None:
        """Validate sample weights distribution and properties."""
        
        logger.debug("Validating sample weights")
        
        weights_analysis = {}
        
        # Basic statistics
        weights_analysis['statistics'] = {
            'mean': float(np.mean(sample_weight)),
            'median': float(np.median(sample_weight)),
            'std': float(np.std(sample_weight)),
            'min': float(np.min(sample_weight)),
            'max': float(np.max(sample_weight)),
            'sum': float(np.sum(sample_weight)),
            'n_unique': int(len(np.unique(sample_weight)))
        }
        
        # Check for negative weights
        negative_weights = np.sum(sample_weight < 0)
        if negative_weights > 0:
            results.validation_errors.append(f"Sample weights contain {negative_weights} negative values")
        
        # Check for zero weights
        zero_weights = np.sum(sample_weight == 0)
        if zero_weights > 0:
            zero_percentage = (zero_weights / len(sample_weight)) * 100
            if zero_percentage > 5:
                results.validation_warnings.append(f"High percentage of zero weights: {zero_percentage:.1f}%")
        
        # Check for extreme weights
        if weights_analysis['statistics']['std'] > 0:
            weight_cv = weights_analysis['statistics']['std'] / weights_analysis['statistics']['mean']
            if weight_cv > 3:
                results.validation_warnings.append(f"High weight variability (CV: {weight_cv:.2f})")
        
        # Weight distribution analysis
        if config.weights_distribution_check:
            # Check for outliers in weights
            Q1 = np.percentile(sample_weight, 25)
            Q3 = np.percentile(sample_weight, 75)
            IQR = Q3 - Q1
            
            outlier_threshold = 3.0
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            weight_outliers = np.sum((sample_weight < lower_bound) | (sample_weight > upper_bound))
            outlier_percentage = (weight_outliers / len(sample_weight)) * 100
            
            weights_analysis['outliers'] = {
                'count': int(weight_outliers),
                'percentage': float(outlier_percentage),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            if outlier_percentage > 5:
                results.validation_warnings.append(f"High percentage of weight outliers: {outlier_percentage:.1f}%")
        
        results.weights_validation = weights_analysis

    def _profile_features(
        self,
        data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig,
        target_column: Optional[str] = None
    ) -> None:
        """Create detailed profiles for each feature."""
        
        logger.debug("Profiling individual features")
        
        feature_profiles = {}
        
        for col in data.columns:
            if col == target_column:
                continue
                
            profile = {
                'dtype': str(data[col].dtype),
                'non_null_count': int(data[col].count()),
                'null_count': int(data[col].isnull().sum()),
                'null_percentage': float(data[col].isnull().sum() / len(data) * 100),
                'unique_count': int(data[col].nunique()),
                'unique_percentage': float(data[col].nunique() / len(data) * 100)
            }
            
            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(data[col]):
                non_null_data = data[col].dropna()
                if len(non_null_data) > 0:
                    profile.update({
                        'mean': float(non_null_data.mean()),
                        'median': float(non_null_data.median()),
                        'std': float(non_null_data.std()),
                        'min': float(non_null_data.min()),
                        'max': float(non_null_data.max()),
                        'q25': float(non_null_data.quantile(0.25)),
                        'q75': float(non_null_data.quantile(0.75))
                    })
            
            elif pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                non_null_data = data[col].dropna()
                if len(non_null_data) > 0:
                    value_counts = non_null_data.value_counts()
                    profile.update({
                        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                        'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'most_frequent_percentage': float(value_counts.iloc[0] / len(non_null_data) * 100) if len(value_counts) > 0 else 0,
                        'top_5_values': value_counts.head().to_dict()
                    })
            
            # Data quality flags
            quality_issues = []
            
            if profile['null_percentage'] > 50:
                quality_issues.append('high_missing_rate')
            
            if profile['unique_count'] == 1:
                quality_issues.append('constant_feature')
            
            if pd.api.types.is_numeric_dtype(data[col]):
                if 'std' in profile and profile['std'] == 0:
                    quality_issues.append('zero_variance')
                
                # Check for potential ID columns
                if profile['unique_percentage'] > 95 and profile['unique_count'] > 100:
                    quality_issues.append('potential_id_column')
            
            profile['quality_issues'] = quality_issues
            feature_profiles[col] = profile
        
        results.feature_profiles = feature_profiles

    def _detect_data_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        results: ValidationResults,
        config: ValidationConfig
    ) -> None:
        """Detect data drift between current and reference datasets."""
        
        logger.debug("Detecting data drift between datasets")
        
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available for drift detection")
            return
        
        drift_results = {}
        common_columns = set(current_data.columns) & set(reference_data.columns)
        
        for col in common_columns:
            try:
                col_drift = {}
                
                current_col = current_data[col].dropna()
                reference_col = reference_data[col].dropna()
                
                if len(current_col) == 0 or len(reference_col) == 0:
                    continue
                
                if pd.api.types.is_numeric_dtype(current_data[col]):
                    # Kolmogorov-Smirnov test for numeric features
                    ks_stat, ks_p = ks_2samp(current_col, reference_col)
                    
                    col_drift.update({
                        'test': 'kolmogorov_smirnov',
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p),
                        'has_drift': ks_p < 0.05,
                        'drift_magnitude': 'high' if ks_stat > 0.2 else 'medium' if ks_stat > 0.1 else 'low'
                    })
                
                else:
                    # Chi-square test for categorical features
                    current_counts = current_col.value_counts()
                    reference_counts = reference_col.value_counts()
                    
                    # Align categories
                    all_categories = set(current_counts.index) | set(reference_counts.index)
                    
                    current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
                    reference_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(current_aligned) > 0 and sum(reference_aligned) > 0:
                        chi2_stat, chi2_p, _, _ = chi2_contingency([current_aligned, reference_aligned])
                        
                        col_drift.update({
                            'test': 'chi_square',
                            'statistic': float(chi2_stat),
                            'p_value': float(chi2_p),
                            'has_drift': chi2_p < 0.05
                        })
                
                if col_drift.get('has_drift', False):
                    results.validation_warnings.append(f"Data drift detected in column '{col}'")
                
                drift_results[col] = col_drift
            
            except Exception as e:
                logger.warning(f"Error detecting drift for column {col}: {e}")
                drift_results[col] = {'error': str(e)}
        
        # Add drift summary to results
        total_features_checked = len(drift_results)
        features_with_drift = sum(1 for drift in drift_results.values() if drift.get('has_drift', False))
        
        drift_summary = {
            'total_features_checked': total_features_checked,
            'features_with_drift': features_with_drift,
            'drift_percentage': (features_with_drift / total_features_checked * 100) if total_features_checked > 0 else 0,
            'feature_drift_details': drift_results
        }
        
        if 'statistical_tests' not in results.statistical_tests:
            results.statistical_tests = {}
        results.statistical_tests['data_drift'] = drift_summary

    def _compute_data_quality_score(self, results: ValidationResults) -> None:
        """Compute overall data quality score (0-100)."""
        
        score_components = []
        
        # Missing data component (0-25 points)
        if results.missing_value_analysis:
            missing_rate = results.missing_value_analysis.get('overall_missing_rate', 0)
            missing_score = max(0, 25 - (missing_rate / 100 * 25))
            score_components.append(('missing_data', missing_score))
        
        # Outlier component (0-25 points)
        if results.outlier_analysis:
            avg_outlier_rate = 0
            outlier_counts = 0
            
            for method, method_results in results.outlier_analysis.items():
                if isinstance(method_results, dict) and 'error' not in method_results:
                    for col, col_results in method_results.items():
                        if isinstance(col_results, dict) and 'percentage' in col_results:
                            avg_outlier_rate += col_results['percentage']
                            outlier_counts += 1
            
            if outlier_counts > 0:
                avg_outlier_rate /= outlier_counts
                outlier_score = max(0, 25 - (avg_outlier_rate / 100 * 25))
                score_components.append(('outliers', outlier_score))
        
        # Schema/type consistency (0-20 points)
        type_issues = len(results.data_type_issues)
        type_score = max(0, 20 - (type_issues * 5))
        score_components.append(('data_types', type_score))
        
        # Correlation issues (0-15 points)
        if results.correlation_analysis and 'high_correlations' in results.correlation_analysis:
            high_corr_count = len(results.correlation_analysis['high_correlations'])
            corr_score = max(0, 15 - (high_corr_count * 2))
            score_components.append(('correlations', corr_score))
        
        # Validation errors penalty (0-15 points)
        error_count = len(results.validation_errors)
        error_score = max(0, 15 - (error_count * 3))
        score_components.append(('validation_errors', error_score))
        
        # Calculate final score
        total_score = sum(score for _, score in score_components)
        max_possible_score = sum([25, 25, 20, 15, 15])  # Adjust if components change
        
        results.data_quality_score = min(100, (total_score / max_possible_score) * 100)

    def _generate_recommendations(self, results: ValidationResults, config: ValidationConfig) -> None:
        """Generate actionable recommendations based on validation results."""
        
        recommendations = []
        
        # Missing data recommendations
        if results.missing_value_analysis:
            high_missing_columns = [
                col for col, stats in results.missing_value_analysis['column_stats'].items()
                if stats['percentage'] > 20
            ]
            
            if high_missing_columns:
                recommendations.append(
                    f"Consider imputing or removing columns with high missing rates: {high_missing_columns}"
                )
        
        # Outlier recommendations
        if results.outlier_analysis:
            recommendations.append(
                "Review detected outliers to determine if they are data errors or legitimate extreme values"
            )
        
        # Correlation recommendations
        if results.correlation_analysis and 'high_correlations' in results.correlation_analysis:
            if len(results.correlation_analysis['high_correlations']) > 0:
                recommendations.append(
                    "Consider removing one feature from highly correlated pairs to reduce multicollinearity"
                )
        
        # Distribution recommendations
        skewed_features = []
        if results.distribution_analysis:
            for col, dist_info in results.distribution_analysis.items():
                if isinstance(dist_info, dict) and 'skewness' in dist_info:
                    if abs(dist_info['skewness']) > 2:
                        skewed_features.append(col)
        
        if skewed_features:
            recommendations.append(
                f"Consider transforming highly skewed features: {skewed_features}"
            )
        
        # Constant feature recommendations
        constant_features = []
        if results.feature_profiles:
            for col, profile in results.feature_profiles.items():
                if 'constant_feature' in profile.get('quality_issues', []):
                    constant_features.append(col)
        
        if constant_features:
            recommendations.append(f"Remove constant features: {constant_features}")
        
        # Sample weights recommendations
        if results.weights_validation:
            if results.weights_validation.get('outliers', {}).get('percentage', 0) > 10:
                recommendations.append("Review sample weights for potential outliers that may bias the model")
        
        # General recommendations based on quality score
        if results.data_quality_score is not None:
            if results.data_quality_score < 60:
                recommendations.append(
                    "Data quality score is low. Consider comprehensive data cleaning before modeling"
                )
            elif results.data_quality_score < 80:
                recommendations.append(
                    "Data quality is moderate. Address key issues identified in the validation"
                )
        
        results.recommendations = recommendations

    def generate_validation_report(
        self,
        results: ValidationResults,
        output_dir: Union[str, Path],
        include_plots: bool = True
    ) -> Dict[str, Path]:
        """Generate comprehensive validation report.
        
        Args:
            results: Validation results to report on
            output_dir: Directory to save report files
            include_plots: Whether to include visualization plots
            
        Returns:
            Dictionary of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        try:
            # Save validation summary
            summary_path = output_dir / "validation_summary.json"
            
            summary_data = {
                'dataset_name': results.dataset_name,
                'validation_timestamp': results.validation_timestamp,
                'is_valid': results.is_valid,
                'data_quality_score': results.data_quality_score,
                'n_samples': results.n_samples,
                'n_features': results.n_features,
                'n_errors': len(results.validation_errors),
                'n_warnings': len(results.validation_warnings),
                'validation_errors': results.validation_errors,
                'validation_warnings': results.validation_warnings,
                'recommendations': results.recommendations
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            generated_files['summary'] = summary_path
            
            # Save detailed analysis results
            detailed_path = output_dir / "detailed_analysis.json"
            
            detailed_data = {
                'missing_value_analysis': results.missing_value_analysis,
                'outlier_analysis': results.outlier_analysis,
                'correlation_analysis': results.correlation_analysis,
                'distribution_analysis': results.distribution_analysis,
                'schema_validation': results.schema_validation,
                'feature_profiles': results.feature_profiles,
                'weights_validation': results.weights_validation,
                'statistical_tests': results.statistical_tests
            }
            
            with open(detailed_path, 'w') as f:
                json.dump(detailed_data, f, indent=2, default=str)
            generated_files['detailed_analysis'] = detailed_path
            
            # Generate plots if requested
            if include_plots and results.config and results.config.include_visualizations:
                plot_files = self._generate_validation_plots(results, output_dir)
                generated_files.update(plot_files)
            
            logger.info(f"Generated validation report in {output_dir}")
            logger.info(f"Files created: {list(generated_files.keys())}")
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return {}

    def _generate_validation_plots(
        self,
        results: ValidationResults,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Generate validation visualization plots."""
        
        plot_files = {}
        
        try:
            # Missing values heatmap
            if results.missing_value_analysis and 'column_stats' in results.missing_value_analysis:
                missing_plot_path = self._plot_missing_values(results.missing_value_analysis, output_dir)
                if missing_plot_path:
                    plot_files['missing_values'] = missing_plot_path
            
            # Correlation heatmap
            if (results.correlation_analysis and 
                'correlation_matrix' in results.correlation_analysis and 
                results.correlation_analysis['correlation_matrix']):
                corr_plot_path = self._plot_correlation_matrix(results.correlation_analysis, output_dir)
                if corr_plot_path:
                    plot_files['correlation_matrix'] = corr_plot_path
            
            # Data quality summary
            quality_plot_path = self._plot_data_quality_summary(results, output_dir)
            if quality_plot_path:
                plot_files['data_quality_summary'] = quality_plot_path
                
        except Exception as e:
            logger.warning(f"Error generating validation plots: {e}")
        
        return plot_files

    def _plot_missing_values(self, missing_analysis: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Generate missing values visualization."""
        try:
            column_stats = missing_analysis.get('column_stats', {})
            
            if not column_stats:
                return None
            
            columns = list(column_stats.keys())
            percentages = [stats['percentage'] for stats in column_stats.values()]
            
            plt.figure(figsize=(12, max(6, len(columns) * 0.3)))
            bars = plt.barh(columns, percentages, color='skyblue', edgecolor='navy')
            
            # Add percentage labels
            for bar, pct in zip(bars, percentages):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{pct:.1f}%', va='center', ha='left')
            
            plt.xlabel('Missing Percentage (%)')
            plt.title('Missing Values by Column')
            plt.tight_layout()
            
            output_path = output_dir / 'missing_values.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Failed to create missing values plot: {e}")
            plt.close()
            return None

    def _plot_correlation_matrix(self, correlation_analysis: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Generate correlation matrix heatmap."""
        try:
            corr_matrix = correlation_analysis.get('correlation_matrix', {})
            
            if not corr_matrix:
                return None
            
            # Convert to DataFrame for plotting
            corr_df = pd.DataFrame(corr_matrix)
            
            if corr_df.empty:
                return None
            
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
            
            sns.heatmap(corr_df, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, fmt='.2f')
            
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            output_path = output_dir / 'correlation_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Failed to create correlation matrix plot: {e}")
            plt.close()
            return None

    def _plot_data_quality_summary(self, results: ValidationResults, output_dir: Path) -> Optional[Path]:
        """Generate data quality summary visualization."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Quality score gauge
            score = results.data_quality_score or 0
            ax1.pie([score, 100-score], labels=['Quality', 'Issues'], 
                   colors=['green' if score > 80 else 'orange' if score > 60 else 'red', 'lightgray'],
                   startangle=90, counterclock=False)
            ax1.set_title(f'Data Quality Score: {score:.1f}/100')
            
            # Errors and warnings
            error_types = ['Errors', 'Warnings']
            error_counts = [len(results.validation_errors), len(results.validation_warnings)]
            
            bars = ax2.bar(error_types, error_counts, color=['red', 'orange'])
            ax2.set_title('Validation Issues')
            ax2.set_ylabel('Count')
            
            # Add count labels on bars
            for bar, count in zip(bars, error_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
            
            # Missing data overview
            if results.missing_value_analysis:
                missing_rate = results.missing_value_analysis.get('overall_missing_rate', 0)
                complete_rate = 100 - missing_rate
                
                ax3.pie([complete_rate, missing_rate], labels=['Complete', 'Missing'],
                       colors=['lightblue', 'lightcoral'], autopct='%1.1f%%')
                ax3.set_title('Data Completeness')
            
            # Feature quality distribution
            if results.feature_profiles:
                quality_categories = ['Good', 'Issues', 'Poor']
                quality_counts = [0, 0, 0]
                
                for profile in results.feature_profiles.values():
                    issues = len(profile.get('quality_issues', []))
                    if issues == 0:
                        quality_counts[0] += 1
                    elif issues <= 2:
                        quality_counts[1] += 1
                    else:
                        quality_counts[2] += 1
                
                ax4.bar(quality_categories, quality_counts, 
                       color=['green', 'orange', 'red'])
                ax4.set_title('Feature Quality Distribution')
                ax4.set_ylabel('Number of Features')
            
            plt.suptitle(f'Data Validation Summary - {results.dataset_name}', fontsize=16)
            plt.tight_layout()
            
            output_path = output_dir / 'data_quality_summary.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Failed to create data quality summary plot: {e}")
            plt.close()
            return None


# Convenience functions
def validate_dataset(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    **kwargs: Any
) -> ValidationResults:
    """Convenience function for dataset validation.
    
    Args:
        data: DataFrame to validate
        target_column: Name of target column (if present)
        sample_weight: Optional sample weights
        **kwargs: Additional validation parameters
        
    Returns:
        Validation results object
        
    Example:
        >>> results = validate_dataset(df, target_column='target', sample_weight=weights)
        >>> print(f"Data quality score: {results.data_quality_score:.2f}")
        >>> print(f"Validation passed: {results.is_valid}")
    """
    validator = DataValidator()
    return validator.validate_dataset(data, target_column, sample_weight, **kwargs)


def quick_data_check(
    data: pd.DataFrame,
    generate_report: bool = False,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Quick data quality check with summary results.
    
    Args:
        data: DataFrame to check
        generate_report: Whether to generate detailed report
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with quick summary results
        
    Example:
        >>> summary = quick_data_check(df, generate_report=True, output_dir='data_check/')
        >>> print(f"Overall quality: {summary['quality_score']:.1f}/100")
        >>> print(f"Issues found: {summary['total_issues']}")
    """
    config = ValidationConfig(
        generate_report=generate_report,
        include_visualizations=generate_report
    )
    
    validator = DataValidator()
    results = validator.validate_dataset(data, config=config)
    
    if generate_report and output_dir:
        validator.generate_validation_report(results, output_dir)
    
    return {
        'quality_score': results.data_quality_score,
        'is_valid': results.is_valid,
        'total_issues': len(results.validation_errors) + len(results.validation_warnings),
        'errors': len(results.validation_errors),
        'warnings': len(results.validation_warnings),
        'missing_rate': results.missing_value_analysis.get('overall_missing_rate', 0),
        'recommendations': results.recommendations
    }


# Export key classes and functions
__all__ = [
    'ValidationConfig',
    'ValidationResults',
    'DataValidator',
    'validate_dataset',
    'quick_data_check'
]