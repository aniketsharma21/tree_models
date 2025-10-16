# tree_models/explainability/scorecard.py
"""Enhanced scorecard conversion for business-friendly risk scores.

This module provides production-ready scorecard conversion with:
- Type-safe interfaces and comprehensive validation
- Multiple scorecard formats (FICO-style, custom ranges)
- Statistical calibration and validation
- Business interpretation and risk categories
- Sample weights integration for proper calibration
- Comprehensive error handling and logging
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import warnings

from .base import BaseConverter, ensure_dir
from ..utils.logger import get_logger
from ..utils.timer import timer, timed_operation
from ..utils.exceptions import (
    ExplainabilityError,
    ConfigurationError,
    DataValidationError,
    handle_and_reraise,
    validate_parameter,
    create_error_context
)

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class ScorecardConfig:
    """Type-safe configuration for scorecard conversion with validation."""
    
    # Score range
    min_score: int = 300
    max_score: int = 850
    
    # Calibration points (probability -> score mapping)
    odds_at_min: float = 50.0  # 50:1 odds at min score (2% probability)
    odds_at_max: float = 1/50.0  # 1:50 odds at max score (98% probability)
    
    # Points to Double the Odds (PDO)
    pdo: int = 20
    
    # Validation thresholds
    min_probability: float = 1e-7
    max_probability: float = 1 - 1e-7
    
    def __post_init__(self) -> None:
        """Validate scorecard configuration."""
        validate_parameter("min_score", self.min_score, min_value=100, max_value=800)
        validate_parameter("max_score", self.max_score, min_value=200, max_value=1000)
        
        if self.min_score >= self.max_score:
            raise ConfigurationError("min_score must be less than max_score")
        
        validate_parameter("odds_at_min", self.odds_at_min, min_value=0.01, max_value=1000.0)
        validate_parameter("odds_at_max", self.odds_at_max, min_value=0.001, max_value=1.0)
        validate_parameter("pdo", self.pdo, min_value=5, max_value=100)
        
        validate_parameter("min_probability", self.min_probability, min_value=1e-10, max_value=0.01)
        validate_parameter("max_probability", self.max_probability, min_value=0.99, max_value=1.0)
        
        # Logical validation
        prob_at_min = self.odds_at_min / (1 + self.odds_at_min)
        prob_at_max = self.odds_at_max / (1 + self.odds_at_max)
        
        if prob_at_min <= prob_at_max:
            raise ConfigurationError(
                "odds_at_min should correspond to lower probability than odds_at_max"
            )
        
        logger.debug(f"ScorecardConfig validated: {self.min_score}-{self.max_score}, PDO={self.pdo}")


class ScorecardConverter(BaseConverter):
    """Enhanced scorecard converter with comprehensive features.
    
    Converts predicted probabilities to business-friendly scorecard scores
    (similar to FICO credit scores) with proper statistical calibration.
    
    Example:
        >>> converter = ScorecardConverter()
        >>> scores = converter.fit_transform(probabilities, sample_weight=weights)
        >>> interpretation = converter.interpret_score(scores[0])
        >>> print(f"Risk Score: {scores[0]:.0f}/850 ({interpretation['risk_category']})")
    """
    
    def __init__(
        self,
        config: Optional[ScorecardConfig] = None,
        scorecard_name: str = "Risk Score",
        **kwargs: Any
    ) -> None:
        """Initialize enhanced scorecard converter.
        
        Args:
            config: Scorecard configuration
            scorecard_name: Name for the scorecard (for reporting)
            **kwargs: Additional converter parameters
        """
        super().__init__(**kwargs)
        
        self.config = config or ScorecardConfig()
        self.scorecard_name = scorecard_name
        
        # Fitted parameters
        self.offset_: Optional[float] = None
        self.factor_: Optional[float] = None
        
        # Calibration statistics
        self.calibration_stats_: Optional[Dict[str, Any]] = None
        self.score_distribution_: Optional[Dict[str, float]] = None
        
        logger.info(f"Initialized ScorecardConverter '{scorecard_name}':")
        logger.info(f"  Range: {self.config.min_score}-{self.config.max_score}, PDO: {self.config.pdo}")

    @timer(name="scorecard_fitting")
    def fit(
        self,
        probabilities: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> 'ScorecardConverter':
        """Fit scorecard parameters to probability distribution.
        
        Args:
            probabilities: Predicted probabilities [0, 1]
            sample_weight: Optional sample weights for calibration
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            ExplainabilityError: If fitting fails
        """
        logger.info(f"📊 Fitting scorecard converter on {len(probabilities)} probabilities")
        
        try:
            with timed_operation("scorecard_calibration"):
                # Validate inputs
                self._validate_probabilities(probabilities)
                
                if sample_weight is not None:
                    if len(sample_weight) != len(probabilities):
                        raise DataValidationError("sample_weight length must match probabilities")
                    
                    if np.any(sample_weight < 0):
                        raise DataValidationError("sample_weight cannot contain negative values")
                
                # Clip probabilities to valid range
                clipped_probs = np.clip(
                    probabilities, 
                    self.config.min_probability, 
                    self.config.max_probability
                )
                
                # Calculate scorecard transformation parameters
                self._calculate_scorecard_parameters()
                
                # Compute calibration statistics
                self.calibration_stats_ = self._compute_calibration_stats(
                    clipped_probs, sample_weight
                )
                
                # Compute score distribution
                test_scores = self.transform(clipped_probs)
                self.score_distribution_ = self._compute_score_distribution(
                    test_scores, sample_weight
                )
            
            self.is_fitted = True
            
            logger.info(f"✅ Scorecard fitting completed:")
            logger.info(f"   Parameters: offset={self.offset_:.2f}, factor={self.factor_:.2f}")
            logger.info(f"   Score distribution: mean={self.score_distribution_['mean']:.1f}, "
                       f"std={self.score_distribution_['std']:.1f}")
            
            return self
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                "Scorecard fitting failed",
                error_code="SCORECARD_FITTING_FAILED",
                context=create_error_context(
                    n_probabilities=len(probabilities),
                    has_weights=sample_weight is not None,
                    config=self.config.__dict__
                )
            )

    def _validate_probabilities(self, probabilities: np.ndarray) -> None:
        """Validate probability array."""
        
        if len(probabilities) == 0:
            raise DataValidationError("Probabilities array cannot be empty")
        
        if not np.issubdtype(probabilities.dtype, np.floating):
            try:
                probabilities = probabilities.astype(float)
            except (ValueError, TypeError):
                raise DataValidationError("Probabilities must be numeric")
        
        # Check for invalid values
        if np.any(np.isnan(probabilities)):
            raise DataValidationError("Probabilities contain NaN values")
        
        if np.any(np.isinf(probabilities)):
            raise DataValidationError("Probabilities contain infinite values")
        
        # Check range (with some tolerance for floating point)
        if np.any(probabilities < -1e-10) or np.any(probabilities > 1.0001):
            invalid_count = np.sum((probabilities < 0) | (probabilities > 1))
            raise DataValidationError(
                f"Probabilities must be in [0, 1] range. Found {invalid_count} invalid values."
            )
        
        logger.debug(f"Probability validation passed: min={probabilities.min():.6f}, max={probabilities.max():.6f}")

    def _calculate_scorecard_parameters(self) -> None:
        """Calculate scorecard transformation parameters using odds-based approach."""
        
        # Calculate log odds at calibration points
        log_odds_min = np.log(self.config.odds_at_min)
        log_odds_max = np.log(self.config.odds_at_max)
        
        # Linear transformation: Score = offset + factor * log(odds)
        # At min_score: log(odds) = log_odds_min
        # At max_score: log(odds) = log_odds_max
        
        self.factor_ = (self.config.max_score - self.config.min_score) / (log_odds_max - log_odds_min)
        self.offset_ = self.config.min_score - self.factor_ * log_odds_min
        
        logger.debug(f"Scorecard parameters: offset={self.offset_:.4f}, factor={self.factor_:.4f}")

    def _compute_calibration_stats(
        self,
        probabilities: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute calibration statistics."""
        
        if sample_weight is not None:
            weights = sample_weight / sample_weight.sum()
            weighted_mean = np.average(probabilities, weights=weights)
            weighted_std = np.sqrt(np.average((probabilities - weighted_mean)**2, weights=weights))
        else:
            weighted_mean = np.mean(probabilities)
            weighted_std = np.std(probabilities)
        
        return {
            'mean_probability': float(weighted_mean),
            'std_probability': float(weighted_std),
            'min_probability': float(probabilities.min()),
            'max_probability': float(probabilities.max()),
            'median_probability': float(np.median(probabilities)),
            'q25_probability': float(np.percentile(probabilities, 25)),
            'q75_probability': float(np.percentile(probabilities, 75)),
            'n_samples': len(probabilities)
        }

    def _compute_score_distribution(
        self,
        scores: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute score distribution statistics."""
        
        if sample_weight is not None:
            weights = sample_weight / sample_weight.sum()
            weighted_mean = np.average(scores, weights=weights)
            weighted_std = np.sqrt(np.average((scores - weighted_mean)**2, weights=weights))
        else:
            weighted_mean = np.mean(scores)
            weighted_std = np.std(scores)
        
        return {
            'mean': float(weighted_mean),
            'std': float(weighted_std),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75))
        }

    def transform(
        self,
        probabilities: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """Transform probabilities to scorecard scores.
        
        Args:
            probabilities: Predicted probabilities [0, 1]
            **kwargs: Additional transformation parameters
            
        Returns:
            Scorecard scores array
            
        Raises:
            ExplainabilityError: If transformation fails
        """
        if not self.is_fitted:
            raise ExplainabilityError(
                "Scorecard converter must be fitted before transform",
                error_code="CONVERTER_NOT_FITTED"
            )
        
        try:
            # Validate and clip probabilities
            self._validate_probabilities(probabilities)
            clipped_probs = np.clip(
                probabilities,
                self.config.min_probability,
                self.config.max_probability
            )
            
            # Convert probabilities to odds
            odds = clipped_probs / (1 - clipped_probs)
            
            # Calculate scores using fitted parameters
            scores = self.offset_ + self.factor_ * np.log(odds)
            
            # Clip to valid score range
            scores = np.clip(scores, self.config.min_score, self.config.max_score)
            
            logger.debug(f"Transformed {len(probabilities)} probabilities to scores")
            
            return scores
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                "Scorecard transformation failed",
                error_code="SCORECARD_TRANSFORM_FAILED",
                context={"n_probabilities": len(probabilities)}
            )

    def interpret_score(self, score: float) -> Dict[str, Any]:
        """Interpret a scorecard score in business terms.
        
        Args:
            score: Scorecard score to interpret
            
        Returns:
            Dictionary with business interpretation
            
        Raises:
            ExplainabilityError: If interpretation fails
        """
        if not self.is_fitted:
            raise ExplainabilityError("Scorecard converter must be fitted before interpretation")
        
        try:
            # Convert back to probability
            log_odds = (score - self.offset_) / self.factor_
            odds = np.exp(log_odds)
            probability = odds / (1 + odds)
            
            # Determine risk category based on score
            risk_info = self._categorize_risk(score)
            
            # Calculate percentile based on fitted distribution
            percentile = self._score_to_percentile(score)
            
            return {
                'score': float(score),
                'probability': float(probability),
                'odds': float(odds),
                'risk_category': risk_info['category'],
                'risk_description': risk_info['description'],
                'color': risk_info['color'],
                'percentile': percentile,
                'scorecard_name': self.scorecard_name
            }
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                f"Score interpretation failed for score {score}",
                error_code="SCORE_INTERPRETATION_FAILED"
            )

    def _categorize_risk(self, score: float) -> Dict[str, str]:
        """Categorize risk based on score ranges."""
        
        score_range = self.config.max_score - self.config.min_score
        
        # Define risk categories based on score percentiles
        if score >= self.config.min_score + 0.8 * score_range:  # Top 20%
            return {
                'category': 'Very Low Risk',
                'description': 'Excellent risk profile',
                'color': 'green'
            }
        elif score >= self.config.min_score + 0.6 * score_range:  # 60-80%
            return {
                'category': 'Low Risk',
                'description': 'Good risk profile',
                'color': 'lightgreen'
            }
        elif score >= self.config.min_score + 0.4 * score_range:  # 40-60%
            return {
                'category': 'Medium Risk',
                'description': 'Average risk profile',
                'color': 'yellow'
            }
        elif score >= self.config.min_score + 0.2 * score_range:  # 20-40%
            return {
                'category': 'High Risk',
                'description': 'Elevated risk profile',
                'color': 'orange'
            }
        else:  # Bottom 20%
            return {
                'category': 'Very High Risk',
                'description': 'Poor risk profile',
                'color': 'red'
            }

    def _score_to_percentile(self, score: float) -> float:
        """Convert score to percentile based on fitted distribution."""
        
        if not self.score_distribution_:
            # Fallback: linear percentile based on score range
            normalized = (score - self.config.min_score) / (self.config.max_score - self.config.min_score)
            return max(0.0, min(100.0, normalized * 100))
        
        # Use fitted distribution for more accurate percentile
        mean = self.score_distribution_['mean']
        std = self.score_distribution_['std']
        
        if std > 0:
            # Approximate percentile using normal distribution
            z_score = (score - mean) / std
            # Clip to reasonable range to avoid extreme values
            z_score = max(-3, min(3, z_score))
            
            # Convert z-score to percentile (approximate)
            from scipy.stats import norm
            try:
                percentile = norm.cdf(z_score) * 100
                return float(percentile)
            except ImportError:
                # Fallback without scipy
                # Rough approximation: z=-2 -> 2%, z=0 -> 50%, z=2 -> 98%
                if z_score <= -2:
                    return 2.0
                elif z_score >= 2:
                    return 98.0
                else:
                    return 50.0 + 24 * z_score  # Linear approximation
        
        return 50.0  # Default to median

    def interpret(self, transformed_data: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Interpret transformed scorecard data (batch interpretation).
        
        Args:
            transformed_data: Array of scorecard scores
            **kwargs: Additional interpretation parameters
            
        Returns:
            Dictionary with batch interpretation results
        """
        if not self.is_fitted:
            raise ExplainabilityError("Scorecard converter must be fitted before interpretation")
        
        # Compute aggregate statistics
        interpretations = [self.interpret_score(score) for score in transformed_data]
        
        # Aggregate results
        risk_categories = [interp['risk_category'] for interp in interpretations]
        probabilities = [interp['probability'] for interp in interpretations]
        
        risk_distribution = pd.Series(risk_categories).value_counts().to_dict()
        
        return {
            'n_scores': len(transformed_data),
            'score_stats': {
                'mean': float(np.mean(transformed_data)),
                'std': float(np.std(transformed_data)),
                'min': float(np.min(transformed_data)),
                'max': float(np.max(transformed_data)),
                'median': float(np.median(transformed_data))
            },
            'probability_stats': {
                'mean': float(np.mean(probabilities)),
                'std': float(np.std(probabilities)),
                'min': float(np.min(probabilities)),
                'max': float(np.max(probabilities))
            },
            'risk_distribution': risk_distribution,
            'scorecard_name': self.scorecard_name
        }

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics from fitting.
        
        Returns:
            Dictionary with calibration statistics
        """
        if not self.is_fitted:
            raise ExplainabilityError("Scorecard converter must be fitted first")
        
        return {
            'config': self.config.__dict__,
            'parameters': {
                'offset': self.offset_,
                'factor': self.factor_
            },
            'calibration_stats': self.calibration_stats_,
            'score_distribution': self.score_distribution_
        }

    def plot_score_distribution(
        self,
        scores: Optional[np.ndarray] = None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Any:
        """Plot scorecard score distribution.
        
        Args:
            scores: Optional scores to plot (uses fitted distribution if None)
            save_path: Optional path to save plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Plot figure
        """
        if not self.is_fitted:
            raise ExplainabilityError("Scorecard converter must be fitted first")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            if scores is not None:
                # Plot provided scores
                ax1.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_title(f'{self.scorecard_name} Distribution')
                ax1.set_xlabel('Score')
                ax1.set_ylabel('Frequency')
                ax1.grid(True, alpha=0.3)
                
                # Risk category distribution
                interpretations = [self.interpret_score(score) for score in scores[:1000]]  # Sample for performance
                risk_cats = [interp['risk_category'] for interp in interpretations]
                risk_counts = pd.Series(risk_cats).value_counts()
                
                colors = {'Very Low Risk': 'green', 'Low Risk': 'lightgreen', 
                         'Medium Risk': 'yellow', 'High Risk': 'orange', 'Very High Risk': 'red'}
                
                bars = ax2.bar(risk_counts.index, risk_counts.values, 
                              color=[colors.get(cat, 'gray') for cat in risk_counts.index])
                ax2.set_title('Risk Category Distribution')
                ax2.set_ylabel('Count')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add percentage labels on bars
                total = len(scores)
                for bar, count in zip(bars, risk_counts.values):
                    percentage = count / total * 100
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                            f'{percentage:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Score distribution plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            plt.close()
            handle_and_reraise(
                e, ExplainabilityError,
                "Failed to create score distribution plot",
                error_code="PLOT_CREATION_FAILED"
            )


# Convenience functions
def convert_to_scorecard(
    probabilities: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[ScorecardConfig] = None,
    **kwargs: Any
) -> Tuple[np.ndarray, ScorecardConverter]:
    """Convert probabilities to scorecard scores with default settings.
    
    Args:
        probabilities: Predicted probabilities
        sample_weight: Optional sample weights
        config: Optional scorecard configuration
        **kwargs: Additional converter parameters
        
    Returns:
        Tuple of (scores, fitted_converter)
        
    Example:
        >>> scores, converter = convert_to_scorecard(y_pred_proba)
        >>> interpretation = converter.interpret_score(scores[0])
        >>> print(f"Risk Score: {scores[0]:.0f}/850")
        >>> print(f"Risk Category: {interpretation['risk_category']}")
    """
    logger.info(f"🎯 Converting {len(probabilities)} probabilities to scorecard scores")
    
    converter = ScorecardConverter(config=config, **kwargs)
    scores = converter.fit_transform(probabilities, sample_weight=sample_weight)
    
    logger.info(f"✅ Scorecard conversion completed:")
    logger.info(f"   Score range: {scores.min():.0f} - {scores.max():.0f}")
    logger.info(f"   Mean score: {scores.mean():.0f}")
    
    return scores, converter


def create_scorecard_report(
    probabilities: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    output_dir: Union[str, Path] = "scorecard_analysis",
    **kwargs: Any
) -> Dict[str, Any]:
    """Create comprehensive scorecard analysis report.
    
    Args:
        probabilities: Predicted probabilities
        sample_weight: Optional sample weights
        output_dir: Directory to save analysis outputs
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with scorecard analysis results
    """
    output_path = Path(output_dir)
    ensure_dir(output_path)
    
    logger.info(f"📊 Creating scorecard analysis report in {output_path}")
    
    # Convert to scorecard
    scores, converter = convert_to_scorecard(
        probabilities, sample_weight=sample_weight, **kwargs
    )
    
    # Generate interpretations
    batch_interpretation = converter.interpret(scores)
    
    # Create visualizations
    fig = converter.plot_score_distribution(
        scores, save_path=output_path / "score_distribution.png"
    )
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'probability': probabilities,
        'score': scores,
        'risk_category': [converter.interpret_score(s)['risk_category'] for s in scores]
    })
    results_df.to_csv(output_path / "scorecard_results.csv", index=False)
    
    # Save calibration stats
    calibration_stats = converter.get_calibration_stats()
    
    import json
    with open(output_path / "calibration_stats.json", 'w') as f:
        json.dump(calibration_stats, f, indent=2, default=str)
    
    # Create summary report
    report = {
        'scorecard_summary': batch_interpretation,
        'calibration_stats': calibration_stats,
        'files_created': [
            str(output_path / "score_distribution.png"),
            str(output_path / "scorecard_results.csv"),
            str(output_path / "calibration_stats.json")
        ],
        'converter': converter
    }
    
    logger.info(f"✅ Scorecard analysis report completed in {output_path}")
    
    return report


# Export key classes and functions
__all__ = [
    'ScorecardConfig',
    'ScorecardConverter',
    'convert_to_scorecard',
    'create_scorecard_report'
]