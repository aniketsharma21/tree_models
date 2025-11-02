# tree_models/explainability/reason_codes.py  
"""Enhanced reason code generation for regulatory compliance and transparency.

This module provides production-ready reason code generation with:
- Type-safe interfaces and comprehensive validation
- SHAP-based and model-agnostic reason extraction  
- Regulatory compliance features (FCRA, GDPR, etc.)
- Customizable text templates and formatting
- Multiple output formats (JSON, text, HTML)
- Comprehensive error handling and logging
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import json
import warnings
from dataclasses import dataclass
from enum import Enum

from .base import BaseReasonCodeGenerator, ensure_dir
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


class ReasonCodeType(Enum):
    """Types of reason codes for different compliance needs."""
    ADVERSE = "adverse"  # Adverse action reasons (FCRA)
    POSITIVE = "positive"  # Positive factors
    NEUTRAL = "neutral"  # Neutral explanations
    IMPROVEMENT = "improvement"  # Improvement suggestions


@dataclass
class ReasonCodeConfig:
    """Configuration for reason code generation with regulatory compliance."""
    
    # Basic settings
    max_reasons: int = 5
    min_impact: float = 0.01
    reason_types: List[ReasonCodeType] = None
    
    # Text formatting
    use_percentiles: bool = True
    use_relative_values: bool = True
    include_improvement_suggestions: bool = True
    
    # Regulatory compliance
    fcra_compliant: bool = False  # FCRA (Fair Credit Reporting Act) compliance
    gdpr_compliant: bool = False  # GDPR right to explanation
    
    # Language settings
    language: str = "en"
    formality_level: str = "professional"  # "casual", "professional", "formal"
    
    def __post_init__(self) -> None:
        """Validate reason code configuration."""
        validate_parameter("max_reasons", self.max_reasons, min_value=1, max_value=20)
        validate_parameter("min_impact", self.min_impact, min_value=0.0, max_value=1.0)
        validate_parameter("language", self.language, valid_values=["en", "es", "fr"])
        validate_parameter("formality_level", self.formality_level, 
                         valid_values=["casual", "professional", "formal"])
        
        if self.reason_types is None:
            self.reason_types = [ReasonCodeType.ADVERSE, ReasonCodeType.POSITIVE]
        
        logger.debug(f"ReasonCodeConfig validated: max_reasons={self.max_reasons}, fcra={self.fcra_compliant}")


class ReasonCodeGenerator(BaseReasonCodeGenerator):
    """Enhanced reason code generator with regulatory compliance.
    
    Generates human-readable explanations for model predictions with
    support for regulatory compliance and multiple output formats.
    
    Example:
        >>> generator = ReasonCodeGenerator(max_reasons=5, fcra_compliant=True)
        >>> reasons = generator.generate_reasons(shap_values, X_test, feature_names)
        >>> for reason in reasons[0]['reasons']:
        ...     print(f"- {reason['text']}")
    """
    
    def __init__(
        self,
        config: Optional[ReasonCodeConfig] = None,
        custom_templates: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize enhanced reason code generator.
        
        Args:
            config: Reason code configuration
            custom_templates: Custom text templates for formatting
            **kwargs: Additional generator parameters
        """
        self.config = config or ReasonCodeConfig()
        
        super().__init__(
            max_reasons=self.config.max_reasons,
            min_impact=self.config.min_impact,
            **kwargs
        )
        
        # Text templates
        self.templates = self._initialize_templates(custom_templates)
        
        # Feature value statistics (for percentile calculations)
        self.feature_stats_: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized ReasonCodeGenerator:")
        logger.info(f"  Max reasons: {self.config.max_reasons}, FCRA: {self.config.fcra_compliant}")
        logger.info(f"  Language: {self.config.language}, Level: {self.config.formality_level}")

    def _initialize_templates(self, custom_templates: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, str]]:
        """Initialize text templates for different languages and formality levels."""
        
        templates = {
            "en": {
                "adverse_high": "{feature_name} ({value}) is {comparison} than {percentile}% of applicants",
                "adverse_low": "{feature_name} ({value}) is {comparison} than {percentile}% of applicants", 
                "positive_high": "{feature_name} ({value}) favorably compares to other applicants",
                "positive_low": "{feature_name} ({value}) is in the favorable range",
                "improvement": "Improving {feature_name} could positively impact your score",
                "neutral": "{feature_name} has a moderate impact on the decision"
            }
        }
        
        # Add custom templates if provided
        if custom_templates:
            for lang, lang_templates in custom_templates.items():
                if lang not in templates:
                    templates[lang] = {}
                templates[lang].update(lang_templates)
        
        return templates

    @timer(name="reason_code_generation")
    def generate_reasons(
        self,
        explanation_data: np.ndarray,
        instance_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        instance_indices: Optional[List[int]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Generate reason codes for predictions with comprehensive features.
        
        Args:
            explanation_data: SHAP values or other explanation values
            instance_data: Original feature data
            feature_names: Names of features
            instance_indices: Specific instances to explain (all if None)
            **kwargs: Additional generation parameters
            
        Returns:
            List of reason code dictionaries with detailed explanations
            
        Raises:
            ExplainabilityError: If reason generation fails
        """
        logger.info(f"🔍 Generating reason codes for explanations")
        
        try:
            with timed_operation("reason_code_processing"):
                # Validate inputs
                self._validate_inputs(explanation_data, instance_data, feature_names)
                
                if feature_names is None:
                    feature_names = list(instance_data.columns)
                
                if instance_indices is None:
                    instance_indices = list(range(len(explanation_data)))
                
                # Compute feature statistics for percentile calculations
                if self.config.use_percentiles and self.feature_stats_ is None:
                    self._compute_feature_statistics(instance_data, feature_names)
                
                # Generate reasons for each instance
                reason_codes = []
                
                for idx in instance_indices:
                    if idx >= len(explanation_data):
                        logger.warning(f"Instance index {idx} out of range, skipping")
                        continue
                    
                    instance_reasons = self._generate_instance_reasons(
                        explanation_data[idx],
                        instance_data.iloc[idx] if idx < len(instance_data) else None,
                        feature_names,
                        idx
                    )
                    
                    reason_codes.append(instance_reasons)
                
            logger.info(f"✅ Generated reason codes for {len(reason_codes)} instances")
            logger.info(f"   Average reasons per instance: {np.mean([len(rc['reasons']) for rc in reason_codes]):.1f}")
            
            return reason_codes
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                "Reason code generation failed",
                error_code="REASON_GENERATION_FAILED",
                context=create_error_context(
                    n_instances=len(explanation_data) if explanation_data is not None else 0,
                    n_features=len(feature_names) if feature_names else 0
                )
            )

    def _validate_inputs(
        self,
        explanation_data: np.ndarray,
        instance_data: pd.DataFrame,
        feature_names: Optional[List[str]]
    ) -> None:
        """Validate inputs for reason generation."""
        
        if explanation_data is None:
            raise DataValidationError("Explanation data cannot be None")
        
        if instance_data is None or instance_data.empty:
            raise DataValidationError("Instance data cannot be empty")
        
        if len(explanation_data) == 0:
            raise DataValidationError("Explanation data cannot be empty")
        
        if explanation_data.ndim != 2:
            raise DataValidationError(f"Explanation data must be 2D, got shape {explanation_data.shape}")
        
        if feature_names and len(feature_names) != explanation_data.shape[1]:
            raise DataValidationError(
                f"Feature names length ({len(feature_names)}) must match explanation data features ({explanation_data.shape[1]})"
            )

    def _compute_feature_statistics(self, instance_data: pd.DataFrame, feature_names: List[str]) -> None:
        """Compute feature statistics for percentile-based explanations."""
        
        logger.debug("Computing feature statistics for percentile calculations")
        
        stats_data = []
        
        for feature in feature_names:
            if feature in instance_data.columns:
                feature_values = instance_data[feature].dropna()
                
                if len(feature_values) > 0 and pd.api.types.is_numeric_dtype(feature_values):
                    stats_data.append({
                        'feature': feature,
                        'mean': feature_values.mean(),
                        'median': feature_values.median(),
                        'std': feature_values.std(),
                        'min': feature_values.min(),
                        'max': feature_values.max(),
                        'q25': feature_values.quantile(0.25),
                        'q75': feature_values.quantile(0.75),
                        'is_numeric': True
                    })
                else:
                    # For categorical features
                    stats_data.append({
                        'feature': feature,
                        'mode': feature_values.mode()[0] if len(feature_values.mode()) > 0 else None,
                        'unique_count': feature_values.nunique(),
                        'is_numeric': False
                    })
        
        self.feature_stats_ = pd.DataFrame(stats_data)
        logger.debug(f"Computed statistics for {len(self.feature_stats_)} features")

    def _generate_instance_reasons(
        self,
        instance_explanation: np.ndarray,
        instance_features: Optional[pd.Series],
        feature_names: List[str],
        instance_idx: int
    ) -> Dict[str, Any]:
        """Generate reason codes for a single instance."""
        
        # Calculate feature contributions
        contributions = []
        
        for i, (feature_name, explanation_value) in enumerate(zip(feature_names, instance_explanation)):
            if abs(explanation_value) >= self.min_impact:
                
                feature_value = instance_features[feature_name] if instance_features is not None else None
                
                contribution = {
                    'feature': feature_name,
                    'explanation_value': float(explanation_value),
                    'feature_value': feature_value,
                    'impact_direction': 'increases' if explanation_value > 0 else 'decreases',
                    'abs_impact': float(abs(explanation_value)),
                    'rank': None  # Will be set after sorting
                }
                
                # Add percentile information if available
                if self.config.use_percentiles and self.feature_stats_ is not None:
                    percentile_info = self._calculate_percentile_info(feature_name, feature_value)
                    if percentile_info:
                        contribution.update(percentile_info)
                
                contributions.append(contribution)
        
        # Sort by absolute impact and assign ranks
        contributions.sort(key=lambda x: x['abs_impact'], reverse=True)
        for rank, contrib in enumerate(contributions, 1):
            contrib['rank'] = rank
        
        # Take top reasons
        top_contributions = contributions[:self.max_reasons]
        
        # Generate human-readable reasons
        formatted_reasons = []
        for contrib in top_contributions:
            reason_text = self.format_reason(
                contrib['feature'],
                contrib['explanation_value'], 
                contrib['feature_value'],
                percentile_info=contrib.get('percentile_info')
            )
            
            formatted_reasons.append({
                'text': reason_text,
                'feature': contrib['feature'],
                'impact': contrib['explanation_value'],
                'value': contrib['feature_value'],
                'rank': contrib['rank'],
                'reason_type': self._determine_reason_type(contrib),
                'confidence': self._calculate_confidence(contrib)
            })
        
        # Add improvement suggestions if requested
        improvement_suggestions = []
        if self.config.include_improvement_suggestions:
            improvement_suggestions = self._generate_improvement_suggestions(top_contributions)
        
        return {
            'instance_index': instance_idx,
            'total_impact': float(np.sum(instance_explanation)),
            'reasons': formatted_reasons,
            'all_contributions': contributions,
            'improvement_suggestions': improvement_suggestions,
            'compliance': {
                'fcra_compliant': self.config.fcra_compliant,
                'gdpr_compliant': self.config.gdpr_compliant,
                'generated_timestamp': pd.Timestamp.now().isoformat()
            }
        }

    def _calculate_percentile_info(self, feature_name: str, feature_value: Any) -> Optional[Dict[str, Any]]:
        """Calculate percentile information for a feature value."""
        
        if self.feature_stats_ is None or feature_value is None:
            return None
        
        feature_stats = self.feature_stats_[self.feature_stats_['feature'] == feature_name]
        
        if len(feature_stats) == 0:
            return None
        
        stats = feature_stats.iloc[0]
        
        if stats['is_numeric'] and pd.notna(feature_value):
            try:
                z_score = 0.0
                # Calculate approximate percentile
                if stats['std'] > 0:
                    z_score = (feature_value - stats['mean']) / stats['std']
                    # Approximate percentile from z-score
                    if z_score <= -2:
                        percentile = 2
                    elif z_score >= 2:
                        percentile = 98
                    else:
                        # Linear approximation for middle range
                        percentile = 50 + 24 * z_score
                    
                    percentile = max(1, min(99, percentile))
                else:
                    percentile = 50  # Default for zero variance
                
                return {
                    'percentile_info': {
                        'percentile': float(percentile),
                        'comparison_to_mean': 'above' if feature_value > stats['mean'] else 'below',
                        'comparison_to_median': 'above' if feature_value > stats['median'] else 'below',
                        'is_outlier': abs(z_score) > 2,
                        'z_score': float(z_score)
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to calculate percentile for {feature_name}: {e}")
                return None
        
        return None

    def format_reason(
        self,
        feature: str,
        impact: float,
        value: Any,
        percentile_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> str:
        """Format a single reason into human-readable text with regulatory compliance.
        
        Args:
            feature: Feature name
            impact: Impact value (e.g., SHAP value)
            value: Feature value
            percentile_info: Optional percentile information
            **kwargs: Additional formatting parameters
            
        Returns:
            Human-readable reason text
        """
        try:
            # Clean feature name for display
            clean_feature = self._clean_feature_name(feature)
            
            # Format value for display
            formatted_value = self._format_feature_value(value)
            
            # Determine reason type
            is_adverse = impact < 0
            
            # Get appropriate template
            templates = self.templates.get(self.config.language, self.templates["en"])
            
            if percentile_info and self.config.use_percentiles:
                # Use percentile-based explanation
                pct_info = percentile_info.get('percentile_info', {})
                percentile = pct_info.get('percentile', 50)
                
                if is_adverse:
                    if impact < -0.05:  # Significant negative impact
                        template_key = "adverse_high"
                        comparison = "higher" if percentile > 50 else "lower"
                    else:
                        template_key = "adverse_low"
                        comparison = "higher" if percentile > 50 else "lower"
                else:
                    template_key = "positive_high" if percentile > 70 else "positive_low"
                    comparison = "favorably"
                
                template = templates.get(template_key, templates.get("neutral", "{feature_name} affects the decision"))
                
                # Format with percentile information
                if self.config.fcra_compliant:
                    # FCRA-compliant language
                    if is_adverse:
                        reason_text = f"{clean_feature} is {comparison} relative to other applicants in our database"
                    else:
                        reason_text = f"{clean_feature} compares favorably to other applicants"
                else:
                    # Standard formatting
                    reason_text = template.format(
                        feature_name=clean_feature,
                        value=formatted_value,
                        comparison=comparison,
                        percentile=int(percentile) if percentile < 50 else int(100 - percentile)
                    )
            else:
                # Simple impact-based explanation
                if is_adverse:
                    direction = "increases risk" if abs(impact) > 0.05 else "slightly increases risk"
                else:
                    direction = "decreases risk" if impact > 0.05 else "slightly decreases risk"
                
                if self.config.fcra_compliant:
                    reason_text = f"{clean_feature} ({formatted_value}) is a factor in this decision"
                else:
                    reason_text = f"{clean_feature} ({formatted_value}) {direction}"
            
            # Apply formality adjustments
            if self.config.formality_level == "formal":
                reason_text = self._apply_formal_language(reason_text)
            elif self.config.formality_level == "casual":
                reason_text = self._apply_casual_language(reason_text)
            
            return reason_text
            
        except Exception as e:
            logger.warning(f"Failed to format reason for {feature}: {e}")
            # Fallback reason
            return f"{self._clean_feature_name(feature)} affects this decision"

    def _clean_feature_name(self, feature_name: str) -> str:
        """Clean feature name for human-readable display."""
        
        # Replace underscores with spaces
        clean_name = feature_name.replace('_', ' ').replace('-', ' ')
        
        # Convert to title case
        clean_name = clean_name.title()
        
        # Handle common abbreviations
        abbreviation_map = {
            'Id': 'ID',
            'Ssn': 'SSN', 
            'Dob': 'Date of Birth',
            'Amt': 'Amount',
            'Num': 'Number',
            'Pct': 'Percentage',
            'Avg': 'Average',
            'Min': 'Minimum',
            'Max': 'Maximum'
        }
        
        for abbrev, full_form in abbreviation_map.items():
            clean_name = clean_name.replace(abbrev, full_form)
        
        return clean_name

    def _format_feature_value(self, value: Any) -> str:
        """Format feature value for display."""
        
        if value is None or pd.isna(value):
            return "N/A"
        
        if isinstance(value, (int, float)):
            if abs(value) < 0.01:
                return f"{value:.4f}"
            elif abs(value) < 1:
                return f"{value:.2f}"
            elif abs(value) < 1000:
                return f"{value:.1f}"
            else:
                return f"{value:,.0f}"
        
        return str(value)

    def _determine_reason_type(self, contribution: Dict[str, Any]) -> ReasonCodeType:
        """Determine the type of reason code based on contribution."""
        
        impact = contribution['explanation_value']
        
        if impact < -0.05:
            return ReasonCodeType.ADVERSE
        elif impact > 0.05:
            return ReasonCodeType.POSITIVE
        else:
            return ReasonCodeType.NEUTRAL

    def _calculate_confidence(self, contribution: Dict[str, Any]) -> float:
        """Calculate confidence score for a reason."""
        
        # Simple confidence based on impact magnitude
        abs_impact = contribution['abs_impact']
        
        if abs_impact > 0.1:
            return 0.9
        elif abs_impact > 0.05:
            return 0.7
        elif abs_impact > 0.02:
            return 0.5
        else:
            return 0.3

    def _generate_improvement_suggestions(self, contributions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate improvement suggestions based on adverse factors."""
        
        suggestions = []
        
        for contrib in contributions[:3]:  # Top 3 adverse factors
            if contrib['explanation_value'] < 0:  # Adverse impact
                feature_name = self._clean_feature_name(contrib['feature'])
                
                suggestion_text = f"Consider improving {feature_name} to potentially increase your score"
                
                suggestions.append({
                    'feature': contrib['feature'],
                    'suggestion': suggestion_text,
                    'priority': 'high' if contrib['abs_impact'] > 0.1 else 'medium'
                })
        
        return suggestions

    def _apply_formal_language(self, text: str) -> str:
        """Apply formal language transformations."""
        
        # Replace casual terms with formal ones
        formal_replacements = {
            "your": "the applicant's",
            "you": "the applicant",
            "affects": "influences",
            "increases": "elevates",
            "decreases": "reduces"
        }
        
        for casual, formal in formal_replacements.items():
            text = text.replace(casual, formal)
        
        return text

    def _apply_casual_language(self, text: str) -> str:
        """Apply casual language transformations."""
        
        # Replace formal terms with casual ones
        casual_replacements = {
            "the applicant": "you",
            "influences": "affects",
            "elevates": "increases",
            "reduces": "decreases"
        }
        
        for formal, casual in casual_replacements.items():
            text = text.replace(formal, casual)
        
        return text

    def export_reasons(
        self,
        reason_codes: List[Dict[str, Any]],
        output_format: str = "json",
        save_path: Optional[Union[str, Path]] = None
    ) -> Union[str, Dict[str, Any]]:
        """Export reason codes in specified format.
        
        Args:
            reason_codes: Generated reason codes
            output_format: Output format ('json', 'text', 'html', 'csv')
            save_path: Optional path to save output
            
        Returns:
            Formatted output string or dictionary
        """
        try:
            if output_format == "json":
                output = self._export_json(reason_codes)
            elif output_format == "text":
                output = self._export_text(reason_codes)
            elif output_format == "html":
                output = self._export_html(reason_codes)
            elif output_format == "csv":
                output = self._export_csv(reason_codes)
            else:
                raise ConfigurationError(f"Unknown output format: {output_format}")
            
            # Save to file if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                if output_format == "csv":
                    output.to_csv(save_path, index=False)
                else:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        if isinstance(output, dict):
                            json.dump(output, f, indent=2, ensure_ascii=False)
                        else:
                            f.write(output)
                
                logger.info(f"Reason codes exported to {save_path}")
            
            return output
            
        except Exception as e:
            handle_and_reraise(
                e, ExplainabilityError,
                f"Failed to export reason codes in {output_format} format",
                error_code="EXPORT_FAILED"
            )

    def _export_json(self, reason_codes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export reason codes as JSON."""
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        serializable_codes = []
        for code in reason_codes:
            serializable_code = {}
            for key, value in code.items():
                if key == 'reasons':
                    serializable_code[key] = [
                        {k: convert_types(v) for k, v in reason.items()}
                        for reason in value
                    ]
                else:
                    serializable_code[key] = convert_types(value)
            serializable_codes.append(serializable_code)
        
        return {
            'reason_codes': serializable_codes,
            'generation_config': {
                'max_reasons': self.config.max_reasons,
                'min_impact': self.config.min_impact,
                'fcra_compliant': self.config.fcra_compliant,
                'language': self.config.language
            },
            'metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'total_instances': len(reason_codes)
            }
        }

    def _export_text(self, reason_codes: List[Dict[str, Any]]) -> str:
        """Export reason codes as formatted text."""
        
        output_lines = []
        output_lines.append("EXPLANATION REPORT")
        output_lines.append("=" * 50)
        output_lines.append("")
        
        for i, code in enumerate(reason_codes):
            output_lines.append(f"Instance {i + 1} (Index: {code['instance_index']}):")
            output_lines.append("-" * 30)
            
            for j, reason in enumerate(code['reasons'], 1):
                output_lines.append(f"{j}. {reason['text']}")
            
            if code.get('improvement_suggestions'):
                output_lines.append("\nImprovement Suggestions:")
                for suggestion in code['improvement_suggestions']:
                    output_lines.append(f"• {suggestion['suggestion']}")
            
            output_lines.append("")
        
        return "\n".join(output_lines)

    def _export_html(self, reason_codes: List[Dict[str, Any]]) -> str:
        """Export reason codes as HTML."""
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explanation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .instance { border: 1px solid #ccc; padding: 15px; margin: 10px 0; }
                .reason { margin: 5px 0; }
                .adverse { color: #d32f2f; }
                .positive { color: #388e3c; }
                .neutral { color: #1976d2; }
            </style>
        </head>
        <body>
            <h1>Model Explanation Report</h1>
        """
        
        for code in reason_codes:
            html += f'<div class="instance">'
            html += f'<h3>Instance {code["instance_index"]}</h3>'
            
            for reason in code['reasons']:
                reason_class = reason.get('reason_type', ReasonCodeType.NEUTRAL).value.lower()
                html += f'<div class="reason {reason_class}">• {reason["text"]}</div>'
            
            if code.get('improvement_suggestions'):
                html += '<h4>Improvement Suggestions:</h4>'
                for suggestion in code['improvement_suggestions']:
                    html += f'<div class="suggestion">• {suggestion["suggestion"]}</div>'
            
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html

    def _export_csv(self, reason_codes: List[Dict[str, Any]]) -> pd.DataFrame:
        """Export reason codes as CSV DataFrame."""
        
        rows = []
        
        for code in reason_codes:
            instance_idx = code['instance_index']
            
            for rank, reason in enumerate(code['reasons'], 1):
                rows.append({
                    'instance_index': instance_idx,
                    'reason_rank': rank,
                    'feature': reason['feature'],
                    'reason_text': reason['text'],
                    'impact': reason['impact'],
                    'feature_value': reason['value'],
                    'reason_type': reason.get('reason_type', 'neutral'),
                    'confidence': reason.get('confidence', 0.0)
                })
        
        return pd.DataFrame(rows)


# Convenience functions
def generate_reason_codes(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_reasons: int = 5,
    fcra_compliant: bool = False,
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """Generate reason codes with default settings.
    
    Args:
        shap_values: SHAP values for explanations
        X: Original feature data
        feature_names: Feature names (uses X.columns if None)
        max_reasons: Maximum reasons per prediction
        fcra_compliant: Whether to use FCRA-compliant language
        **kwargs: Additional generator parameters
        
    Returns:
        List of reason code dictionaries
        
    Example:
        >>> reasons = generate_reason_codes(shap_values, X_test, max_reasons=3)
        >>> for reason in reasons[0]['reasons']:
        ...     print(f"- {reason['text']}")
    """
    config = ReasonCodeConfig(
        max_reasons=max_reasons,
        fcra_compliant=fcra_compliant,
        **kwargs
    )
    
    generator = ReasonCodeGenerator(config=config)
    
    if feature_names is None:
        feature_names = list(X.columns)
    
    return generator.generate_reasons(shap_values, X, feature_names)


# Export key classes and functions
__all__ = [
    'ReasonCodeType',
    'ReasonCodeConfig', 
    'ReasonCodeGenerator',
    'generate_reason_codes'
]