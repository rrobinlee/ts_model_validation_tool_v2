"""
Probabilistic forecasts
"""

import numpy as np
from typing import Dict, List, Optional

from ..core.validators import BaseValidator
from ..core.results import ValidationResult


class ProbabilisticValidator(BaseValidator):
    def validate(self, *args, **kwargs) -> List[ValidationResult]:
        """General validate method."""
        return []
    
    def validate_prediction_intervals(
        self,
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        confidence_level: float = 0.95
    ) -> List[ValidationResult]:
        """
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_lower : np.ndarray
            Lower bounds of prediction intervals
        y_upper : np.ndarray
            Upper bounds of prediction intervals
        confidence_level : float
            Target confidence level (e.g., 0.95 for 95% intervals)
        Returns:
        --------
        List[ValidationResult]
        """
        results = []
        
        # Coverage test
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        coverage_deviation = abs(coverage - confidence_level)
        
        results.append(ValidationResult(
            test_name="Prediction Interval Coverage",
            statistic=coverage,
            passed=coverage_deviation < 0.05,  # Within 5% of target
            metadata={
                'target_coverage': confidence_level,
                'actual_coverage': coverage,
                'deviation': coverage_deviation
            }
        ))
        
        # Interval width
        avg_width = np.mean(y_upper - y_lower)
        std_y = np.std(y_true)
        relative_width = avg_width / std_y if std_y > 0 else np.nan
        
        results.append(ValidationResult(
            test_name="Average Interval Width",
            statistic=avg_width,
            metadata={
                'relative_width': relative_width,
                'interpretation': 'Width relative to data std'
            }
        ))
        
        # Winkler score (lower is better)
        alpha = 1 - confidence_level
        width = y_upper - y_lower
        lower_miss = (2 / alpha) * (y_lower - y_true) * (y_true < y_lower)
        upper_miss = (2 / alpha) * (y_true - y_upper) * (y_true > y_upper)
        winkler_score = np.mean(width + lower_miss + upper_miss)
        
        results.append(ValidationResult(
            test_name="Winkler Score",
            statistic=winkler_score,
            metadata={
                'interpretation': 'Lower is better - penalizes wide intervals and misses'
            }
        ))
        
        # Interval sharpness (smaller is better, but not at expense of coverage)
        sharpness = np.mean(y_upper - y_lower)
        results.append(ValidationResult(
            test_name="Interval Sharpness",
            statistic=sharpness,
            metadata={
                'interpretation': 'Average interval width - smaller indicates sharper forecasts'
            }
        ))
        
        return results
    
    def validate_quantile_forecasts(
        self,
        y_true: np.ndarray,
        quantile_forecasts: Dict[float, np.ndarray]
    ) -> List[ValidationResult]:
        """
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        quantile_forecasts : Dict[float, np.ndarray]
            Dictionary mapping quantiles (0-1) to forecast arrays
        Returns:
        --------
        List[ValidationResult]
        """
        results = []
        
        for q, y_pred_q in sorted(quantile_forecasts.items()):
            # Pinball loss
            error = y_true - y_pred_q
            loss = np.where(error >= 0, q * error, (q - 1) * error)
            pinball_score = np.mean(loss)
            
            # Empirical coverage (should match theoretical quantile)
            coverage = np.mean(y_true <= y_pred_q)
            coverage_deviation = abs(coverage - q)
            
            results.append(ValidationResult(
                test_name=f"Quantile {q:.2f} Validation",
                statistic=pinball_score,
                p_value=coverage,
                passed=coverage_deviation < 0.05,
                metadata={
                    'target_quantile': q,
                    'empirical_coverage': coverage,
                    'coverage_deviation': coverage_deviation,
                    'pinball_loss': pinball_score,
                    'interpretation': 'Coverage should match target quantile'
                }
            ))
        
        return results
    
    def validate_probabilistic_calibration(
        self,
        y_true: np.ndarray,
        predicted_probs: np.ndarray,
        n_bins: int = 10
    ) -> List[ValidationResult]:
        """
        Parameters:
        -----------
        y_true : np.ndarray
            Binary actual outcomes (0 or 1)
        predicted_probs : np.ndarray
            Predicted probabilities
        n_bins : int
            Number of bins for calibration assessment
        Returns:
        --------
        List[ValidationResult]
        """
        results = []
        
        # Calibration by binning
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        
        calibration_errors = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_predicted = np.mean(predicted_probs[mask])
                avg_actual = np.mean(y_true[mask])
                calibration_errors.append(abs(avg_predicted - avg_actual))
        
        # Expected Calibration Error (ECE)
        ece = np.mean(calibration_errors) if calibration_errors else np.nan
        
        results.append(ValidationResult(
            test_name="Expected Calibration Error",
            statistic=ece,
            passed=ece < 0.1,  # Less than 10% error
            metadata={
                'interpretation': 'Lower is better - measures calibration accuracy',
                'n_bins': n_bins
            }
        ))
        
        return results
    
    def validate_continuous_ranked_probability_score(
        self,
        y_true: np.ndarray,
        forecast_samples: np.ndarray
    ) -> ValidationResult:
        """
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        forecast_samples : np.ndarray
            Forecast samples (shape: n_samples x n_forecasts)
        Returns:
        --------
        ValidationResult
        """
        # Simplified CRPS calculation
        n_samples = forecast_samples.shape[1]
        
        # Term 1: Expected distance from forecast to observation
        term1 = np.mean([
            np.mean(np.abs(forecast_samples[:, i] - y_true))
            for i in range(n_samples)
        ])
        
        # Term 2: Expected distance between forecast samples
        term2 = 0
        count = 0
        for i in range(min(n_samples, 100)):  # Limit for efficiency
            for j in range(i + 1, min(n_samples, 100)):
                term2 += np.mean(np.abs(forecast_samples[:, i] - forecast_samples[:, j]))
                count += 1
        term2 = term2 / count if count > 0 else 0
        
        crps = term1 - 0.5 * term2
        
        return ValidationResult(
            test_name="Continuous Ranked Probability Score",
            statistic=crps,
            metadata={
                'interpretation': 'Lower is better - measures forecast skill',
                'n_samples': n_samples
            }
        )
