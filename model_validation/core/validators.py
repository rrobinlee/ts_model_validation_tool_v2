"""
Statistical validators
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss

from .results import ValidationResult


class BaseValidator(ABC):
    def __init__(self, alpha: float = 0.05):
        """
        Parameters:
        -----------
        alpha : float
            Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> List[ValidationResult]:
        pass


class ResidualValidator(BaseValidator):
    def __init__(self, alpha: float = 0.05, max_lags: int = 10):
        """
        Parameters:
        -----------
        alpha : float
            Significance level for tests
        max_lags : int
            Maximum number of lags for autocorrelation tests
        """
        super().__init__(alpha)
        self.max_lags = max_lags
    
    def validate(self, residuals: np.ndarray) -> List[ValidationResult]:
        """
        Parameters:
        -----------
        residuals : np.ndarray
            Model residuals
        Returns:
        --------
        List[ValidationResult]
            List of validation results
        """
        results = []
        residuals = np.asarray(residuals).flatten()
        residuals = residuals[~np.isnan(residuals)]
        
        if len(residuals) < 10:
            return [ValidationResult(
                test_name="Residual Analysis",
                statistic=np.nan,
                p_value=np.nan,
                passed=False,
                metadata={'error': 'Insufficient residuals for analysis'}
            )]
        
        # Normality tests
        results.extend(self._test_normality(residuals))
        # Autocorrelation tests
        results.extend(self._test_autocorrelation(residuals))
        # Zero mean test
        results.append(self._test_zero_mean(residuals))
        # Homoscedasticity (constant variance)
        results.append(self._test_homoscedasticity(residuals))
        
        return results
    
    def _test_normality(self, residuals: np.ndarray) -> List[ValidationResult]:
        results = []

        # Jarque-Bera test
        try:
            jb_stat, jb_pval = jarque_bera(residuals)
            results.append(ValidationResult(
                test_name="Jarque-Bera Normality",
                statistic=jb_stat,
                p_value=jb_pval,
                passed=jb_pval > self.alpha,
                metadata={'interpretation': 'H0: Residuals are normally distributed'}
            ))
        except Exception as e:
            results.append(ValidationResult(
                test_name="Jarque-Bera Normality",
                statistic=np.nan,
                metadata={'error': str(e)}
            ))
        
        # Shapiro-Wilk test (for smaller samples)
        if len(residuals) <= 5000:
            try:
                sw_stat, sw_pval = shapiro(residuals)
                results.append(ValidationResult(
                    test_name="Shapiro-Wilk Normality",
                    statistic=sw_stat,
                    p_value=sw_pval,
                    passed=sw_pval > self.alpha,
                    metadata={'interpretation': 'H0: Residuals are normally distributed'}
                ))
            except Exception as e:
                results.append(ValidationResult(
                    test_name="Shapiro-Wilk Normality",
                    statistic=np.nan,
                    metadata={'error': str(e)}
                ))
        
        # Kolmogorov-Smirnov test
        try:
            standardized = (residuals - np.mean(residuals)) / np.std(residuals)
            ks_stat, ks_pval = kstest(standardized, 'norm')
            results.append(ValidationResult(
                test_name="Kolmogorov-Smirnov Normality",
                statistic=ks_stat,
                p_value=ks_pval,
                passed=ks_pval > self.alpha,
                metadata={'interpretation': 'H0: Residuals follow normal distribution'}
            ))
        except Exception as e:
            results.append(ValidationResult(
                test_name="Kolmogorov-Smirnov Normality",
                statistic=np.nan,
                metadata={'error': str(e)}
            ))
        
        return results
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> List[ValidationResult]:
        results = []
        
        # Ljung-Box test
        try:
            lags = min(self.max_lags, len(residuals) // 5)
            lb_result = acorr_ljungbox(residuals, lags=lags, return_df=False)
            lb_stat = lb_result[0][-1]
            lb_pval = lb_result[1][-1]
            
            results.append(ValidationResult(
                test_name="Ljung-Box Autocorrelation",
                statistic=lb_stat,
                p_value=lb_pval,
                passed=lb_pval > self.alpha,
                metadata={
                    'interpretation': 'H0: No autocorrelation',
                    'lags_tested': lags
                }
            ))
        except Exception as e:
            results.append(ValidationResult(
                test_name="Ljung-Box Autocorrelation",
                statistic=np.nan,
                metadata={'error': str(e)}
            ))
        
        # Durbin-Watson test
        try:
            dw_stat = durbin_watson(residuals)
            dw_passed = 1.5 <= dw_stat <= 2.5
            
            results.append(ValidationResult(
                test_name="Durbin-Watson Autocorrelation",
                statistic=dw_stat,
                passed=dw_passed,
                metadata={
                    'interpretation': 'Ideal value: ~2.0, Range [1.5, 2.5] acceptable'
                }
            ))
        except Exception as e:
            results.append(ValidationResult(
                test_name="Durbin-Watson Autocorrelation",
                statistic=np.nan,
                metadata={'error': str(e)}
            ))
        
        return results
    
    def _test_zero_mean(self, residuals: np.ndarray) -> ValidationResult:
        try:
            t_stat, p_val = stats.ttest_1samp(residuals, 0)
            
            return ValidationResult(
                test_name="Zero Mean Test",
                statistic=t_stat,
                p_value=p_val,
                passed=p_val > self.alpha,
                metadata={
                    'interpretation': 'H0: Mean of residuals = 0',
                    'mean': np.mean(residuals)
                }
            )
        except Exception as e:
            return ValidationResult(
                test_name="Zero Mean Test",
                statistic=np.nan,
                metadata={'error': str(e)}
            )
    
    def _test_homoscedasticity(self, residuals: np.ndarray) -> ValidationResult:
        try:
            mid = len(residuals) // 2
            first_half = residuals[:mid]
            second_half = residuals[mid:]
            
            stat, p_val = stats.levene(first_half, second_half)
            
            return ValidationResult(
                test_name="Homoscedasticity (Levene)",
                statistic=stat,
                p_value=p_val,
                passed=p_val > self.alpha,
                metadata={
                    'interpretation': 'H0: Constant variance across time',
                    'var_first_half': np.var(first_half),
                    'var_second_half': np.var(second_half)
                }
            )
        except Exception as e:
            return ValidationResult(
                test_name="Homoscedasticity (Levene)",
                statistic=np.nan,
                metadata={'error': str(e)}
            )


class StationarityValidator(BaseValidator):
    def validate(self, series: np.ndarray) -> List[ValidationResult]:
        """
        Parameters:
        -----------
        series : np.ndarray
            Time series data
        Returns:
        --------
        List[ValidationResult]
            List of validation results
        """
        results = []
        series = np.asarray(series).flatten()
        series = series[~np.isnan(series)]
        
        # Augmented Dickey-Fuller test
        results.append(self._adf_test(series))
        
        # KPSS test
        results.append(self._kpss_test(series))
        
        return results
    
    def _adf_test(self, series: np.ndarray) -> ValidationResult:
        try:
            result = adfuller(series, autolag='AIC')
            adf_stat, p_val, _, _, critical_vals, _ = result
            
            return ValidationResult(
                test_name="Augmented Dickey-Fuller",
                statistic=adf_stat,
                p_value=p_val,
                critical_values=critical_vals,
                passed=p_val < self.alpha,
                metadata={
                    'interpretation': 'H0: Series has unit root (non-stationary)',
                    'decision': 'Stationary' if p_val < self.alpha else 'Non-stationary'
                }
            )
        except Exception as e:
            return ValidationResult(
                test_name="Augmented Dickey-Fuller",
                statistic=np.nan,
                metadata={'error': str(e)}
            )
    
    def _kpss_test(self, series: np.ndarray) -> ValidationResult:
        try:
            kpss_stat, p_val, _, critical_vals = kpss(series, regression='c', nlags='auto')
            
            return ValidationResult(
                test_name="KPSS Stationarity",
                statistic=kpss_stat,
                p_value=p_val,
                critical_values=critical_vals,
                passed=p_val > self.alpha,
                metadata={
                    'interpretation': 'H0: Series is stationary',
                    'decision': 'Stationary' if p_val > self.alpha else 'Non-stationary'
                }
            )
        except Exception as e:
            return ValidationResult(
                test_name="KPSS Stationarity",
                statistic=np.nan,
                metadata={'error': str(e)}
            )


class AdvancedResidualValidator(ResidualValidator):
    def validate(
        self,
        residuals: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> List[ValidationResult]:
        """
        Parameters:
        -----------
        residuals : np.ndarray
            Model residuals
        X : np.ndarray, optional
            Predictor variables for heteroscedasticity tests
        Returns:
        --------
        List[ValidationResult]
        """
        # Get base results
        results = super().validate(residuals)
        
        # Runs test for randomness
        results.append(self._test_runs(residuals))
        
        # Outlier detection
        results.append(self._detect_outliers(residuals))
        
        return results
    
    def _test_runs(self, residuals: np.ndarray) -> ValidationResult:
        from scipy.stats import norm
        try:
            signs = np.sign(residuals)
            runs = 1 + np.sum(signs[1:] != signs[:-1])
            n_pos = np.sum(signs > 0)
            n_neg = np.sum(signs < 0)
            n = n_pos + n_neg
            expected_runs = 2 * n_pos * n_neg / n + 1
            var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
            
            z_stat = (runs - expected_runs) / np.sqrt(var_runs)
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            
            return ValidationResult(
                test_name="Runs Test (Randomness)",
                statistic=z_stat,
                p_value=p_value,
                passed=p_value > self.alpha,
                metadata={
                    'interpretation': 'H0: Residuals are random',
                    'n_runs': runs,
                    'expected_runs': expected_runs
                }
            )
        except Exception as e:
            return ValidationResult(
                test_name="Runs Test (Randomness)",
                statistic=np.nan,
                metadata={'error': str(e)}
            )
    
    def _detect_outliers(self, residuals: np.ndarray) -> ValidationResult:
        try:
            Q1 = np.percentile(residuals, 25)
            Q3 = np.percentile(residuals, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = (residuals < lower_bound) | (residuals > upper_bound)
            n_outliers = np.sum(outliers)
            pct_outliers = n_outliers / len(residuals) * 100
            
            return ValidationResult(
                test_name="Outlier Detection",
                statistic=n_outliers,
                passed=pct_outliers < 5,
                metadata={
                    'percentage': pct_outliers,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_indices': np.where(outliers)[0].tolist()[:10]
                }
            )
        except Exception as e:
            return ValidationResult(
                test_name="Outlier Detection",
                statistic=np.nan,
                metadata={'error': str(e)}
            )
