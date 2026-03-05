"""
model comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.stats import t

from ..core.results import ValidationResult, ValidationReport
from ..api import validate_forecast


class ModelComparator:
    def __init__(self):
        self.models = {}
        self.reports = {}
    
    def add_model(
        self,
        name: str,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray
    ):
        """
        Parameters:
        -----------
        name : str
            Model name
        y_train : np.ndarray
            Training actual values
        y_train_pred : np.ndarray
            Training predictions
        y_test : np.ndarray
            Test actual values
        y_test_pred : np.ndarray
            Test predictions
        """
        report = validate_forecast(
            y_train, y_train_pred,
            y_test, y_test_pred,
            model_name=name,
            print_report=False
        )
        
        self.models[name] = {
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_train': y_train,
            'y_train_pred': y_train_pred
        }
        self.reports[name] = report
    
    def compare_metrics(
        self,
        metrics_to_compare: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Parameters:
        -----------
        metrics_to_compare : list, optional
            List of metric names to compare. If None, uses default set.
        Returns:
        --------
        pd.DataFrame
            Comparison table with rankings
        """
        if metrics_to_compare is None:
            metrics_to_compare = [
                'test_RMSE', 'test_MAE', 'test_MAPE', 'test_R2',
                'test_Directional_Accuracy'
            ]
        
        comparison = {}
        for name, report in self.reports.items():
            comparison[name] = {
                metric: report.metrics.get(metric, np.nan)
                for metric in metrics_to_compare
            }
        
        df = pd.DataFrame(comparison).T
        
        # Rank models (lower is better for error metrics)
        error_metrics = [m for m in df.columns if any(
            err in m.lower() for err in ['mse', 'rmse', 'mae', 'mape', 'error']
        )]
        for col in error_metrics:
            if col in df.columns:
                df[f'{col}_rank'] = df[col].rank()
        
        # Higher is better for R2, accuracy, etc.
        score_metrics = [m for m in df.columns if any(
            score in m.lower() for score in ['r2', 'accuracy', 'score']
        )]
        for col in score_metrics:
            if col in df.columns:
                df[f'{col}_rank'] = df[col].rank(ascending=False)
        
        # Calculate average rank
        rank_cols = [c for c in df.columns if c.endswith('_rank')]
        if rank_cols:
            df['avg_rank'] = df[rank_cols].mean(axis=1)
            df = df.sort_values('avg_rank')
        
        return df
    
    def diebold_mariano_test(
        self,
        model1: str,
        model2: str,
        loss_function: str = 'squared'
    ) -> ValidationResult:
        """
        Parameters:
        -----------
        model1 : str
            Name of first model
        model2 : str
            Name of second model
        loss_function : str
            'squared' for squared error, 'absolute' for absolute error
        Returns:
        --------
        ValidationResult
        """
        y_true = self.models[model1]['y_test']
        pred1 = self.models[model1]['y_test_pred']
        pred2 = self.models[model2]['y_test_pred']
        
        e1 = y_true - pred1
        e2 = y_true - pred2
        
        if loss_function == 'squared':
            d = e1**2 - e2**2
        elif loss_function == 'absolute':
            d = np.abs(e1) - np.abs(e2)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        # DM statistic
        dm_stat = d_mean / np.sqrt(d_var / len(d))
        
        # P-value (two-tailed)
        p_value = 2 * (1 - t.cdf(abs(dm_stat), len(d) - 1))
        
        if dm_stat < 0:
            better_model = model1
            interpretation = f"{model1} has significantly lower {loss_function} error"
        else:
            better_model = model2
            interpretation = f"{model2} has significantly lower {loss_function} error"
        
        return ValidationResult(
            test_name=f"Diebold-Mariano: {model1} vs {model2}",
            statistic=dm_stat,
            p_value=p_value,
            passed=p_value < 0.05,
            metadata={
                'interpretation': interpretation if p_value < 0.05 else 'No significant difference',
                'better_model': better_model if p_value < 0.05 else 'tie',
                'loss_function': loss_function
            }
        )
    
    def pairwise_dm_tests(self) -> pd.DataFrame:
        """
        Returns:
        --------
        pd.DataFrame
            Results of all pairwise comparisons
        """
        model_names = list(self.models.keys())
        results = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                dm_result = self.diebold_mariano_test(
                    model_names[i],
                    model_names[j]
                )
                results.append({
                    'model_1': model_names[i],
                    'model_2': model_names[j],
                    'dm_statistic': dm_result.statistic,
                    'p_value': dm_result.p_value,
                    'significant': dm_result.passed,
                    'better_model': dm_result.metadata.get('better_model')
                })
        
        return pd.DataFrame(results)
    
    def generate_report(self) -> str:
        """
        Returns:
        --------
        str
            Formatted report text
        """
        report = "=" * 80 + "\n"
        report += "MODEL COMPARISON REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Metrics comparison
        comparison_df = self.compare_metrics()
        report += "Metrics Comparison:\n"
        report += "-" * 80 + "\n"
        report += comparison_df.to_string() + "\n\n"
        
        # Best model by metric
        report += "Best Model by Metric:\n"
        report += "-" * 80 + "\n"
        
        error_metrics = ['test_RMSE', 'test_MAE', 'test_MAPE']
        for metric in error_metrics:
            if metric in comparison_df.columns:
                best_model = comparison_df[metric].idxmin()
                value = comparison_df.loc[best_model, metric]
                report += f"  {metric}: {best_model} ({value:.4f})\n"
        
        score_metrics = ['test_R2', 'test_Directional_Accuracy']
        for metric in score_metrics:
            if metric in comparison_df.columns:
                best_model = comparison_df[metric].idxmax()
                value = comparison_df.loc[best_model, metric]
                report += f"  {metric}: {best_model} ({value:.4f})\n"
        
        if 'avg_rank' in comparison_df.columns:
            report += f"\nOverall Best Model (by avg rank): {comparison_df.index[0]}\n"
        
        # Statistical tests
        if len(self.models) >= 2:
            report += "\n" + "=" * 80 + "\n"
            report += "Statistical Comparison Tests (Diebold-Mariano):\n"
            report += "=" * 80 + "\n\n"
            
            dm_results = self.pairwise_dm_tests()
            for _, row in dm_results.iterrows():
                report += f"{row['model_1']} vs {row['model_2']}:\n"
                report += f"  DM Statistic: {row['dm_statistic']:.4f}\n"
                report += f"  P-value: {row['p_value']:.4f}\n"
                if row['significant']:
                    report += f"  Result: {row['better_model']} is significantly better ✓\n"
                else:
                    report += f"  Result: No significant difference\n"
                report += "\n"
        
        return report
    
    def get_best_model(self, criterion: str = 'test_RMSE') -> Tuple[str, ValidationReport]:
        """
        Parameters:
        -----------
        criterion : str
            Metric to use for selection (default: 'test_RMSE')
        Returns:
        --------
        model_name : str
            Name of best model
        report : ValidationReport
            Validation report for best model
        """
        comparison = self.compare_metrics([criterion])
        
        # Lower is better for error metrics
        if any(err in criterion.lower() for err in ['mse', 'rmse', 'mae', 'mape', 'error']):
            best_model = comparison[criterion].idxmin()
        else:
            best_model = comparison[criterion].idxmax()
        
        return best_model, self.reports[best_model]
