"""
Performance metrics calculator
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


class MetricsCalculator:
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        prefix : str
            Prefix for metric names (e.g., 'train_', 'test_')
        Returns:
        --------
        Dict[str, float]
            Dictionary of metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {f"{prefix}error": "No valid predictions"}
    
        metrics = {}
        metrics[f'{prefix}MSE'] = mean_squared_error(y_true, y_pred)
        metrics[f'{prefix}RMSE'] = np.sqrt(metrics[f'{prefix}MSE'])
        metrics[f'{prefix}MAE'] = mean_absolute_error(y_true, y_pred)

        try:
            if np.any(y_true == 0):
                epsilon = np.finfo(float).eps
                metrics[f'{prefix}MAPE'] = np.mean(
                    np.abs((y_true - y_pred) / (y_true + epsilon))
                ) * 100
            else:
                metrics[f'{prefix}MAPE'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            metrics[f'{prefix}MAPE'] = np.nan
        
        try:
            metrics[f'{prefix}R2'] = r2_score(y_true, y_pred)
        except:
            metrics[f'{prefix}R2'] = np.nan

        residuals = y_true - y_pred
        metrics[f'{prefix}Mean_Error'] = np.mean(residuals)
        metrics[f'{prefix}Std_Error'] = np.std(residuals)
        metrics[f'{prefix}Max_Error'] = np.max(np.abs(residuals))
        metrics[f'{prefix}Min_Error'] = np.min(residuals)
        
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            direction_correct = np.sum(np.sign(y_true_diff) == np.sign(y_pred_diff))
            metrics[f'{prefix}Directional_Accuracy'] = direction_correct / len(y_true_diff) * 100

        metrics[f'{prefix}Bias'] = np.mean(residuals)

        if np.std(y_true) > 0:
            metrics[f'{prefix}NRMSE'] = metrics[f'{prefix}RMSE'] / np.std(y_true)
        else:
            metrics[f'{prefix}NRMSE'] = np.nan
        
        if len(y_true) > 1:
            mae_naive = np.mean(np.abs(np.diff(y_true)))
            if mae_naive > 0:
                metrics[f'{prefix}MASE'] = metrics[f'{prefix}MAE'] / mae_naive
            else:
                metrics[f'{prefix}MASE'] = np.nan
        
        return metrics
    
    @staticmethod
    def calculate_quantile_metrics(
        y_true: np.ndarray,
        quantile_forecasts: Dict[float, np.ndarray]
    ) -> Dict[str, float]:
        """
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        quantile_forecasts : Dict[float, np.ndarray]
            Dictionary mapping quantiles to forecasts
        Returns:
        --------
        Dict[str, float]
            Quantile forecast metrics
        """
        metrics = {}
        
        for q, y_pred_q in quantile_forecasts.items():
            # Pinball loss
            error = y_true - y_pred_q
            loss = np.where(error >= 0, q * error, (q - 1) * error)
            metrics[f'quantile_{q:.2f}_pinball_loss'] = np.mean(loss)
            
            # Empirical coverage
            coverage = np.mean(y_true <= y_pred_q)
            metrics[f'quantile_{q:.2f}_coverage'] = coverage
            
            # Coverage deviation
            metrics[f'quantile_{q:.2f}_coverage_deviation'] = abs(coverage - q)
        
        return metrics
