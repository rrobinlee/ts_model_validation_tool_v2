"""
backtesting framework
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass

from ..core.results import ValidationReport
from ..core.framework import ModelValidationFramework


@dataclass
class BacktestResults:
    predictions: List[np.ndarray]
    actuals: List[np.ndarray]
    timestamps: List
    validation_report: ValidationReport
    metadata: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        y_pred_flat = np.concatenate(self.predictions)
        y_true_flat = np.concatenate(self.actuals)
        errors = y_true_flat - y_pred_flat
        
        all_timestamps = []
        for i, ts in enumerate(self.timestamps):
            all_timestamps.extend([ts] * len(self.predictions[i]))
        
        return pd.DataFrame({
            'timestamp': all_timestamps,
            'actual': y_true_flat,
            'predicted': y_pred_flat,
            'error': errors,
            'abs_error': np.abs(errors),
            'squared_error': errors ** 2
        })
    
    def summary_stats(self) -> Dict[str, float]:
        df = self.to_dataframe()
        
        return {
            'mean_error': df['error'].mean(),
            'std_error': df['error'].std(),
            'mean_abs_error': df['abs_error'].mean(),
            'rmse': np.sqrt(df['squared_error'].mean()),
            'min_error': df['error'].min(),
            'max_error': df['error'].max()
        }


class BacktestingFramework:
    
    def __init__(
        self,
        initial_window: int,
        step_size: int = 1,
        forecasting_horizon: int = 1,
        refit_frequency: int = 1
    ):
        """
        Parameters:
        -----------
        initial_window : int
            Size of initial training window
        step_size : int
            Number of samples to step forward each iteration
        forecasting_horizon : int
            Number of steps to forecast ahead
        refit_frequency : int
            How often to refit the model (1 = refit every iteration)
        """
        self.initial_window = initial_window
        self.step_size = step_size
        self.forecasting_horizon = forecasting_horizon
        self.refit_frequency = refit_frequency
    
    def backtest_with_predictions(
        self,
        data: pd.Series,
        all_predictions: np.ndarray,
        model_name: str = "Backtest"
    ) -> BacktestResults:
        """
        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index
        all_predictions : np.ndarray
            All predictions (same length as data)
        model_name : str
            Name for validation report
        Returns:
        --------
        BacktestResults
        """
        n = len(data)
        predictions = []
        actuals = []
        timestamps = []
        
        for i in range(self.initial_window, n - self.forecasting_horizon, self.step_size):
            forecast_start = i
            forecast_end = i + self.forecasting_horizon
            pred = all_predictions[forecast_start:forecast_end]
            actual = data.iloc[forecast_start:forecast_end].values
            predictions.append(pred)
            actuals.append(actual)
            timestamps.append(data.index[forecast_start])
        
        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(actuals)
        framework = ModelValidationFramework(model_name)
        framework.calculate_metrics(y_true, y_pred, prefix="backtest_")
        framework.validate_residuals(y_true - y_pred)
        report = framework.get_report()
        
        return BacktestResults(
            predictions=predictions,
            actuals=actuals,
            timestamps=timestamps,
            validation_report=report,
            metadata={
                'initial_window': self.initial_window,
                'step_size': self.step_size,
                'forecasting_horizon': self.forecasting_horizon,
                'n_backtests': len(predictions)
            }
        )
    
    def backtest_with_model(
        self,
        data: pd.Series,
        model_factory: Callable,
        fit_params: Optional[Dict] = None,
        predict_params: Optional[Dict] = None
    ) -> BacktestResults:
        """
        Parameters:
        -----------
        data : pd.Series
            Time series data
        model_factory : Callable
            Function that returns a new model instance
        fit_params : dict, optional
            Parameters to pass to model.fit()
        predict_params : dict, optional
            Parameters to pass to model.predict()
        Returns:
        --------
        BacktestResults
        """
        fit_params = fit_params or {}
        predict_params = predict_params or {}
        
        n = len(data)
        predictions = []
        actuals = []
        timestamps = []
        
        model = None
        refit_counter = 0
        
        for i in range(self.initial_window, n - self.forecasting_horizon, self.step_size):
            # Training data
            train_data = data.iloc[:i]
            if model is None or refit_counter % self.refit_frequency == 0:
                model = model_factory()
                model.fit(train_data, **fit_params)
            
            try:
                forecast = model.predict(self.forecasting_horizon, **predict_params)
                if hasattr(forecast, 'values'):
                    forecast = forecast.values
            except Exception as e:
                continue
            
            actual = data.iloc[i:i + self.forecasting_horizon].values
            predictions.append(forecast.flatten())
            actuals.append(actual.flatten())
            timestamps.append(data.index[i])
            
            refit_counter += 1
        
        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(actuals)
        
        framework = ModelValidationFramework("Backtest")
        framework.calculate_metrics(y_true, y_pred, prefix="backtest_")
        framework.validate_residuals(y_true - y_pred)
        report = framework.get_report()
        
        return BacktestResults(
            predictions=predictions,
            actuals=actuals,
            timestamps=timestamps,
            validation_report=report,
            metadata={
                'initial_window': self.initial_window,
                'step_size': self.step_size,
                'forecasting_horizon': self.forecasting_horizon,
                'refit_frequency': self.refit_frequency,
                'n_backtests': len(predictions)
            }
        )
    
    def rolling_window_forecast_errors(
        self,
        results: BacktestResults,
        window_size: int = 10
    ) -> pd.DataFrame:
        """
        Parameters:
        -----------
        results : BacktestResults
            Backtest results
        window_size : int
            Size of rolling window
        Returns:
        --------
        pd.DataFrame
            Rolling statistics
        """
        df = results.to_dataframe()
        
        rolling_stats = pd.DataFrame({
            'rolling_mae': df['abs_error'].rolling(window_size).mean(),
            'rolling_rmse': np.sqrt(df['squared_error'].rolling(window_size).mean()),
            'rolling_bias': df['error'].rolling(window_size).mean(),
            'rolling_std': df['error'].rolling(window_size).std()
        })
        
        rolling_stats['timestamp'] = df['timestamp']
        
        return rolling_stats
