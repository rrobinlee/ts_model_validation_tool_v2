"""
High-level API for common validation tasks.
"""

import numpy as np
from typing import Optional, Union
import pandas as pd

from .core.framework import ModelValidationFramework
from .core.results import ValidationReport
from .extensions.comparison import ModelComparator
from .extensions.backtesting import BacktestingFramework, BacktestResults


def quick_validate(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    print_report: bool = True
) -> ValidationReport:
    """
    Parameters:
    -----------
    y_true : np.ndarray or pd.Series
        Actual values
    y_pred : np.ndarray or pd.Series
        Predicted values
    model_name : str
        Name of the model
    print_report : bool
        Whether to print the report
    Returns:
    --------
    ValidationReport
    
    Examples:
    ---------
    >>> report = quick_validate(y_test, predictions, model_name="ARIMA")
    """
    framework = ModelValidationFramework(model_name=model_name)
    
    # Calculate metrics
    framework.calculate_metrics(y_true, y_pred, prefix="")
    
    # Validate residuals
    residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
    framework.validate_residuals(residuals)
    
    # Generate warnings
    framework._generate_warnings()
    
    if print_report:
        framework.print_summary()
    
    return framework.get_report()


def validate_forecast(
    y_train: Union[np.ndarray, pd.Series],
    y_train_pred: Union[np.ndarray, pd.Series],
    y_test: Union[np.ndarray, pd.Series],
    y_test_pred: Union[np.ndarray, pd.Series],
    model_name: str = "Forecast Model",
    print_report: bool = True,
    check_stationarity: bool = True
) -> ValidationReport:
    """
    Parameters:
    -----------
    y_train : array-like
        Training actual values
    y_train_pred : array-like
        Training predictions
    y_test : array-like
        Test actual values
    y_test_pred : array-like
        Test predictions
    model_name : str
        Name of the model
    print_report : bool
        Whether to print summary
    check_stationarity : bool
        Whether to test stationarity
    Returns:
    --------
    ValidationReport
    
    Examples:
    ---------
    >>> report = validate_forecast(
    ...     y_train, train_preds, y_test, test_preds,
    ...     model_name="Prophet"
    ... )
    """
    framework = ModelValidationFramework(model_name=model_name)
    
    report = framework.run_comprehensive_validation(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        check_stationarity=check_stationarity
    )
    
    if print_report:
        framework.print_summary()
    
    return report


def compare_models(
    models_data: dict,
    print_report: bool = True
) -> ModelComparator:
    """
    Parameters:
    -----------
    models_data : dict
        Dictionary mapping model names to tuples of
        (y_train, y_train_pred, y_test, y_test_pred)
    print_report : bool
        Whether to print comparison report
    Returns:
    --------
    ModelComparator
    
    Examples:
    ---------
    >>> models = {
    ...     'ARIMA': (y_train, arima_train_pred, y_test, arima_test_pred),
    ...     'Prophet': (y_train, prophet_train_pred, y_test, prophet_test_pred)
    ... }
    >>> comparator = compare_models(models)
    """
    comparator = ModelComparator()
    
    for name, data in models_data.items():
        y_train, y_train_pred, y_test, y_test_pred = data
        comparator.add_model(name, y_train, y_train_pred, y_test, y_test_pred)
    
    if print_report:
        print(comparator.generate_report())
    
    return comparator


def run_backtest(
    data: pd.Series,
    all_predictions: np.ndarray,
    initial_window: int,
    step_size: int = 1,
    forecasting_horizon: int = 1,
    model_name: str = "Backtest"
) -> BacktestResults:
    """
    Parameters:
    -----------
    data : pd.Series
        Time series data with datetime index
    all_predictions : np.ndarray
        All predictions (same length as data)
    initial_window : int
        Size of initial training window
    step_size : int
        Number of samples to step forward
    forecasting_horizon : int
        Number of steps to forecast ahead
    model_name : str
        Name for validation report
    Returns:
    --------
    BacktestResults
    
    Examples:
    ---------
    >>> results = run_backtest(
    ...     data, predictions,
    ...     initial_window=100,
    ...     forecasting_horizon=10
    ... )
    """
    backtester = BacktestingFramework(
        initial_window=initial_window,
        step_size=step_size,
        forecasting_horizon=forecasting_horizon
    )
    
    return backtester.backtest_with_predictions(data, all_predictions, model_name)
