"""
sktime integration utilities
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Tuple, Dict


def convert_to_sktime_format(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    freq: Optional[str] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None
) -> pd.Series:
    """
    Parameters:
    -----------
    data : pd.DataFrame, pd.Series, or np.ndarray
        Input data
    freq : str, optional
        Frequency string (e.g., 'D', 'M', 'H', 'Y')
        If None, will try to infer from data
    start_date : str or pd.Timestamp, optional
        Start date for the time series
        If None and data doesn't have datetime index, will use today's date
    Returns:
    --------
    pd.Series
        Time series in sktime format with datetime/period index
    """
    
    # Convert DataFrame to Series
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            data = data.iloc[:, 0]
        else:
            raise ValueError(
                f"DataFrame has {len(data.columns)} columns. "
                "Please select a single column or pass a Series/array."
            )
    
    # Convert numpy array to Series
    if isinstance(data, np.ndarray):
        data = pd.Series(data.flatten())
    
    # Ensure we have a Series
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Check if already has datetime/period index
    if isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if data.name is None:
            data.name = 'value'
        return data
    
    # Create datetime index
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()
    else:
        start_date = pd.Timestamp(start_date)
    
    # Infer or use provided frequency
    if freq is None:
        freq = 'D'
        warnings.warn(
            f"No frequency provided. Defaulting to daily frequency ('{freq}'). "
            "Specify freq parameter for accurate time indexing.",
            UserWarning
        )
    
    # Create new datetime index
    try:
        new_index = pd.date_range(start=start_date, periods=len(data), freq=freq)
        data.index = new_index
    except Exception as e:
        raise ValueError(f"Could not create datetime index with freq='{freq}': {e}")
    
    # Set name if not present
    if data.name is None:
        data.name = 'value'
    
    return data


def prepare_sktime_train_test(
    data: Union[pd.Series, np.ndarray],
    train_size: Optional[Union[int, float]] = None,
    test_size: Optional[Union[int, float]] = None,
    freq: Optional[str] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Input time series data
    train_size : int or float, optional
        Size of training set (if float, proportion; if int, number of samples)
    test_size : int or float, optional
        Size of test set (if float, proportion; if int, number of samples)
    freq : str, optional
        Frequency of time series
    start_date : str or pd.Timestamp, optional
        Start date for time series
    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        Training and test sets in sktime format
    """
    # Convert to sktime format first
    series = convert_to_sktime_format(data, freq=freq, start_date=start_date)
    
    n = len(series)
    
    # Determine split point
    if train_size is not None:
        if isinstance(train_size, float) and 0 < train_size < 1:
            split_idx = int(n * train_size)
        elif isinstance(train_size, int):
            split_idx = train_size
        else:
            raise ValueError("train_size must be float in (0,1) or positive integer")
    elif test_size is not None:
        if isinstance(test_size, float) and 0 < test_size < 1:
            split_idx = int(n * (1 - test_size))
        elif isinstance(test_size, int):
            split_idx = n - test_size
        else:
            raise ValueError("test_size must be float in (0,1) or positive integer")
    else:
        # Default to 80/20 split
        split_idx = int(n * 0.8)
    
    # Split the data
    y_train = series.iloc[:split_idx]
    y_test = series.iloc[split_idx:]
    
    return y_train, y_test


def create_sktime_dataset_from_arrays(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    freq: str = 'D',
    start_date: Optional[Union[str, pd.Timestamp]] = None
) -> Dict[str, pd.Series]:
    """
    Parameters:
    -----------
    y_train : np.ndarray
        Training actuals
    y_test : np.ndarray
        Test actuals
    y_train_pred : np.ndarray
        Training predictions
    y_test_pred : np.ndarray
        Test predictions
    freq : str
        Frequency string
    start_date : str or pd.Timestamp, optional
        Start date
    Returns:
    --------
    Dict[str, pd.Series]
        Dictionary with 'y_train', 'y_test', 'y_train_pred', 'y_test_pred'
    """
    # Combine all data to create continuous index
    n_train = len(y_train)
    n_total = n_train + len(y_test)
    
    # Create full datetime index
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()
    
    full_index = pd.date_range(start=start_date, periods=n_total, freq=freq)
    
    # Create Series with proper indices
    y_train_series = pd.Series(y_train, index=full_index[:n_train], name='actual')
    y_test_series = pd.Series(y_test, index=full_index[n_train:], name='actual')
    y_train_pred_series = pd.Series(y_train_pred, index=full_index[:n_train], name='predicted')
    y_test_pred_series = pd.Series(y_test_pred, index=full_index[n_train:], name='predicted')
    
    return {
        'y_train': y_train_series,
        'y_test': y_test_series,
        'y_train_pred': y_train_pred_series,
        'y_test_pred': y_test_pred_series
    }
