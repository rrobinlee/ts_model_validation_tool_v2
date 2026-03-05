"""
cross-validation
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Optional
from dataclasses import dataclass, field

from ..core.results import ValidationReport
from ..core.framework import ModelValidationFramework


class TimeSeriesCrossValidator:
    def __init__(
        self,
        n_splits: int = 5,
        strategy: str = 'expanding',
        gap: int = 0,
        test_size: Optional[int] = None
    ):
        """
        Parameters:
        -----------
        n_splits : int
            Number of splits
        strategy : str
            'expanding' - growing train window, fixed test window
            'rolling' - fixed train and test windows
            'blocked' - non-overlapping blocks
        gap : int
            Number of samples to skip between train and test
        test_size : int, optional
            Size of test set (if None, auto-determined)
        """
        self.n_splits = n_splits
        self.strategy = strategy
        self.gap = gap
        self.test_size = test_size
        
        if strategy not in ['expanding', 'rolling', 'blocked']:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def split(
        self,
        X: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Parameters:
        -----------
        X : np.ndarray
            Data to split
        Yields:
        -------
        train_indices, test_indices : np.ndarray
            Indices for train and test sets
        """
        n = len(X)
        
        if self.strategy == 'expanding':
            yield from self._expanding_window_split(n)
        elif self.strategy == 'rolling':
            yield from self._rolling_window_split(n)
        elif self.strategy == 'blocked':
            yield from self._blocked_split(n)
    
    def _expanding_window_split(self, n: int):
        test_size = self.test_size or max(10, n // (self.n_splits + 1))
        initial_train_size = n - (self.n_splits * test_size) - (self.n_splits * self.gap)
        
        for i in range(self.n_splits):
            train_end = initial_train_size + i * test_size
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            if test_end > n:
                break
            
            yield (np.arange(train_end), np.arange(test_start, test_end))
    
    def _rolling_window_split(self, n: int):
        test_size = self.test_size or max(10, n // (self.n_splits + 1))
        train_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_start = i * test_size
            train_end = train_start + train_size
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            if test_end > n:
                break
            
            yield (np.arange(train_start, train_end), np.arange(test_start, test_end))
    
    def _blocked_split(self, n: int):
        block_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_indices = []
            for j in range(i + 1):
                start = j * block_size
                end = (j + 1) * block_size
                train_indices.extend(range(start, end))
            
            test_start = (i + 1) * block_size + self.gap
            test_end = min((i + 2) * block_size, n)
            
            if test_end > n:
                break
            
            yield (np.array(train_indices), np.arange(test_start, test_end))
    
    def get_n_splits(self) -> int:
        return self.n_splits


@dataclass
class CrossValidationResults:
    fold_reports: List[ValidationReport] = field(default_factory=list)
    aggregate_metrics: dict = field(default_factory=dict)
    
    def add_fold_report(self, report: ValidationReport):
        self.fold_reports.append(report)
    
    def compute_aggregates(self):
        if not self.fold_reports:
            return
        
        metric_names = list(self.fold_reports[0].metrics.keys())
        
        for metric_name in metric_names:
            values = []
            for report in self.fold_reports:
                if metric_name in report.metrics:
                    value = report.metrics[metric_name]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)
            
            if values:
                self.aggregate_metrics[f'{metric_name}_mean'] = np.mean(values)
                self.aggregate_metrics[f'{metric_name}_std'] = np.std(values)
                self.aggregate_metrics[f'{metric_name}_min'] = np.min(values)
                self.aggregate_metrics[f'{metric_name}_max'] = np.max(values)
    
    def summary(self) -> pd.DataFrame:
        data = []
        for i, report in enumerate(self.fold_reports):
            row = {'fold': i + 1}
            row.update(report.metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if self.aggregate_metrics:
            agg_row = {'fold': 'mean/std'}
            for col in df.columns:
                if col != 'fold':
                    mean_key = f'{col}_mean'
                    std_key = f'{col}_std'
                    if mean_key in self.aggregate_metrics:
                        agg_row[col] = (
                            f"{self.aggregate_metrics[mean_key]:.4f} "
                            f"± {self.aggregate_metrics[std_key]:.4f}"
                        )
            df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)
        
        return df
    
    def __repr__(self) -> str:
        n_folds = len(self.fold_reports)
        return f"CrossValidationResults(n_folds={n_folds})"


def cross_validate_model(
    y: np.ndarray,
    y_pred_all: np.ndarray,
    cv: TimeSeriesCrossValidator,
    model_name: str = "Model"
) -> CrossValidationResults:
    """
    Parameters:
    -----------
    y : np.ndarray
        Actual values
    y_pred_all : np.ndarray
        Predicted values (same length as y)
    cv : TimeSeriesCrossValidator
        Cross-validator
    model_name : str
        Name of model
    Returns:
    --------
    CrossValidationResults
    """
    results = CrossValidationResults()
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(y)):
        # Get train/test data
        y_train = y[train_idx]
        y_train_pred = y_pred_all[train_idx]
        y_test = y[test_idx]
        y_test_pred = y_pred_all[test_idx]
        
        framework = ModelValidationFramework(model_name=f"{model_name}_fold_{fold_idx+1}")
        report = framework.run_comprehensive_validation(
            y_train, y_train_pred,
            y_test, y_test_pred,
            check_stationarity=False
        )
        
        results.add_fold_report(report)
    
    results.compute_aggregates()
    return results
