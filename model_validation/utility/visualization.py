"""
Visualization utilities for validation results.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats


def plot_residuals(
    residuals: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Residual Analysis"
) -> plt.Figure:
    """
    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals
    timestamps : pd.DatetimeIndex, optional
        Timestamps for residuals
    figsize : tuple
        Figure size
    title : str
        Main title
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    x_axis = timestamps if timestamps is not None else np.arange(len(residuals))
    
    # Residuals over time
    axes[0, 0].plot(x_axis, residuals, linewidth=1.5, alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time' if timestamps is not None else 'Index')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Normal curve overlay
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(
        x,
        stats.norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 30,
        'r-',
        linewidth=2,
        label='Normal'
    )
    axes[0, 1].legend()
    
    # QQ plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs fitted
    axes[1, 1].scatter(np.arange(len(residuals)), residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Residuals vs Index', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    train_test_split: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Predictions vs Actuals"
) -> plt.Figure:
    """
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    timestamps : pd.DatetimeIndex, optional
        Timestamps
    train_test_split : int, optional
        Index where train/test split occurs
    figsize : tuple
        Figure size
    title : str
        Plot title
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_axis = timestamps if timestamps is not None else np.arange(len(y_true))
    
    ax.plot(x_axis, y_true, linewidth=2, label='Actual', alpha=0.8)
    ax.plot(x_axis, y_pred, linewidth=2, label='Predicted', alpha=0.8)

    if train_test_split is not None:
        split_x = x_axis[train_test_split] if timestamps is not None else train_test_split
        ax.axvline(
            x=split_x,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label='Train/Test Split'
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time' if timestamps is not None else 'Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_diagnostics(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """
    Parameters:
    -----------
    y_train : np.ndarray
        Training actuals
    y_train_pred : np.ndarray
        Training predictions
    y_test : np.ndarray
        Test actuals
    y_test_pred : np.ndarray
        Test predictions
    figsize : tuple
        Figure size
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # training predictions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(y_train, linewidth=2, label='Actual', alpha=0.8)
    ax1.plot(y_train_pred, linewidth=2, label='Predicted', alpha=0.8)
    ax1.set_title('Training Set', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # test predictions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(y_test, linewidth=2, label='Actual', alpha=0.8)
    ax2.plot(y_test_pred, linewidth=2, label='Predicted', alpha=0.8)
    ax2.set_title('Test Set', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # training residuals
    train_residuals = y_train - y_train_pred
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(train_residuals, linewidth=1.5)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title('Training Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # test residuals
    test_residuals = y_test - y_test_pred
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(test_residuals, linewidth=1.5, color='orange')
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_title('Test Residuals', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # training residual histogram
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(train_residuals, bins=30, alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--')
    ax5.set_title('Training Residual Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Residual')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    # test residual histogram
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(test_residuals, bins=20, alpha=0.7, edgecolor='black', color='orange')
    ax6.axvline(x=0, color='red', linestyle='--')
    ax6.set_title('Test Residual Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Residual')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Diagnostics', fontsize=16, fontweight='bold')
    return fig


def create_validation_dashboard(
    validation_report,
    residuals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Parameters:
    -----------
    validation_report : ValidationReport
        Validation report object
    residuals : np.ndarray
        Model residuals
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    timestamps : pd.DatetimeIndex, optional
        Timestamps
    figsize : tuple
        Figure size
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    x_axis = timestamps if timestamps is not None else np.arange(len(y_true))
    
    # predictions/metrics
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(x_axis, y_true, linewidth=2, label='Actual', alpha=0.8)
    ax1.plot(x_axis, y_pred, linewidth=2, label='Predicted', alpha=0.8)
    ax1.set_title('Predictions vs Actuals', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # text box
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_metrics.axis('off')
    metrics_text = "Key Metrics:\n" + "="*20 + "\n"
    key_metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    for metric in key_metrics:
        if metric in validation_report.metrics:
            value = validation_report.metrics[metric]
            metrics_text += f"{metric}: {value:.4f}\n"
    ax_metrics.text(
        0.1, 0.5, metrics_text,
        fontsize=11,
        verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # residuals
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_axis, residuals, linewidth=1.5)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_title('Residuals Over Time', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.set_title('Residual Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax4 = fig.add_subplot(gs[1, 2])
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ACF and test results
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        ax5 = fig.add_subplot(gs[2, :2])
        plot_acf(residuals, lags=20, ax=ax5)
        ax5.set_title('Autocorrelation Function', fontsize=11, fontweight='bold')
    except:
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.text(0.5, 0.5, 'ACF plot requires statsmodels', 
                ha='center', va='center')
        ax5.axis('off')
    
    # test summary
    ax_tests = fig.add_subplot(gs[2, 2])
    ax_tests.axis('off')
    passed = sum(1 for r in validation_report.results if r.passed)
    failed = sum(1 for r in validation_report.results if not r.passed)
    total = len(validation_report.results)
    tests_text = f"Tests:\n{'='*20}\n"
    tests_text += f"Passed: {passed}/{total}\n"
    tests_text += f"Failed: {failed}/{total}\n"
    ax_tests.text(
        0.1, 0.5, tests_text,
        fontsize=11,
        verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    fig.suptitle(
        f'Validation Dashboard: {validation_report.model_name}',
        fontsize=16,
        fontweight='bold'
    )
    return fig
