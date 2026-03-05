"""
Model Validation Framework
===========================

Author: Robin Lee
Version: 2.0.0
"""

# Core classes
from .core.validators import (
    BaseValidator,
    ResidualValidator,
    StationarityValidator,
    AdvancedResidualValidator
)

from .core.metrics import MetricsCalculator

from .core.results import (
    ValidationResult,
    ValidationReport
)

from .core.framework import ModelValidationFramework

# Extensions
from .extensions.cross_validation import (
    TimeSeriesCrossValidator,
    CrossValidationResults
)

from .extensions.probabilistic import ProbabilisticValidator

from .extensions.comparison import ModelComparator

from .extensions.backtesting import (
    BacktestingFramework,
    BacktestResults
)

# Utilities
from .utils.sktime_utils import (
    convert_to_sktime_format,
    prepare_sktime_train_test,
    create_sktime_dataset_from_arrays
)

from .utils.visualization import (
    plot_residuals,
    plot_predictions,
    plot_diagnostics,
    create_validation_dashboard
)

from .utils.reporting import (
    HTMLReportGenerator,
    export_to_csv,
    export_to_json
)

# Convenience functions
from .api import (
    quick_validate,
    validate_forecast,
    compare_models,
    run_backtest
)

__version__ = "2.0.0"

__all__ = [
    # Core
    'BaseValidator',
    'ResidualValidator',
    'StationarityValidator',
    'AdvancedResidualValidator',
    'MetricsCalculator',
    'ValidationResult',
    'ValidationReport',
    'ModelValidationFramework',
    
    # Extensions
    'TimeSeriesCrossValidator',
    'CrossValidationResults',
    'ProbabilisticValidator',
    'ModelComparator',
    'BacktestingFramework',
    'BacktestResults',
    
    # Utilities
    'convert_to_sktime_format',
    'prepare_sktime_train_test',
    'create_sktime_dataset_from_arrays',
    'plot_residuals',
    'plot_predictions',
    'plot_diagnostics',
    'create_validation_dashboard',
    'HTMLReportGenerator',
    'export_to_csv',
    'export_to_json',
    
    # API
    'quick_validate',
    'validate_forecast',
    'compare_models',
    'run_backtest',
]