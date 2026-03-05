# Time Series Model Validation Tool

Consolidated Python module for model diagnostics and validation; streamlines statistical tests, residual analysis, and out-of-sample metrics for time series forecasting.

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Optional Dependencies

```bash
# Install with sktime support
pip install -e ".[sktime]"

# Install with visualization tools
pip install -e ".[viz]"

# Install all optional dependencies
pip install -e ".[sktime,viz,dev]"
```

## Package Structure

```
model_validation/
│
├── __init__.py              # Main imports
├── api.py                   # High-level convenience functions
│
├── core/                    # Core validation components
│   ├── __init__.py
│   ├── framework.py         # ModelValidationFramework class
│   ├── validators.py        # Residual & stationarity validators
│   ├── metrics.py           # Performance metrics calculator
│   └── results.py           # ValidationResult & ValidationReport
│
├── extensions/              # Advanced features
│   ├── __init__.py
│   ├── cross_validation.py  # Time series cross-validation
│   ├── probabilistic.py     # Probabilistic forecast validation
│   ├── comparison.py        # Model comparison tools
│   └── backtesting.py       # Backtesting framework
│
└── utils/                   # Utility functions
    ├── __init__.py
    ├── sktime_utils.py      # sktime integration
    ├── visualization.py     # Plotting functions
    └── reporting.py         # Report generation
```

## Available Tests

### Residual Diagnostics

| Test | Null Hypothesis | Purpose |
|------|----------------|---------|
| **Jarque-Bera** | Residuals are normally distributed | Test normality |
| **Shapiro-Wilk** | Residuals are normally distributed | Test normality (small samples) |
| **Kolmogorov-Smirnov** | Residuals follow normal distribution | Test normality |
| **Ljung-Box** | No autocorrelation in residuals | Detect autocorrelation |
| **Durbin-Watson** | No first-order autocorrelation | Quick autocorrelation check |
| **Zero Mean Test** | Mean of residuals = 0 | Check for bias |
| **Levene's Test** | Constant variance over time | Test homoscedasticity |

### Stationarity Tests

| Test | Null Hypothesis | Interpretation |
|------|----------------|----------------|
| **ADF (Augmented Dickey-Fuller)** | Series has unit root (non-stationary) | Reject H0 → Stationary |
| **KPSS** | Series is stationary | Fail to reject H0 → Stationary |

### Performance Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MSE** | Mean Squared Error | Lower is better |
| **RMSE** | Root Mean Squared Error | Lower is better |
| **MAE** | Mean Absolute Error | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | < 10% excellent, < 20% good |
| **R²** | Coefficient of Determination | > 0.7 good, > 0.9 excellent |
| **NRMSE** | Normalized RMSE | < 0.5 acceptable |
| **Directional Accuracy** | % of correct direction predictions | > 50% better than random |
| **Bias** | Mean residual | Close to 0 |

### Common Issues and Solutions

| Issue | Symptom | Potential Solutions |
|-------|---------|-------------------|
| **Non-normal residuals** | Failed normality tests | Transform target variable, check for outliers |
| **Autocorrelated residuals** | Failed Ljung-Box | Add AR terms, increase model order |
| **Heteroscedasticity** | Failed Levene's test | Log transform, use weighted regression |
| **High MAPE** | MAPE > 20% | Feature engineering, try different model |
| **Low R²** | R² < 0.5 | Add features, increase model complexity |
| **Non-stationary series** | Failed ADF/KPSS | Difference the series, add trend terms |

## Quick Start

### Validation Methods

```python
# basic validation
from model_validation import quick_validate
report = quick_validate(y_test, predictions, model_name="model name")

# comprehensive validation
from model_validation import validate_forecast
report = validate_forecast(
    y_train, train_preds,
    y_test, test_preds,
    model_name="SARIMA"
)
report.summary() # summary dataframe

# comparing multiple models
from model_validation import compare_models
models = {
    'ARIMA': (y_train, arima_train, y_test, arima_test),
    'Prophet': (y_train, prophet_train, y_test, prophet_test),
    'LSTM': (y_train, lstm_train, y_test, lstm_test)
}
comparator = compare_models(models)
comparison_df = comparator.compare_metrics()
dm_result = comparator.diebold_mariano_test('ARIMA', 'Prophet')

# cross-validation
from model_validation.extensions import TimeSeriesCrossValidator, cross_validate_model
cv = TimeSeriesCrossValidator(
    n_splits=5,
    strategy='expanding',  # or 'rolling', 'blocked'
    gap=0
)
cv_results = cross_validate_model(y, y_pred_all, cv, model_name="model name")
summary_df = cv_results.summary()
print(cv_results.aggregate_metrics)

# backtesting
from model_validation import run_backtest
results = run_backtest(
    data=time_series,
    all_predictions=predictions,
    initial_window=100,
    step_size=1,
    forecasting_horizon=10
)
print(results.validation_report.metrics)
df = results.to_dataframe()

# probabilistic intervals
from model_validation.extensions import ProbabilisticValidator
prob_validator = ProbabilisticValidator()
interval_results = prob_validator.validate_prediction_intervals(
    y_test, lower_bounds, upper_bounds, confidence_level=0.95
)
quantile_forecasts = {
    0.1: q10_predictions,
    0.5: q50_predictions,
    0.9: q90_predictions
}
quantile_results = prob_validator.validate_quantile_forecasts(
    y_test, quantile_forecasts
)
```

### Visualization
```python
# basic plots
from model_validation.utils import plot_residuals, plot_predictions, plot_diagnostics
fig = plot_residuals(residuals, timestamps=data.index)
fig = plot_predictions(y_test, predictions, timestamps=test_dates)
fig = plot_diagnostics(y_train, train_pred, y_test, test_pred)

# dashboard
fig = create_validation_dashboard(
    validation_report=report,
    residuals=residuals,
    y_true=y_test,
    y_pred=predictions,
    timestamps=test_dates
)

# html report
from model_validation.utils import HTMLReportGenerator
generator = HTMLReportGenerator()
html_content = generator.generate_html_report(report)
with open('validation_report.html', 'w') as f:
    f.write(html_content)

# export as file
from model_validation.utils import export_to_csv, export_to_json
export_to_csv(report, 'results.csv')
export_to_json(report, 'results.json', pretty=True)

# markdown report
from model_validation.utils import create_markdown_report
markdown = create_markdown_report(report)
with open('report.md', 'w') as f:
    f.write(markdown)
```

### Advanced Validation
```python
# custom validator
from model_validation.core import BaseValidator, ValidationResult
class MyCustomValidator(BaseValidator):
    def validate(self, data):
        # Your custom validation logic
        result = ValidationResult(
            test_name="My Custom Test",
            statistic=some_value,
            p_value=p_val,
            passed=p_val > self.alpha
        )
        return [result]
from model_validation.core import ModelValidationFramework
framework = ModelValidationFramework()
framework.report.results.extend(MyCustomValidator().validate(data))

# advanced residual validation
from model_validation.core import AdvancedResidualValidator

validator = AdvancedResidualValidator(alpha=0.05, max_lags=20)
results = validator.validate(residuals)
# runs test for randomness and outlier detection
```

## Module Reference

### Core Modules

#### `model_validation.core.framework`
- `ModelValidationFramework`: Main validation class

#### `model_validation.core.validators`
- `ResidualValidator`: Residual diagnostics
- `StationarityValidator`: Stationarity tests
- `AdvancedResidualValidator`: Extended diagnostics

#### `model_validation.core.metrics`
- `MetricsCalculator`: Performance metrics

#### `model_validation.core.results`
- `ValidationResult`: Individual test result
- `ValidationReport`: Comprehensive report

### Extension Modules

#### `model_validation.extensions.cross_validation`
- `TimeSeriesCrossValidator`: CV strategies
- `CrossValidationResults`: CV results container
- `cross_validate_model()`: Convenience function

#### `model_validation.extensions.probabilistic`
- `ProbabilisticValidator`: Probabilistic forecast validation

#### `model_validation.extensions.comparison`
- `ModelComparator`: Model comparison and statistical tests

#### `model_validation.extensions.backtesting`
- `BacktestingFramework`: Automated backtesting
- `BacktestResults`: Backtest results container

### Utility Modules

#### `model_validation.utils.sktime_utils`
- `convert_to_sktime_format()`: Convert to sktime
- `prepare_sktime_train_test()`: Train/test split
- `create_sktime_dataset_from_arrays()`: Create dataset

#### `model_validation.utils.visualization`
- `plot_residuals()`: Residual plots
- `plot_predictions()`: Prediction plots
- `plot_diagnostics()`: Comprehensive diagnostics
- `create_validation_dashboard()`: Full dashboard

#### `model_validation.utils.reporting`
- `HTMLReportGenerator`: HTML report generation
- `export_to_csv()`: CSV export
- `export_to_json()`: JSON export
- `create_markdown_report()`: Markdown reports

### API Module

#### `model_validation.api`
- `quick_validate()`: Quick validation
- `validate_forecast()`: Comprehensive validation
- `compare_models()`: Model comparison
- `run_backtest()`: Backtesting
