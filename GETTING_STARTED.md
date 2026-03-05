# Getting Started

## Installation

### Option 1: Direct Installation

```bash
cd model_validation_package
pip install -e .
```

### Option 2: Install with Optional Dependencies

```bash
# With sktime support
pip install -e ".[sktime]"

# With visualization tools
pip install -e ".[viz]"

# Everything
pip install -e ".[sktime,viz,dev]"
```

## Quick Start

### 1. Basic Validation

```python
from model_validation import quick_validate
y_test = [...]  # actual values
predictions = [...]  # predicted values
report = quick_validate(y_test, predictions, model_name="My Model")
# Access results
print(report.metrics['RMSE'])
print(report.metrics['MAPE'])
```

### 2. Comprehensive Validation (Recommended)

```python
from model_validation import validate_forecast
report = validate_forecast(y_train, train_predictions,
                           y_test, test_predictions,
                           model_name="SARIMA")
# Automatically runs:
# 1. All performance metrics (RMSE, MAE, MAPE, R2, etc.)
# 2. Residual diagnostics (normality, autocorrelation)
# 3. Stationarity tests
# 4. Generates warnings for issues
```

### 3. Compare Models

```python
from model_validation import compare_models
models = {
    'ARIMA': (y_train, arima_train, y_test, arima_test),
    'Prophet': (y_train, prophet_train, y_test, prophet_test)
}
comparator = compare_models(models)
comparison_df = comparator.compare_metrics()
result = comparator.diebold_mariano_test('ARIMA', 'Prophet')
```

--

## Module-by-Module Guide

### Core Modules

#### 1. `model_validation.core.framework`

The main validation framework class.

```python
from model_validation.core import ModelValidationFramework

# Create framework
framework = ModelValidationFramework(
    model_name="My Model",
    alpha=0.05,  # significance level
    max_lags=10  # for autocorrelation tests
)

# Add metrics
framework.calculate_metrics(y_test, predictions, prefix="test_")

# Add residual validation
framework.validate_residuals(residuals)

# Add stationarity check
framework.validate_stationarity(time_series)

# Get report
report = framework.get_report()
framework.print_summary()
```

#### 2. `model_validation.core.validators`

Individual validators for specific tests.

```python
from model_validation.core import (
    ResidualValidator,
    StationarityValidator,
    AdvancedResidualValidator
)

# Basic residual validation
validator = ResidualValidator(alpha=0.05, max_lags=10)
results = validator.validate(residuals)

# Stationarity tests
stat_validator = StationarityValidator(alpha=0.05)
results = stat_validator.validate(time_series)

# Advanced (includes outlier detection, runs test)
adv_validator = AdvancedResidualValidator(alpha=0.05)
results = adv_validator.validate(residuals)
```

#### 3. `model_validation.core.metrics`

Performance metrics calculator.

```python
from model_validation.core import MetricsCalculator

calc = MetricsCalculator()
metrics = calc.calculate_all_metrics(
    y_true, y_pred, prefix="custom_"
)

# Returns: RMSE, MAE, MAPE, R², directional accuracy, etc.
```

#### 4. `model_validation.core.results`

Result containers.

```python
from model_validation.core import ValidationResult, ValidationReport

# Individual test result
result = ValidationResult(
    test_name="My Test",
    statistic=2.5,
    p_value=0.03,
    passed=True
)

# Report with multiple results
report = ValidationReport(model_name="Model")
report.add_result(result)
report.add_metric("RMSE", 5.2)
report.add_warning("High error detected")

# Get summary
df = report.summary()
```

### Extension Modules

#### 1. `model_validation.extensions.cross_validation`

Time series cross-validation.

```python
from model_validation.extensions import (
    TimeSeriesCrossValidator,
    cross_validate_model
)

# Create CV splitter
cv = TimeSeriesCrossValidator(
    n_splits=5,
    strategy='expanding',  # or 'rolling', 'blocked'
    gap=0,
    test_size=20
)

# Run cross-validation
results = cross_validate_model(y, y_pred_all, cv)

# Get aggregate statistics
print(results.aggregate_metrics)
print(results.summary())
```

#### 2. `model_validation.extensions.probabilistic`

Probabilistic forecast validation.

```python
from model_validation.extensions import ProbabilisticValidator

validator = ProbabilisticValidator()

# Validate prediction intervals
results = validator.validate_prediction_intervals(
    y_true, lower_bounds, upper_bounds,
    confidence_level=0.95
)

# Validate quantile forecasts
quantiles = {
    0.1: q10_predictions,
    0.5: median_predictions,
    0.9: q90_predictions
}
results = validator.validate_quantile_forecasts(y_true, quantiles)
```

#### 3. `model_validation.extensions.comparison`

Model comparison tools.

```python
from model_validation.extensions import ModelComparator

comparator = ModelComparator()

# Add models
comparator.add_model("Model1", y_train, train1, y_test, test1)
comparator.add_model("Model2", y_train, train2, y_test, test2)

# Compare
df = comparator.compare_metrics()

# Statistical tests
dm_test = comparator.diebold_mariano_test("Model1", "Model2")
all_tests = comparator.pairwise_dm_tests()

# Get best model
best_name, best_report = comparator.get_best_model(criterion='test_RMSE')
```

#### 4. `model_validation.extensions.backtesting`

Backtesting framework.

```python
from model_validation.extensions import BacktestingFramework

backtester = BacktestingFramework(
    initial_window=100,
    step_size=1,
    forecasting_horizon=10,
    refit_frequency=5
)

# Run backtest
results = backtester.backtest_with_predictions(
    data=time_series,
    all_predictions=predictions
)

# Analyze
df = results.to_dataframe()
stats = results.summary_stats()
```

### Utility Modules

#### 1. `model_validation.utils.sktime_utils`

sktime integration.

```python
from model_validation.utils import (
    convert_to_sktime_format,
    prepare_sktime_train_test
)

# Convert to sktime
y_sktime = convert_to_sktime_format(
    data, freq='M', start_date='2020-01-01'
)

# Train/test split
y_train, y_test = prepare_sktime_train_test(
    data, train_size=0.8, freq='M'
)
```

#### 2. `model_validation.utils.visualization`

Plotting functions.

```python
from model_validation.utils import (
    plot_residuals,
    plot_predictions,
    plot_diagnostics,
    create_validation_dashboard
)

# Individual plots
fig = plot_residuals(residuals, timestamps)
fig = plot_predictions(y_test, predictions)
fig = plot_diagnostics(y_train, train_pred, y_test, test_pred)

# Full dashboard
fig = create_validation_dashboard(
    report, residuals, y_test, predictions
)
```

#### 3. `model_validation.utils.reporting`

Report generation and export.

```python
from model_validation.utils import (
    HTMLReportGenerator,
    export_to_csv,
    export_to_json,
    create_markdown_report
)

# HTML report
generator = HTMLReportGenerator()
html = generator.generate_html_report(report)
with open('report.html', 'w') as f:
    f.write(html)

# CSV/JSON export
export_to_csv(report, 'results.csv')
export_to_json(report, 'results.json')

# Markdown
markdown = create_markdown_report(report)
```
