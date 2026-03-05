"""
Core result containers
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class ValidationResult:
    test_name: str
    statistic: float
    p_value: Optional[float] = None
    critical_values: Optional[Dict[str, float]] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL" if self.passed is not None else "INFO"
        p_val_str = f"{self.p_value:.4f}" if self.p_value is not None else "N/A"
        return f"{status} | {self.test_name}: stat={self.statistic:.4f}, p={p_val_str}"


@dataclass
class ValidationReport:
    model_name: str
    results: List[ValidationResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: ValidationResult):
        self.results.append(result)
    
    def add_metric(self, name: str, value: float):
        self.metrics[name] = value
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def summary(self) -> pd.DataFrame:
        data = []
        for result in self.results:
            row = {
                'Test': result.test_name,
                'Statistic': result.statistic,
                'P-Value': result.p_value,
                'Passed': result.passed,
            }
            row.update(result.metadata)
            data.append(row)
        return pd.DataFrame(data)
    
    def get_failed_tests(self) -> List[ValidationResult]:
        return [r for r in self.results if r.passed is False]
    
    def get_passed_tests(self) -> List[ValidationResult]:
        return [r for r in self.results if r.passed is True]
    
    def __repr__(self) -> str:
        n_tests = len(self.results)
        n_passed = sum(1 for r in self.results if r.passed is True)
        n_failed = sum(1 for r in self.results if r.passed is False)
        return (f"ValidationReport(model={self.model_name}, "
                f"tests={n_tests}, passed={n_passed}, failed={n_failed})")
