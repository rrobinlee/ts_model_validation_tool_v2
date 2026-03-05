"""
Reporting and export utilities.
"""

import json
import pandas as pd
from typing import Optional
from ..core.results import ValidationReport


class HTMLReportGenerator:
    @staticmethod
    def generate_html_report(
        report: ValidationReport,
        include_plots: bool = False
    ) -> str:
        """
        Parameters:
        -----------
        report : ValidationReport
            Validation report
        include_plots : bool
            Whether to include plot placeholders
        Returns:
        --------
        str
            HTML content
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Validation Report - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            color: white;
            min-width: 150px;
        }}
        .metric-name {{
            font-size: 14px;
            font-weight: 300;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .warning-box {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .warning-box h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .warning-box ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .info-box {{
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Validation Report: {model_name}</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        
        <h2>Performance Metrics</h2>
        <div class="metrics-container">
            {metrics_html}
        </div>
        
        <h2>Diagnostic Tests</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Statistic</th>
                    <th>P-Value</th>
                    <th>Status</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
                {tests_html}
            </tbody>
        </table>
        
        {warnings_section}
        
        <div class="info-box">
            <h3>Report Information</h3>
            <p><strong>Total Tests:</strong> {n_tests}</p>
            <p><strong>Tests Passed:</strong> {n_passed}</p>
            <p><strong>Tests Failed:</strong> {n_failed}</p>
        </div>
    </div>
</body>
</html>
"""
        metrics_html = ""
        key_metrics = {
            'test_RMSE': 'RMSE',
            'test_MAE': 'MAE',
            'test_MAPE': 'MAPE (%)',
            'test_R2': 'R2',
            'test_Directional_Accuracy': 'Dir. Accuracy (%)'
        }
        
        for metric_key, metric_label in key_metrics.items():
            if metric_key in report.metrics:
                value = report.metrics[metric_key]
                metrics_html += f'''
                <div class="metric">
                    <div class="metric-name">{metric_label}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
                '''
        tests_html = ""
        for result in report.results:
            status_class = "pass" if result.passed else "fail"
            status_text = "PASS" if result.passed else "FAIL"
            p_val_text = f"{result.p_value:.4f}" if result.p_value is not None else "N/A"
            interpretation = result.metadata.get('interpretation', '')
            
            tests_html += f'''
            <tr>
                <td>{result.test_name}</td>
                <td>{result.statistic:.4f}</td>
                <td>{p_val_text}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{interpretation}</td>
            </tr>
            '''
        if report.warnings:
            warnings_html = "<li>" + "</li><li>".join(report.warnings) + "</li>"
            warnings_section = f'''
            <div class="warning-box">
                <h3>Warnings</h3>
                <ul>
                    {warnings_html}
                </ul>
            </div>
            '''
        else:
            warnings_section = '''
            <div class="info-box">
                <h3>No Warnings</h3>
                <p>All validation checks passed successfully!</p>
            </div>
            '''
        html = html_template.format(
            model_name=report.model_name,
            timestamp=report.timestamp,
            metrics_html=metrics_html,
            tests_html=tests_html,
            warnings_section=warnings_section,
            n_tests=len(report.results),
            n_passed=len(report.get_passed_tests()),
            n_failed=len(report.get_failed_tests())
        )
        return html


def export_to_csv(
    report: ValidationReport,
    filepath: str,
    include_metadata: bool = True
):
    """
    Parameters:
    -----------
    report : ValidationReport
        Validation report
    filepath : str
        Output file path
    include_metadata : bool
        Whether to include metadata columns
    """
    df = report.summary()
    
    if not include_metadata:
        # Keep only main columns
        main_cols = ['Test', 'Statistic', 'P-Value', 'Passed']
        df = df[[c for c in main_cols if c in df.columns]]
    
    df.to_csv(filepath, index=False)


def export_to_json(
    report: ValidationReport,
    filepath: str,
    pretty: bool = True
):
    """
    Parameters:
    -----------
    report : ValidationReport
        Validation report
    filepath : str
        Output file path
    pretty : bool
        Whether to pretty-print JSON
    """
    data = {
        'model_name': report.model_name,
        'timestamp': str(report.timestamp),
        'metrics': {
            k: float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in report.metrics.items()
        },
        'tests': [
            {
                'name': r.test_name,
                'statistic': float(r.statistic) if not pd.isna(r.statistic) else None,
                'p_value': float(r.p_value) if r.p_value is not None else None,
                'passed': r.passed,
                'metadata': r.metadata
            }
            for r in report.results
        ],
        'warnings': report.warnings,
        'summary': {
            'total_tests': len(report.results),
            'passed': len(report.get_passed_tests()),
            'failed': len(report.get_failed_tests())
        }
    }
    indent = 2 if pretty else None
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def create_markdown_report(report: ValidationReport) -> str:
    """
    Parameters:
    -----------
    report : ValidationReport
        Validation report
    Returns:
    --------
    str
        Markdown formatted report
    """
    md = f"# Model Validation Report: {report.model_name}\n\n"
    md += f"**Generated:** {report.timestamp}\n\n"
    md += "## Performance Metrics\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    for name, value in sorted(report.metrics.items()):
        if isinstance(value, (int, float)):
            md += f"| {name} | {value:.4f} |\n"
    md += "\n"
    md += "## Diagnostic Tests\n\n"
    md += "| Test | Statistic | P-Value | Status |\n"
    md += "|------|-----------|---------|--------|\n"
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        p_val = f"{result.p_value:.4f}" if result.p_value is not None else "N/A"
        md += f"| {result.test_name} | {result.statistic:.4f} | {p_val} | {status} |\n"
    md += "\n"
    if report.warnings:
        md += "## Warnings\n\n"
        for warning in report.warnings:
            md += f"- {warning}\n"
        md += "\n"
    md += "## Summary\n\n"
    md += f"- Total Tests: {len(report.results)}\n"
    md += f"- Passed: {len(report.get_passed_tests())}\n"
    md += f"- Failed: {len(report.get_failed_tests())}\n"
    
    return md
