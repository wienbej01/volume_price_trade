"""HTML reporting for training results."""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

# Configure logger
logger = logging.getLogger(__name__)


def generate_train_report(
    run_metadata: Dict[str, Any],
    metrics: Dict[str, List[float]],
    feature_names: List[str],
    output_path: str
) -> str:
    """
    Generate a comprehensive training report.

    Args:
        run_metadata: Training run metadata
        metrics: Per-fold metrics dictionary
        feature_names: List of feature names used
        output_path: Path to save the HTML report

    Returns:
        Path to the generated report
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate summary statistics
    summary_stats = _calculate_summary_stats(metrics)

    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Report - {run_metadata['run_id']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .positive {{
            color: green;
        }}
        .negative {{
            color: red;
        }}
        .metric-card {{
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .fold-metrics {{
            margin: 20px 0;
        }}
        .feature-list {{
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <h1>Training Report</h1>
    <p><strong>Run ID:</strong> {run_metadata['run_id']}</p>
    <p><strong>Timestamp:</strong> {run_metadata['timestamp']}</p>
    <p><strong>Config:</strong> {run_metadata.get('config_path', 'N/A')}</p>

    <div class="grid">
        <div class="metric-card">
            <h3>Accuracy</h3>
            <div class="metric-value">{summary_stats['accuracy']['mean']:.3f}</div>
            <div>±{summary_stats['accuracy']['std']:.3f}</div>
        </div>

        <div class="metric-card">
            <h3>F1 Score</h3>
            <div class="metric-value">{summary_stats['f1']['mean']:.3f}</div>
            <div>±{summary_stats['f1']['std']:.3f}</div>
        </div>

        <div class="metric-card">
            <h3>Precision</h3>
            <div class="metric-value">{summary_stats['precision']['mean']:.3f}</div>
            <div>±{summary_stats['precision']['std']:.3f}</div>
        </div>

        <div class="metric-card">
            <h3>Recall</h3>
            <div class="metric-value">{summary_stats['recall']['mean']:.3f}</div>
            <div>±{summary_stats['recall']['std']:.3f}</div>
        </div>
    </div>

    <h2>Summary Statistics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Mean</th>
            <th>Std Dev</th>
            <th>Min</th>
            <th>Max</th>
        </tr>
"""

    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'log_loss']:
        if metric_name in summary_stats:
            stats = summary_stats[metric_name]
            html_content += f"""
        <tr>
            <td>{metric_name.title()}</td>
            <td>{stats['mean']:.4f}</td>
            <td>{stats['std']:.4f}</td>
            <td>{stats['min']:.4f}</td>
            <td>{stats['max']:.4f}</td>
        </tr>
"""

    if 'auc' in summary_stats:
        stats = summary_stats['auc']
        html_content += f"""
        <tr>
            <td>AUC</td>
            <td>{stats['mean']:.4f}</td>
            <td>{stats['std']:.4f}</td>
            <td>{stats['min']:.4f}</td>
            <td>{stats['max']:.4f}</td>
        </tr>
"""

    html_content += """
    </table>

    <h2>Per-Fold Metrics</h2>
    <div class="fold-metrics">
    <table>
        <tr>
            <th>Fold</th>
"""

    # Add metric headers
    for metric_name in metrics.keys():
        html_content += f"<th>{metric_name.title()}</th>"

    html_content += "</tr>"

    # Add fold data
    for fold_idx in range(len(next(iter(metrics.values())))):
        html_content += f"<tr><td>{fold_idx + 1}</td>"
        for metric_name, metric_values in metrics.items():
            html_content += f"<td>{metric_values[fold_idx]:.4f}</td>"
        html_content += "</tr>"

    html_content += """
    </table>
    </div>

    <h2>Dataset Information</h2>
    <table>
        <tr>
            <td>Samples</td>
            <td>{run_metadata['n_samples']}</td>
        </tr>
        <tr>
            <td>Features</td>
            <td>{run_metadata['n_features']}</td>
        </tr>
        <tr>
            <td>CV Folds</td>
            <td>{run_metadata['n_folds']}</td>
        </tr>
    </table>

    <h2>Features Used</h2>
    <div class="feature-list">
    <ul>
"""

    for feature in sorted(feature_names):
        html_content += f"<li>{feature}</li>"

    html_content += """
    </ul>
    </div>

    <h2>Configuration</h2>
    <pre>{json.dumps(run_metadata.get('config', {}), indent=2)}</pre>

    <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
</body>
</html>
"""

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Training report generated at {output_path}")
    return output_path


def _calculate_summary_stats(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate summary statistics for metrics across folds.

    Args:
        metrics: Dictionary of metric lists

    Returns:
        Dictionary with summary stats for each metric
    """
    summary = {}
    for metric_name, values in metrics.items():
        if values:
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    return summary