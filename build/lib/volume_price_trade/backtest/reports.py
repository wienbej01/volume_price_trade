"""HTML/MD reporting."""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
from pathlib import Path
import uuid

from .metrics import compute_all_metrics

# Configure logger
logger = logging.getLogger(__name__)


def make_report(
    bt_result: Dict[str, Any],
    path: str,
    format: str = "html",
    include_plots: bool = True
) -> str:
    """
    Generate a backtest report.
    
    Args:
        bt_result: Backtest result dictionary
        path: Output path for the report
        format: Report format ('html' or 'markdown')
        include_plots: Whether to include plots (for HTML)
        
    Returns:
        Path to the generated report
    """
    # Create output directory if it doesn't exist
    output_dir = Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    metrics = compute_all_metrics(
        trades=bt_result['trades'],
        equity_curve=bt_result['equity_curve']
    )
    
    # Generate report based on format
    if format.lower() == 'html':
        return _generate_html_report(bt_result, metrics, path, include_plots)
    elif format.lower() == 'markdown':
        return _generate_markdown_report(bt_result, metrics, path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _generate_html_report(
    bt_result: Dict[str, Any],
    metrics: Dict[str, Any],
    path: str,
    include_plots: bool
) -> str:
    """Generate HTML report."""
    # Generate plots if requested
    plot_paths = {}
    if include_plots:
        plot_paths = _generate_plots(bt_result, path)
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
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
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
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
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
    
    <div class="grid">
        <div class="metric-card">
            <h3>Total Return</h3>
            <div class="metric-value {'positive' if metrics['equity_metrics']['total_return'] > 0 else 'negative'}">
                {metrics['equity_metrics']['total_return']:.2%}
            </div>
        </div>
        
        <div class="metric-card">
            <h3>Sharpe Ratio</h3>
            <div class="metric-value">
                {metrics['equity_metrics']['sharpe_ratio']:.2f}
            </div>
        </div>
        
        <div class="metric-card">
            <h3>Max Drawdown</h3>
            <div class="metric-value negative">
                {metrics['equity_metrics']['max_drawdown']:.2%}
            </div>
        </div>
        
        <div class="metric-card">
            <h3>Win Rate</h3>
            <div class="metric-value">
                {metrics['trade_stats']['win_rate']:.2%}
            </div>
        </div>
    </div>
    
    <h2>Trade Statistics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Total Trades</td>
            <td>{metrics['trade_stats']['total_trades']}</td>
        </tr>
        <tr>
            <td>Winning Trades</td>
            <td>{metrics['trade_stats']['winning_trades']}</td>
        </tr>
        <tr>
            <td>Losing Trades</td>
            <td>{metrics['trade_stats']['losing_trades']}</td>
        </tr>
        <tr>
            <td>Win Rate</td>
            <td>{metrics['trade_stats']['win_rate']:.2%}</td>
        </tr>
        <tr>
            <td>Profit Factor</td>
            <td>{metrics['trade_stats']['profit_factor']:.2f}</td>
        </tr>
        <tr>
            <td>Expectancy</td>
            <td>{metrics['trade_stats']['expectancy']:.2f}</td>
        </tr>
        <tr>
            <td>Average Win</td>
            <td class="positive">${metrics['trade_stats']['avg_win']:.2f}</td>
        </tr>
        <tr>
            <td>Average Loss</td>
            <td class="negative">${metrics['trade_stats']['avg_loss']:.2f}</td>
        </tr>
        <tr>
            <td>Average R Multiple</td>
            <td>{metrics['trade_stats']['avg_r_multiple']:.2f}</td>
        </tr>
        <tr>
            <td>Average Hold Time</td>
            <td>{metrics['trade_stats']['avg_hold_minutes']:.0f} minutes</td>
        </tr>
        <tr>
            <td>Max Win</td>
            <td class="positive">${metrics['trade_stats']['max_win']:.2f}</td>
        </tr>
        <tr>
            <td>Max Loss</td>
            <td class="negative">${metrics['trade_stats']['max_loss']:.2f}</td>
        </tr>
        <tr>
            <td>Total P&L</td>
            <td class="{'positive' if metrics['trade_stats']['total_pnl'] > 0 else 'negative'}">
                ${metrics['trade_stats']['total_pnl']:.2f}
            </td>
        </tr>
    </table>
    
    <h2>Equity Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Total Return</td>
            <td class="{'positive' if metrics['equity_metrics']['total_return'] > 0 else 'negative'}">
                {metrics['equity_metrics']['total_return']:.2%}
            </td>
        </tr>
        <tr>
            <td>Annualized Return</td>
            <td class="{'positive' if metrics['equity_metrics']['annualized_return'] > 0 else 'negative'}">
                {metrics['equity_metrics']['annualized_return']:.2%}
            </td>
        </tr>
        <tr>
            <td>Volatility</td>
            <td>{metrics['equity_metrics']['volatility']:.2%}</td>
        </tr>
        <tr>
            <td>Sharpe Ratio</td>
            <td>{metrics['equity_metrics']['sharpe_ratio']:.2f}</td>
        </tr>
        <tr>
            <td>Max Drawdown</td>
            <td class="negative">{metrics['equity_metrics']['max_drawdown']:.2%}</td>
        </tr>
        <tr>
            <td>Max Drawdown Duration</td>
            <td>{metrics['equity_metrics']['max_drawdown_duration']} periods</td>
        </tr>
        <tr>
            <td>Calmar Ratio</td>
            <td>{metrics['equity_metrics']['calmar_ratio']:.2f}</td>
        </tr>
    </table>
    
    {_generate_ticker_table_html(metrics['ticker_stats'])}
    
    {_generate_time_of_day_table_html(metrics['time_of_day_stats'])}
    
    {_generate_monthly_table_html(metrics['monthly_stats'])}
    
    {_generate_plot_section_html(plot_paths) if include_plots else ""}
    
    <h2>Configuration</h2>
    <pre>{json.dumps(bt_result['config'], indent=2)}</pre>
    
    <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
</body>
</html>
    """
    
    # Write to file
    with open(path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at {path}")
    return path


def _generate_markdown_report(
    bt_result: Dict[str, Any],
    metrics: Dict[str, Any],
    path: str
) -> str:
    """Generate Markdown report."""
    md_content = f"""# Backtest Report

## Summary

| Metric | Value |
|--------|-------|
| Total Return | {metrics['equity_metrics']['total_return']:.2%} |
| Sharpe Ratio | {metrics['equity_metrics']['sharpe_ratio']:.2f} |
| Max Drawdown | {metrics['equity_metrics']['max_drawdown']:.2%} |
| Win Rate | {metrics['trade_stats']['win_rate']:.2%} |

## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {metrics['trade_stats']['total_trades']} |
| Winning Trades | {metrics['trade_stats']['winning_trades']} |
| Losing Trades | {metrics['trade_stats']['losing_trades']} |
| Win Rate | {metrics['trade_stats']['win_rate']:.2%} |
| Profit Factor | {metrics['trade_stats']['profit_factor']:.2f} |
| Expectancy | {metrics['trade_stats']['expectancy']:.2f} |
| Average Win | ${metrics['trade_stats']['avg_win']:.2f} |
| Average Loss | ${metrics['trade_stats']['avg_loss']:.2f} |
| Average R Multiple | {metrics['trade_stats']['avg_r_multiple']:.2f} |
| Average Hold Time | {metrics['trade_stats']['avg_hold_minutes']:.0f} minutes |
| Max Win | ${metrics['trade_stats']['max_win']:.2f} |
| Max Loss | ${metrics['trade_stats']['max_loss']:.2f} |
| Total P&L | ${metrics['trade_stats']['total_pnl']:.2f} |

## Equity Metrics

| Metric | Value |
|--------|-------|
| Total Return | {metrics['equity_metrics']['total_return']:.2%} |
| Annualized Return | {metrics['equity_metrics']['annualized_return']:.2%} |
| Volatility | {metrics['equity_metrics']['volatility']:.2%} |
| Sharpe Ratio | {metrics['equity_metrics']['sharpe_ratio']:.2f} |
| Max Drawdown | {metrics['equity_metrics']['max_drawdown']:.2%} |
| Max Drawdown Duration | {metrics['equity_metrics']['max_drawdown_duration']} periods |
| Calmar Ratio | {metrics['equity_metrics']['calmar_ratio']:.2f} |

{_generate_ticker_table_md(metrics['ticker_stats'])}

{_generate_time_of_day_table_md(metrics['time_of_day_stats'])}

{_generate_monthly_table_md(metrics['monthly_stats'])}

## Configuration

```json
{json.dumps(bt_result['config'], indent=2)}
```

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
    """
    
    # Write to file
    with open(path, 'w') as f:
        f.write(md_content)
    
    logger.info(f"Markdown report generated at {path}")
    return path


def _generate_plots(bt_result: Dict[str, Any], report_path: str) -> Dict[str, str]:
    """Generate plots for the report."""
    plot_paths = {}
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Create plots directory
        plots_dir = Path(report_path).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Equity curve plot
        plt.figure(figsize=(12, 6))
        equity_curve = bt_result['equity_curve']
        plt.plot(equity_curve['timestamp'], equity_curve['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        
        # Format x-axis
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        equity_curve_path = plots_dir / "equity_curve.png"
        plt.savefig(equity_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['equity_curve'] = str(equity_curve_path)
        
        # Drawdown plot
        plt.figure(figsize=(12, 6))
        equity_curve = bt_result['equity_curve'].copy()
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak'] * 100
        
        plt.fill_between(equity_curve['timestamp'], 0, equity_curve['drawdown'], color='red', alpha=0.3)
        plt.plot(equity_curve['timestamp'], equity_curve['drawdown'], color='red')
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Format x-axis
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        drawdown_path = plots_dir / "drawdown.png"
        plt.savefig(drawdown_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['drawdown'] = str(drawdown_path)
        
        # Monthly returns heatmap
        if bt_result['trades']:
            trades_df = pd.DataFrame(bt_result['trades'])
            trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
            monthly_returns = trades_df.groupby('month')['pnl'].sum()

            # Reshape to calendar format
            monthly_returns.index = monthly_returns.index.to_timestamp()  # type: ignore
            monthly_returns = monthly_returns.resample('M').sum()

            # Create year and month columns
            monthly_returns_df = monthly_returns.reset_index()
            monthly_returns_df['year'] = monthly_returns_df['entry_time'].dt.year
            monthly_returns_df['month'] = monthly_returns_df['entry_time'].dt.month

            # Pivot to create heatmap data
            heatmap_data = monthly_returns_df.pivot(index='year', columns='month', values='pnl')
            
            plt.figure(figsize=(12, 8))
            plt.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto')
            plt.colorbar(label='Monthly P&L')
            plt.title('Monthly Returns Heatmap')
            plt.xlabel('Month')
            plt.ylabel('Year')
            
            # Set ticks
            plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.yticks(range(len(heatmap_data.index)), heatmap_data.index.astype(str).tolist())
            
            # Add text annotations
            for i in range(len(heatmap_data.index)):
                for j in range(12):
                    value = heatmap_data.iloc[i, j]
                    if not pd.isna(value):
                        plt.text(j, i, f'{value:.0f}', ha='center', va='center',
                                color='black' if abs(float(value)) < heatmap_data.values.max() / 2 else 'white')  # type: ignore
            
            monthly_heatmap_path = plots_dir / "monthly_heatmap.png"
            plt.savefig(monthly_heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['monthly_heatmap'] = str(monthly_heatmap_path)
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return plot_paths


def _generate_ticker_table_html(ticker_stats: Dict[str, Dict[str, Any]]) -> str:
    """Generate HTML table for ticker statistics."""
    if not ticker_stats:
        return ""
    
    html = """
    <h2>Ticker Statistics</h2>
    <table>
        <tr>
            <th>Ticker</th>
            <th>Total Trades</th>
            <th>Win Rate</th>
            <th>Profit Factor</th>
            <th>Total P&L</th>
        </tr>
    """
    
    for ticker, stats in ticker_stats.items():
        html += f"""
        <tr>
            <td>{ticker}</td>
            <td>{stats['total_trades']}</td>
            <td>{stats['win_rate']:.2%}</td>
            <td>{stats['profit_factor']:.2f}</td>
            <td class="{'positive' if stats['total_pnl'] > 0 else 'negative'}">${stats['total_pnl']:.2f}</td>
        </tr>
        """
    
    html += "</table>"
    return html


def _generate_time_of_day_table_html(time_stats: Dict[str, Dict[str, Any]]) -> str:
    """Generate HTML table for time of day statistics."""
    if not time_stats:
        return ""
    
    html = """
    <h2>Time of Day Statistics</h2>
    <table>
        <tr>
            <th>Time Period</th>
            <th>Total Trades</th>
            <th>Win Rate</th>
            <th>Profit Factor</th>
            <th>Total P&L</th>
        </tr>
    """
    
    for period, stats in time_stats.items():
        html += f"""
        <tr>
            <td>{period}</td>
            <td>{stats['total_trades']}</td>
            <td>{stats['win_rate']:.2%}</td>
            <td>{stats['profit_factor']:.2f}</td>
            <td class="{'positive' if stats['total_pnl'] > 0 else 'negative'}">${stats['total_pnl']:.2f}</td>
        </tr>
        """
    
    html += "</table>"
    return html


def _generate_monthly_table_html(monthly_stats: Dict[str, Dict[str, Any]]) -> str:
    """Generate HTML table for monthly statistics."""
    if not monthly_stats:
        return ""
    
    html = """
    <h2>Monthly Statistics</h2>
    <table>
        <tr>
            <th>Month</th>
            <th>Total Trades</th>
            <th>Win Rate</th>
            <th>Profit Factor</th>
            <th>Total P&L</th>
        </tr>
    """
    
    for month, stats in monthly_stats.items():
        html += f"""
        <tr>
            <td>{month}</td>
            <td>{stats['total_trades']}</td>
            <td>{stats['win_rate']:.2%}</td>
            <td>{stats['profit_factor']:.2f}</td>
            <td class="{'positive' if stats['total_pnl'] > 0 else 'negative'}">${stats['total_pnl']:.2f}</td>
        </tr>
        """
    
    html += "</table>"
    return html


def _generate_plot_section_html(plot_paths: Dict[str, str]) -> str:
    """Generate HTML section for plots."""
    if not plot_paths:
        return ""
    
    html = "<h2>Plots</h2>"
    
    if 'equity_curve' in plot_paths:
        html += f"""
        <div class="plot">
            <h3>Equity Curve</h3>
            <img src="{Path(plot_paths['equity_curve']).name}" alt="Equity Curve">
        </div>
        """
    
    if 'drawdown' in plot_paths:
        html += f"""
        <div class="plot">
            <h3>Drawdown</h3>
            <img src="{Path(plot_paths['drawdown']).name}" alt="Drawdown">
        </div>
        """
    
    if 'monthly_heatmap' in plot_paths:
        html += f"""
        <div class="plot">
            <h3>Monthly Returns Heatmap</h3>
            <img src="{Path(plot_paths['monthly_heatmap']).name}" alt="Monthly Returns Heatmap">
        </div>
        """
    
    return html


def _generate_ticker_table_md(ticker_stats: Dict[str, Dict[str, Any]]) -> str:
    """Generate Markdown table for ticker statistics."""
    if not ticker_stats:
        return ""
    
    md = "## Ticker Statistics\n\n"
    md += "| Ticker | Total Trades | Win Rate | Profit Factor | Total P&L |\n"
    md += "|--------|--------------|----------|---------------|----------|\n"
    
    for ticker, stats in ticker_stats.items():
        md += f"| {ticker} | {stats['total_trades']} | {stats['win_rate']:.2%} | {stats['profit_factor']:.2f} | ${stats['total_pnl']:.2f} |\n"
    
    return md


def _generate_time_of_day_table_md(time_stats: Dict[str, Dict[str, Any]]) -> str:
    """Generate Markdown table for time of day statistics."""
    if not time_stats:
        return ""
    
    md = "## Time of Day Statistics\n\n"
    md += "| Time Period | Total Trades | Win Rate | Profit Factor | Total P&L |\n"
    md += "|-------------|--------------|----------|---------------|----------|\n"
    
    for period, stats in time_stats.items():
        md += f"| {period} | {stats['total_trades']} | {stats['win_rate']:.2%} | {stats['profit_factor']:.2f} | ${stats['total_pnl']:.2f} |\n"
    
    return md


def _generate_monthly_table_md(monthly_stats: Dict[str, Dict[str, Any]]) -> str:
    """Generate Markdown table for monthly statistics."""
    if not monthly_stats:
        return ""
    
    md = "## Monthly Statistics\n\n"
    md += "| Month | Total Trades | Win Rate | Profit Factor | Total P&L |\n"
    md += "|-------|--------------|----------|---------------|----------|\n"
    
    for month, stats in monthly_stats.items():
        md += f"| {month} | {stats['total_trades']} | {stats['win_rate']:.2%} | {stats['profit_factor']:.2f} | ${stats['total_pnl']:.2f} |\n"
    
    return md
