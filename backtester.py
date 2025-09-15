# ============================================
# backtester.py - Enhanced Performance Evaluation
# ============================================
"""Comprehensive backtesting with portfolio metrics"""
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict

# Import your Config and setup_logging
from config import Config
from utils import setup_logging

class Backtester:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.results_df = None

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray,
                 actuals: pd.Series, returns: pd.Series = None,
                 dates: pd.Series = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        is_classification = self.config.prediction_target == "direction"

        # Create results DataFrame for saving
        self.results_df = pd.DataFrame({
            'date': dates,
            'prediction_raw': predictions,
            'prediction_label': labels,
            'actual_label': actuals,
            'strategy_return_raw': returns
        }).sort_values('date').reset_index(drop=True)

        # Classification metrics
        if is_classification:
            metrics['accuracy'] = accuracy_score(actuals, labels)
            metrics['precision'] = precision_score(actuals, labels, zero_division=0)
            metrics['recall'] = recall_score(actuals, labels, zero_division=0)
            metrics['f1'] = f1_score(actuals, labels, zero_division=0)
            # Store for plotting
            self.cm = confusion_matrix(actuals, labels)

        # Trading metrics
        if returns is not None:
            ic, ic_pvalue = spearmanr(predictions, returns)
            metrics['information_coefficient'] = ic
            metrics['ic_pvalue'] = ic_pvalue

            transaction_cost = 0.001  # 10 bps

            if is_classification:
                # Long/short based on binary signal
                strategy_returns = returns * (2 * labels - 1) - transaction_cost
            else:
                # For regression, we need a signal.
                # Simple example: long if predicted > 0, short if < 0
                signal = np.sign(labels)
                strategy_returns = returns * signal - transaction_cost

            self.results_df['strategy_return_net'] = strategy_returns

            metrics['total_return'] = strategy_returns.sum()
            metrics['mean_return'] = strategy_returns.mean()
            metrics['volatility'] = strategy_returns.std()
            if metrics['volatility'] > 0:
                metrics['event_sharpe'] = strategy_returns.mean() / strategy_returns.std()
            else:
                metrics['event_sharpe'] = 0.0

            metrics['win_rate'] = (strategy_returns > 0).mean()
            metrics['max_drawdown'] = self._calculate_max_drawdown(strategy_returns)

            self.strategy_returns = strategy_returns

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def generate_report(self, metrics: Dict, save_path: str = None) -> str:
        report = "=" * 50 + "\nBACKTEST RESULTS\n" + "=" * 50 + "\n\n"

        if self.config.prediction_target == "direction":
            report += "Classification Metrics:\n"
            report += f"  Accuracy:   {metrics.get('accuracy', 0):.3f}\n"
            report += f"  Precision:  {metrics.get('precision', 0):.3f}\n"
            report += f"  Recall:     {metrics.get('recall', 0):.3f}\n"
            report += f"  F1 Score:   {metrics.get('f1', 0):.3f}\n\n"

        report += "Alpha Metrics:\n"
        report += f"  Information Coefficient: {metrics.get('information_coefficient', 0):.3f}\n"
        report += f"  IC P-value: {metrics.get('ic_pvalue', 1):.4f}\n\n"

        report += "Trading Performance:\n"
        report += f"  Total Return:     {metrics.get('total_return', 0):.2%}\n"
        report += f"  Event Sharpe:     {metrics.get('event_sharpe', 0):.2f}\n"
        report += f"  Win Rate:         {metrics.get('win_rate', 0):.2%}\n"
        report += f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.2%}\n"
        report += f"  Mean Return:      {metrics.get('mean_return', 0):.3%}\n"
        report += f"  Volatility:       {metrics.get('volatility', 0):.3%}\n"

        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
        return report

    def _plot_confusion_matrix(self, save_path: str):
        """Plots the confusion matrix."""
        if not hasattr(self, 'cm'):
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(self.cm, display_labels=[0, 1])
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix (Test Set)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_results(self, save_dir: str = None):
        """Generate and save performance plots"""
        if save_dir is None:
            save_dir = self.config.output_dir
        Path(save_dir).mkdir(exist_ok=True)

        if hasattr(self, 'strategy_returns') and not self.strategy_returns.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            cumulative = (1 + self.strategy_returns).cumprod()
            ax1.plot(cumulative.index, cumulative.values)
            ax1.set_title('Cumulative Strategy Returns (Event-Based)')
            ax1.set_ylabel('Cumulative Return')
            ax1.grid(True)

            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown %')
            ax2.set_xlabel('Trade Number')
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(Path(save_dir) / 'performance.png')
            plt.close()

        # Plot confusion matrix if it's a classification task
        if self.config.prediction_target == "direction":
            self._plot_confusion_matrix(Path(save_dir) / 'confusion_matrix.png')

    def save_full_results(self, save_dir: str = None):
        """Saves the full per-prediction results to a parquet file."""
        if save_dir is None:
            save_dir = self.config.output_dir
        Path(save_dir).mkdir(exist_ok=True)

        if self.results_df is not None:
            save_path = Path(save_dir) / "full_backtest_results.parquet"
            self.results_df.to_parquet(save_path)
            self.logger.info(f"Full backtest results saved to {save_path}")