"""Backtesting with metrics"""
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path

from config import Config
from utils import setup_logging


class Backtester:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray,
                 actuals: pd.Series, returns: pd.Series) -> Dict:
        metrics = {}
        is_classification = self.config.prediction_target == "direction"

        if is_classification:
            metrics['accuracy'] = accuracy_score(actuals, labels)
            metrics['precision'] = precision_score(actuals, labels, zero_division=0)
            metrics['recall'] = recall_score(actuals, labels, zero_division=0)
            metrics['f1'] = f1_score(actuals, labels, zero_division=0)

        ic, ic_pvalue = spearmanr(predictions, returns)
        metrics['information_coefficient'] = ic
        metrics['ic_pvalue'] = ic_pvalue

        transaction_cost = 0.001
        if is_classification:
            strategy_returns = returns * (2 * labels - 1) - transaction_cost
        else:
            signal = np.clip(predictions / np.abs(predictions).max(), -1, 1)
            strategy_returns = returns * signal - transaction_cost

        strategy_returns = np.clip(strategy_returns, -0.5, 0.5)  # Cap extreme

        metrics['total_return'] = strategy_returns.sum()
        metrics['mean_return'] = strategy_returns.mean()
        metrics['volatility'] = strategy_returns.std()
        metrics['event_sharpe'] = metrics['mean_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        metrics['win_rate'] = (strategy_returns > 0).mean()
        metrics['max_drawdown'] = self._calculate_max_drawdown(strategy_returns)

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def generate_report(self, metrics: Dict, save_path: str = None) -> str:
        report = "=" * 50 + "\nBACKTEST RESULTS\n" + "=" * 50 + "\n\n"
        if self.config.prediction_target == "direction":
            report += f"Accuracy: {metrics['accuracy']:.3f}\nPrecision: {metrics['precision']:.3f}\nRecall: {metrics['recall']:.3f}\nF1: {metrics['f1']:.3f}\n\n"
        report += f"IC: {metrics['information_coefficient']:.3f} (p={metrics['ic_pvalue']:.4f})\n\n"
        report += f"Total Return: {metrics['total_return']:.2%}\nEvent Sharpe: {metrics['event_sharpe']:.2f}\nWin Rate: {metrics['win_rate']:.2%}\nMax Drawdown: {metrics['max_drawdown']:.2%}\n"
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        return report