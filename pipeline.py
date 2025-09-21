"""Pipeline"""
from datetime import timedelta
import pandas as pd
from pathlib import Path

from config import Config
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from alpha_model import AlphaModel
from backtester import Backtester
from utils import setup_logging


class EarningsNLPPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.data_loader = DataLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        self.alpha_model = AlphaModel(config)
        self.backtester = Backtester(config)

    def _determine_split_date(self, df: pd.DataFrame) -> pd.Timestamp:
        dates = pd.to_datetime(df['date']).sort_values()
        split_idx = int(len(dates) * 0.7)
        return dates.iloc[split_idx]

    def run(self):
        transcripts = self.data_loader.load_transcripts()
        features_df = self.feature_extractor.extract_features(transcripts)

        min_date = min(t.date for t in transcripts) - timedelta(days=5)
        max_date = max(t.date for t in transcripts) + timedelta(days=30)
        tickers = {t.ticker for t in transcripts}
        prices_df = self.data_loader.fetch_prices_batch(tickers, min_date, max_date)

        benchmark_df = self.data_loader.fetch_benchmark_prices(min_date, max_date) if self.config.use_market_neutral else None

        labeled_data = self.alpha_model.prepare_labels(features_df, prices_df, benchmark_df, self.config.holding_periods[0])

        split_date = pd.to_datetime(self.config.train_test_split_date) if pd.to_datetime(self.config.train_test_split_date) < labeled_data['date'].max() else self._determine_split_date(labeled_data)

        train_df = labeled_data[labeled_data['date'] < split_date]
        test_df = labeled_data[labeled_data['date'] >= split_date]

        feature_cols = [c for c in train_df.columns if c.startswith('emb_') or c in ['sentiment', 'word_count', 'readability']]
        X_train, y_train = train_df[feature_cols].fillna(0), train_df['label']
        X_test, y_test = test_df[feature_cols].fillna(0), test_df['label']

        val_score = self.alpha_model.train(X_train, y_train)
        predictions, pred_labels = self.alpha_model.predict(X_test)

        return_col = 'relative_return' if self.config.use_market_neutral else 'future_return'
        metrics = self.backtester.evaluate(predictions, pred_labels, y_test, test_df[return_col])

        report_path = Path(self.config.output_dir) / "backtest_report.txt"
        self.backtester.generate_report(metrics, report_path)

        self.alpha_model.save()
        return metrics