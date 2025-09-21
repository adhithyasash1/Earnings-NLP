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
        self.logger.info("Loading transcripts...")
        transcripts = self.data_loader.load_transcripts()

        if not transcripts:
            raise ValueError("No transcripts found. Please check transcript_dir path and file format.")

        self.logger.info(f"Loaded {len(transcripts)} transcripts")

        self.logger.info("Extracting features...")
        features_df = self.feature_extractor.extract_features(transcripts)

        if features_df.empty:
            raise ValueError("No features extracted from transcripts")

        self.logger.info(f"Extracted features for {len(features_df)} transcripts")

        # Calculate date range with buffer
        min_date = min(t.date for t in transcripts) - timedelta(days=10)
        max_date = max(t.date for t in transcripts) + timedelta(days=30)
        tickers = {t.ticker for t in transcripts}

        self.logger.info(f"Fetching prices for {len(tickers)} tickers from {min_date.date()} to {max_date.date()}")

        try:
            prices_df = self.data_loader.fetch_prices_batch(tickers, min_date, max_date)
            if prices_df.empty:
                raise ValueError("No price data fetched")
            self.logger.info(f"Fetched {len(prices_df)} price records")
        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}")
            raise

        benchmark_df = None
        if self.config.use_market_neutral:
            try:
                self.logger.info(f"Fetching benchmark data for {self.config.benchmark_ticker}")
                benchmark_df = self.data_loader.fetch_benchmark_prices(min_date, max_date)
                if benchmark_df.empty:
                    self.logger.warning("No benchmark data found, disabling market neutral")
                    self.config.use_market_neutral = False
                else:
                    self.logger.info(f"Fetched {len(benchmark_df)} benchmark records")
            except Exception as e:
                self.logger.warning(f"Error fetching benchmark data: {e}. Disabling market neutral.")
                self.config.use_market_neutral = False

        self.logger.info("Preparing labels...")
        labeled_data = self.alpha_model.prepare_labels(
            features_df, prices_df, benchmark_df, self.config.holding_periods[0]
        )

        if labeled_data.empty:
            # Provide detailed debugging info
            self.logger.error("No labeled data created. Debugging info:")
            self.logger.error(f"Features date range: {features_df['date'].min()} to {features_df['date'].max()}")
            self.logger.error(f"Price data tickers: {sorted(prices_df['ticker'].unique()) if not prices_df.empty else 'None'}")
            self.logger.error(f"Price date range: {prices_df['timestamp'].min()} to {prices_df['timestamp'].max() if not prices_df.empty else 'None'}")
            self.logger.error(f"Features tickers: {sorted(features_df['ticker'].unique())}")

            # Check for ticker mismatches
            feature_tickers = set(features_df['ticker'].unique())
            price_tickers = set(prices_df['ticker'].unique()) if not prices_df.empty else set()
            missing_tickers = feature_tickers - price_tickers
            if missing_tickers:
                self.logger.error(f"Tickers in features but not in prices: {missing_tickers}")

            raise ValueError("No valid data found for analysis. Check ticker symbols and date ranges.")

        self.logger.info(f"Created {len(labeled_data)} labeled samples")

        # Determine split date
        config_split_date = pd.to_datetime(self.config.train_test_split_date)
        if config_split_date < labeled_data['date'].max():
            split_date = config_split_date
        else:
            split_date = self._determine_split_date(labeled_data)
            self.logger.info(f"Using computed split date: {split_date}")

        train_df = labeled_data[labeled_data['date'] < split_date]
        test_df = labeled_data[labeled_data['date'] >= split_date]

        if train_df.empty:
            raise ValueError("No training data available. Check train_test_split_date configuration.")
        if test_df.empty:
            raise ValueError("No test data available. Check train_test_split_date configuration.")

        self.logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        feature_cols = [c for c in train_df.columns
                       if c.startswith('emb_') or c in ['sentiment', 'word_count', 'readability']]

        if not feature_cols:
            raise ValueError("No feature columns found")

        self.logger.info(f"Using {len(feature_cols)} features")

        X_train, y_train = train_df[feature_cols].fillna(0), train_df['label']
        X_test, y_test = test_df[feature_cols].fillna(0), test_df['label']

        self.logger.info("Training model...")
        val_score = self.alpha_model.train(X_train, y_train)
        self.logger.info(f"Validation score: {val_score:.3f}")

        self.logger.info("Making predictions...")
        predictions, pred_labels = self.alpha_model.predict(X_test)

        return_col = 'relative_return' if self.config.use_market_neutral else 'future_return'

        self.logger.info("Evaluating results...")
        metrics = self.backtester.evaluate(predictions, pred_labels, y_test, test_df[return_col])

        report_path = Path(self.config.output_dir) / "backtest_report.txt"
        report = self.backtester.generate_report(metrics, report_path)
        self.logger.info("Backtest report generated")

        print(report)

        self.logger.info("Saving model...")
        self.alpha_model.save()

        return metrics