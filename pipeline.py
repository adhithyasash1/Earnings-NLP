# ============================================
# pipeline.py - Updated Pipeline
# ============================================
"""End-to-end pipeline with proper out-of-sample testing"""
import logging
from datetime import timedelta
import pandas as pd
from pathlib import Path

# Import your classes
from config import Config
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from alpha_model import AlphaModel
from backtester import Backtester
from utils import setup_logging


class EarningsNLPPipeline:
    def __init__(self, config: Config):
        self.config = config

        # --- NEW: Set correct scoring metric ---
        if self.config.prediction_target == "direction" and self.config.tuner_scoring_metric not in ["accuracy", "f1",
                                                                                                     "precision",
                                                                                                     "recall",
                                                                                                     "roc_auc"]:
            config.tuner_scoring_metric = "accuracy"
        elif self.config.prediction_target == "returns" and self.config.tuner_scoring_metric not in [
            "neg_mean_squared_error", "r2"]:
            config.tuner_scoring_metric = "neg_mean_squared_error"
        # ---

        self.config.setup_directories()
        self.logger = setup_logging(config.log_level)

        self.data_loader = DataLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        self.alpha_model = AlphaModel(config)
        self.backtester = Backtester(config)

    def run(self):
        """Execute complete pipeline with proper train/test split"""
        self.logger.info("Starting Earnings NLP Pipeline...")
        self.logger.info(f"Prediction Target: {self.config.prediction_target}")
        self.logger.info(f"Model: {self.config.alpha_model}")

        # ... (Steps 1-7 are unchanged) ...
        # 1. Load transcripts
        self.logger.info("Loading transcripts...")
        transcripts = self.data_loader.load_transcripts()

        # 2. Extract features (cached)
        self.logger.info("Extracting NLP features...")
        features_df = self.feature_extractor.extract_features(transcripts)

        # 3. Get date range for price data
        min_date = min(t.date for t in transcripts) - timedelta(days=1)
        max_date = max(t.date for t in transcripts) + timedelta(days=max(self.config.holding_periods) + 30)
        tickers = set(t.ticker for t in transcripts)

        # 4. Batch fetch all price data (cached)
        self.logger.info("Fetching price data (batch)...")
        prices_df = self.data_loader.fetch_prices_batch(tickers, min_date, max_date)

        # 5. Fetch benchmark prices if using market-neutral
        benchmark_df = None
        if self.config.use_market_neutral:
            self.logger.info("Fetching benchmark prices...")
            benchmark_df = self.data_loader.fetch_benchmark_prices(min_date, max_date)

        # 6. Prepare labels
        self.logger.info("Preparing labels...")
        labeled_data = self.alpha_model.prepare_labels(
            features_df, prices_df, benchmark_df,
            holding_period=self.config.holding_periods[0]
        )

        # 7. CRITICAL: Train/Test Split
        split_date = pd.to_datetime(self.config.train_test_split_date)
        labeled_data = labeled_data.sort_values('date')
        train_df = labeled_data[labeled_data['date'] < split_date]
        test_df = labeled_data[labeled_data['date'] >= split_date]
        self.logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        if len(test_df) == 0:
            self.logger.error("No test samples! Adjust split_date in config.")
            return {}

        # 8. Train model (on training data only)
        emb_cols = [c for c in train_df.columns if c.startswith('emb_')]
        linguistic_cols = [
            'sentiment', 'sentiment_remarks', 'sentiment_qa',
            'word_count', 'readability'
        ]
        all_feature_cols = emb_cols + linguistic_cols
        feature_cols = [
            c for c in all_feature_cols
            if c in train_df.columns and not train_df[c].isnull().all()
        ]
        self.logger.info(f"Training on {len(feature_cols)} features.")

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label']

        self.logger.info("Training alpha model...")
        val_score = self.alpha_model.train(X_train, y_train)
        self.logger.info(f"Training validation score: {val_score:.3f}")

        # 9. Out-of-sample evaluation
        self.logger.info("Evaluating on out-of-sample test set...")
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['label']

        predictions, pred_labels = self.alpha_model.predict(X_test)

        # 10. Calculate metrics
        metrics = self.backtester.evaluate(
            predictions,
            pred_labels,
            y_test,
            test_df.get('relative_return' if self.config.use_market_neutral else 'future_return'),
            test_df['date']
        )

        # 11. Generate and save report
        report_path = Path(self.config.output_dir) / "backtest_report.txt"
        report = self.backtester.generate_report(metrics, report_path)
        self.logger.info("\n" + report)

        # 12. Generate plots
        self.backtester.plot_results()

        # --- NEW: Save full results ---
        self.backtester.save_full_results()

        # 13. Save model
        self.alpha_model.save()

        return metrics