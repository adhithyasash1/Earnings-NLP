# ============================================
# pipeline.py - ROBUST PIPELINE WITH IMPROVED VALIDATION
# ============================================
"""End-to-end pipeline with robust validation and leakage prevention"""
import logging
from datetime import timedelta
import pandas as pd
import numpy as np
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

        # Set correct scoring metric
        if self.config.prediction_target == "direction" and self.config.tuner_scoring_metric not in ["accuracy", "f1",
                                                                                                     "precision",
                                                                                                     "recall",
                                                                                                     "roc_auc"]:
            config.tuner_scoring_metric = "accuracy"
        elif self.config.prediction_target == "returns" and self.config.tuner_scoring_metric not in [
            "neg_mean_squared_error", "r2"]:
            config.tuner_scoring_metric = "neg_mean_squared_error"

        self.config.setup_directories()
        self.logger = setup_logging(config.log_level)

        self.data_loader = DataLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        self.alpha_model = AlphaModel(config)
        self.backtester = Backtester(config)

    def _determine_robust_split_date(self, features_df: pd.DataFrame) -> str:
        """
        Determine split date ensuring adequate test samples and temporal separation
        """
        dates = pd.to_datetime(features_df['date']).sort_values()
        min_date = dates.min()
        max_date = dates.max()

        total_samples = len(features_df)
        self.logger.info(f"Data range: {min_date.date()} to {max_date.date()} ({total_samples} samples)")

        # Ensure minimum 30 test samples or 20% of data, whichever is smaller
        min_test_samples = min(30, max(10, int(total_samples * 0.2)))

        # Sort by date and take the split point
        sorted_dates = features_df.sort_values('date')['date']
        split_idx = len(sorted_dates) - min_test_samples
        split_date = sorted_dates.iloc[split_idx]

        self.logger.info(f"Target test samples: {min_test_samples}")
        self.logger.info(f"Calculated split date: {split_date.date()}")

        return split_date.strftime('%Y-%m-%d')

    def _validate_data_quality(self, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data quality checks and filtering
        """
        initial_count = len(labeled_data)

        # Remove extreme outliers (returns > 300% or < -90%)
        if 'future_return' in labeled_data.columns:
            before_filter = len(labeled_data)
            labeled_data = labeled_data[
                (labeled_data['future_return'] > -0.9) &
                (labeled_data['future_return'] < 3.0)
                ]
            after_filter = len(labeled_data)
            if before_filter != after_filter:
                self.logger.info(f"Filtered {before_filter - after_filter} extreme return outliers")

        # Check for duplicate entries
        if labeled_data.duplicated(subset=['ticker', 'date']).any():
            before_dedup = len(labeled_data)
            labeled_data = labeled_data.drop_duplicates(subset=['ticker', 'date'])
            self.logger.info(f"Removed {before_dedup - len(labeled_data)} duplicate entries")

        self.logger.info(f"Data quality check: {initial_count} → {len(labeled_data)} samples")
        return labeled_data

    def _analyze_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Analyze feature quality and potential issues
        """
        # Check feature consistency between train/test
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        if train_cols != test_cols:
            self.logger.warning("Train/test feature mismatch detected!")

        # Check for features with zero variance
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        zero_var_features = []
        for col in numeric_cols:
            if col not in ['label', 'future_return', 'relative_return']:
                if train_df[col].std() == 0:
                    zero_var_features.append(col)

        if zero_var_features:
            self.logger.warning(f"Found {len(zero_var_features)} zero-variance features")

        # Check for highly correlated features in embeddings
        emb_cols = [c for c in numeric_cols if c.startswith('emb_')]
        if len(emb_cols) > 1:
            corr_matrix = train_df[emb_cols].corr().abs()
            high_corr_pairs = (corr_matrix > 0.99).sum().sum() - len(emb_cols)
            if high_corr_pairs > 0:
                self.logger.warning(f"Found {high_corr_pairs} highly correlated embedding pairs")

    def run(self):
        """Execute complete pipeline with robust validation"""
        self.logger.info("Starting Robust Earnings NLP Pipeline...")
        self.logger.info(f"Prediction Target: {self.config.prediction_target}")
        self.logger.info(f"Model: {self.config.alpha_model}")

        # 1. Load transcripts
        self.logger.info("Loading transcripts...")
        transcripts = self.data_loader.load_transcripts()
        if len(transcripts) < 50:
            self.logger.warning(f"Only {len(transcripts)} transcripts found - results may be unreliable")

        # 2. Extract features
        self.logger.info("Extracting NLP features...")
        features_df = self.feature_extractor.extract_features(transcripts)

        # 3. Get price data with buffer for leakage prevention
        min_date = min(t.date for t in transcripts) - timedelta(days=5)
        max_date = max(t.date for t in transcripts) + timedelta(days=max(self.config.holding_periods) + 30)
        tickers = set(t.ticker for t in transcripts)

        # 4. Fetch price data
        self.logger.info("Fetching price data...")
        prices_df = self.data_loader.fetch_prices_batch(tickers, min_date, max_date)

        # Validate price data coverage
        unique_price_tickers = set(prices_df['ticker'].unique())
        missing_tickers = tickers - unique_price_tickers
        if missing_tickers:
            self.logger.warning(f"Missing price data for {len(missing_tickers)} tickers: {list(missing_tickers)[:5]}")

        # 5. Fetch benchmark data
        benchmark_df = None
        if self.config.use_market_neutral:
            self.logger.info("Fetching benchmark prices...")
            benchmark_df = self.data_loader.fetch_benchmark_prices(min_date, max_date)

        # 6. Prepare labels with leakage prevention
        self.logger.info("Preparing labels with leakage prevention...")
        labeled_data = self.alpha_model.prepare_labels(
            features_df, prices_df, benchmark_df,
            holding_period=self.config.holding_periods[0]
        )

        if len(labeled_data) == 0:
            self.logger.error("No valid labeled data generated!")
            return {}

        # 7. Data quality validation
        labeled_data = self._validate_data_quality(labeled_data)

        # 8. Robust train/test split
        if pd.to_datetime(self.config.train_test_split_date) >= pd.to_datetime(labeled_data['date']).max():
            self.logger.warning("Configured split date is after all data - using robust auto-split")
            split_date_str = self._determine_robust_split_date(labeled_data)
            split_date = pd.to_datetime(split_date_str)
        else:
            split_date = pd.to_datetime(self.config.train_test_split_date)
            split_date_str = self.config.train_test_split_date

        # Apply split with validation
        labeled_data = labeled_data.sort_values('date')
        train_df = labeled_data[labeled_data['date'] < split_date].copy()
        test_df = labeled_data[labeled_data['date'] >= split_date].copy()

        self.logger.info(f"Split date: {split_date_str}")
        self.logger.info(
            f"Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
        self.logger.info(
            f"Test: {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

        # Validate split quality
        if len(test_df) < 10:
            self.logger.error(f"Only {len(test_df)} test samples - insufficient for validation!")
            return {}

        if len(train_df) < 30:
            self.logger.warning(f"Only {len(train_df)} training samples - results may be unreliable")

        # 9. Feature preparation
        emb_cols = [c for c in train_df.columns if c.startswith('emb_')]
        linguistic_cols = ['sentiment', 'sentiment_remarks', 'sentiment_qa', 'word_count', 'readability']
        all_feature_cols = emb_cols + linguistic_cols

        # Only use features that exist and have some variance
        feature_cols = []
        for col in all_feature_cols:
            if col in train_df.columns and not train_df[col].isnull().all():
                if train_df[col].std() > 0:  # Has some variance
                    feature_cols.append(col)

        self.logger.info(
            f"Using {len(feature_cols)} features ({len(emb_cols)} embeddings, {len([c for c in linguistic_cols if c in feature_cols])} linguistic)")

        # Feature analysis
        self._analyze_features(train_df[feature_cols], test_df[feature_cols])

        # 10. Prepare training data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label']

        # Check label distribution
        if self.config.prediction_target == "direction":
            train_label_dist = y_train.value_counts()
            self.logger.info(f"Training label distribution: {train_label_dist.to_dict()}")

            if len(train_label_dist) < 2:
                self.logger.error("Training set has only one class - cannot train classifier!")
                return {}

            minority_ratio = train_label_dist.min() / len(y_train)
            if minority_ratio < 0.05:
                self.logger.warning(f"Severe class imbalance in training: {minority_ratio:.1%} minority class")

        # 11. Train model
        self.logger.info("Training model with regularization...")
        val_score = self.alpha_model.train(X_train, y_train)
        self.logger.info(f"Cross-validation score: {val_score:.4f}")

        # 12. Out-of-sample evaluation
        self.logger.info("Evaluating on out-of-sample test set...")
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['label']

        # Check test label distribution
        if self.config.prediction_target == "direction":
            test_label_dist = y_test.value_counts()
            self.logger.info(f"Test label distribution: {test_label_dist.to_dict()}")

        predictions, pred_labels = self.alpha_model.predict(X_test)

        # 13. Calculate metrics with validation
        return_col = 'relative_return' if self.config.use_market_neutral else 'future_return'
        metrics = self.backtester.evaluate(
            predictions,
            pred_labels,
            y_test,
            test_df.get(return_col),
            test_df['date']
        )

        # 14. Add model validation metrics
        metrics['train_samples'] = len(train_df)
        metrics['test_samples'] = len(test_df)
        metrics['cv_score'] = val_score
        metrics['feature_count'] = len(feature_cols)

        # Sanity checks on results
        if metrics.get('total_return', 0) > 5.0:  # >500% return
            self.logger.warning("Suspiciously high returns - possible data issues!")

        if metrics.get('win_rate', 0) > 0.9:  # >90% win rate
            self.logger.warning("Suspiciously high win rate - possible overfitting!")

        # 15. Generate comprehensive report
        report_path = Path(self.config.output_dir) / "robust_backtest_report.txt"
        report = self.backtester.generate_report(metrics, report_path)

        # Add validation info to report
        validation_info = f"\n\nMODEL VALIDATION:\n"
        validation_info += f"Cross-validation Score: {val_score:.4f}\n"
        validation_info += f"Train/Test Samples: {len(train_df)}/{len(test_df)}\n"
        validation_info += f"Features Used: {len(feature_cols)}\n"
        validation_info += f"Data Date Range: {labeled_data['date'].min().date()} to {labeled_data['date'].max().date()}\n"

        with open(report_path, 'a') as f:
            f.write(validation_info)

        self.logger.info("\n" + report + validation_info)

        # 16. Generate plots
        try:
            self.backtester.plot_results()
        except Exception as e:
            self.logger.warning(f"Could not generate plots: {e}")

        # 17. Save results
        self.backtester.save_full_results()
        self.alpha_model.save()

        return metrics