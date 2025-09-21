# ============================================
# alpha_model.py - ULTRA-CONSERVATIVE VERSION
# ============================================
"""Alpha signal generation with extreme overfitting prevention"""
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from pathlib import Path
import warnings

from config import Config
from utils import setup_logging


class AlphaModel:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.model_pipeline: Pipeline = None
        self.labeling_strategy = "median_split"

    def _get_model(self) -> Any:
        is_classification = self.config.prediction_target == "direction"

        # Use very conservative parameters to prevent overfitting
        if self.config.alpha_model == "logistic":
            if not is_classification:
                return Ridge(random_state=self.config.random_seed, alpha=100.0)  # High regularization
            return LogisticRegression(
                max_iter=2000,
                random_state=self.config.random_seed,
                C=0.1,  # Strong regularization
                penalty='l2',
                solver='liblinear'  # More stable for small datasets
            )
        elif self.config.alpha_model == "ridge":
            if is_classification:
                return LogisticRegression(max_iter=2000, random_state=self.config.random_seed,
                                          C=0.1, penalty='l2', solver='liblinear')
            return Ridge(random_state=self.config.random_seed, alpha=100.0)
        elif self.config.alpha_model == "xgboost":
            if is_classification:
                return xgb.XGBClassifier(
                    random_state=self.config.random_seed,
                    n_estimators=10,  # Very few trees
                    max_depth=2,  # Shallow trees
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=10.0,  # Very high regularization
                    reg_lambda=10.0
                )
            else:
                return xgb.XGBRegressor(
                    random_state=self.config.random_seed,
                    n_estimators=10,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=10.0,
                    reg_lambda=10.0
                )
        raise ValueError(f"Unknown alpha_model: {self.config.alpha_model}")

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Ultra-conservative feature selection to prevent overfitting"""

        emb_cols = [c for c in X.columns if c.startswith('emb_')]
        linguistic_cols = [c for c in X.columns if not c.startswith('emb_')]

        # EXTREMELY aggressive feature reduction
        n_samples = X.shape[0]
        max_total_features = min(8, max(3, n_samples // 15))  # Very conservative ratio

        self.logger.info(f"Ultra-conservative feature limit: {max_total_features} features for {n_samples} samples")

        transformers = []

        # Prioritize linguistic features (they're often more interpretable)
        if linguistic_cols:
            max_linguistic = min(3, len(linguistic_cols), max_total_features)

            linguistic_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(
                    score_func=mutual_info_classif if self.config.prediction_target == "direction" else f_regression,
                    k=max_linguistic
                ))
            ])

            transformers.append(('linguistic', linguistic_pipeline, linguistic_cols))
            remaining_features = max_total_features - max_linguistic
            self.logger.info(f"Using {max_linguistic} linguistic features")
        else:
            remaining_features = max_total_features

        # Only use embeddings if we have remaining budget and they're likely to help
        if emb_cols and remaining_features > 0 and n_samples > 30:
            max_components = min(remaining_features, 5, len(emb_cols) // 100)  # Very few components

            if max_components > 0:
                emb_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=max_components, random_state=self.config.random_seed))
                ])

                transformers.append(('embeddings', emb_pipeline, emb_cols))
                self.logger.info(f"Using PCA with {max_components} components from {len(emb_cols)} embeddings")
            else:
                self.logger.info("Skipping embeddings due to sample size constraints")
        else:
            self.logger.info("Skipping embeddings to prevent overfitting")

        if not transformers:
            # Fallback: use just a few linguistic features
            self.logger.warning("No features selected! Using minimal fallback.")
            basic_pipeline = Pipeline([('scaler', StandardScaler())])
            safe_cols = linguistic_cols[:3] if linguistic_cols else X.columns[:3]
            transformers = [('basic', basic_pipeline, safe_cols)]

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _create_balanced_labels(self, merged: pd.DataFrame, return_col: str) -> pd.DataFrame:
        """Simple median split with data quality checks"""

        returns = merged[return_col].copy()

        self.logger.info(f"Return statistics: mean={returns.mean():.4f}, std={returns.std():.4f}")
        self.logger.info(f"Return range: {returns.min():.4f} to {returns.max():.4f}")
        self.logger.info(f"Positive returns: {(returns > 0).sum()} ({(returns > 0).mean():.1%})")

        # Data quality warnings
        if abs(returns.mean()) > 0.01:  # Mean return > 1%
            self.logger.warning(f"High average return ({returns.mean():.2%}) - check for data bias")

        if returns.std() > 0.1:  # Volatility > 10%
            self.logger.warning(f"High return volatility ({returns.std():.1%}) - possible data quality issues")

        # Simple median split
        median_return = returns.median()
        merged['label'] = (merged[return_col] >= median_return).astype(int)

        label_dist = merged['label'].value_counts()
        self.logger.info(f"Median split (threshold={median_return:.4f}) label distribution: {label_dist.to_dict()}")

        return merged

    def prepare_labels(self, features_df: pd.DataFrame, prices_df: pd.DataFrame,
                       benchmark_df: pd.DataFrame = None,
                       holding_period: int = 5) -> pd.DataFrame:

        self.logger.info(f"Preparing labels with holding period: {holding_period}")
        self.logger.info(f"Features shape: {features_df.shape}, Prices shape: {prices_df.shape}")

        # Create copies and prevent leakage
        features_df = features_df.copy()
        prices_df = prices_df.copy()

        features_df['date'] = pd.to_datetime(features_df['date'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

        # 2-day buffer to prevent leakage
        features_df['entry_date'] = features_df['date'] + pd.Timedelta(days=2)
        features_df['exit_date'] = features_df['entry_date'] + pd.Timedelta(days=holding_period)

        features_df = features_df.dropna(subset=['ticker', 'entry_date', 'exit_date'])
        prices_df = prices_df.dropna(subset=['ticker', 'timestamp', 'close'])

        # Price matching with enhanced validation
        results = []

        for _, feature_row in features_df.iterrows():
            ticker = feature_row['ticker']
            earnings_date = feature_row['date']
            entry_date = feature_row['entry_date']
            exit_date = feature_row['exit_date']

            ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
            if ticker_prices.empty:
                continue

            # Exclude earnings date and sort by timestamp
            valid_prices = ticker_prices[ticker_prices['timestamp'] != earnings_date]
            valid_prices = valid_prices.sort_values('timestamp')

            if len(valid_prices) < 2:
                continue

            # Find entry price
            entry_prices = valid_prices[valid_prices['timestamp'] >= entry_date]
            if entry_prices.empty:
                continue
            entry_price = entry_prices.iloc[0]['close']
            entry_actual_date = entry_prices.iloc[0]['timestamp']

            # Find exit price
            exit_prices = valid_prices[valid_prices['timestamp'] <= exit_date]
            if exit_prices.empty:
                continue
            exit_price = exit_prices.iloc[-1]['close']
            exit_actual_date = exit_prices.iloc[-1]['timestamp']

            actual_holding_days = (exit_actual_date - entry_actual_date).days
            if actual_holding_days < 1:
                continue

            # Enhanced return calculation validation
            if entry_price <= 0 or exit_price <= 0:
                self.logger.warning(f"Invalid prices for {ticker}: entry=${entry_price}, exit=${exit_price}")
                continue

            future_return = (exit_price - entry_price) / entry_price

            # More conservative outlier filtering
            if abs(future_return) > 0.5:  # >50% return - likely data error
                self.logger.warning(f"Extreme return filtered: {ticker} {future_return:.1%} "
                                    f"(${entry_price} -> ${exit_price})")
                continue

            # Additional sanity checks
            if entry_actual_date >= exit_actual_date:
                self.logger.warning(f"Invalid date order for {ticker}")
                continue

            result_row = feature_row.to_dict()
            result_row.update({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_actual_date': entry_actual_date,
                'exit_actual_date': exit_actual_date,
                'actual_holding_days': actual_holding_days,
                'future_return': future_return
            })

            results.append(result_row)

        if not results:
            self.logger.error("No valid price matches found!")
            return pd.DataFrame()

        merged = pd.DataFrame(results)
        self.logger.info(f"After price matching: {merged.shape}")

        # Handle benchmark (keeping existing logic)
        if self.config.use_market_neutral and benchmark_df is not None:
            benchmark_df = benchmark_df.copy()
            benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
            benchmark_df = benchmark_df.dropna(subset=['timestamp', 'close'])

            benchmark_results = []

            for _, row in merged.iterrows():
                earnings_date = row['date']
                entry_actual_date = row['entry_actual_date']
                exit_actual_date = row['exit_actual_date']

                valid_benchmark = benchmark_df[benchmark_df['timestamp'] != earnings_date]
                valid_benchmark = valid_benchmark.sort_values('timestamp')

                bench_entry_prices = valid_benchmark[valid_benchmark['timestamp'] >= entry_actual_date]
                if bench_entry_prices.empty:
                    continue
                bench_entry = bench_entry_prices.iloc[0]['close']

                bench_exit_prices = valid_benchmark[valid_benchmark['timestamp'] <= exit_actual_date]
                if bench_exit_prices.empty:
                    continue
                bench_exit = bench_exit_prices.iloc[-1]['close']

                if bench_entry <= 0 or bench_exit <= 0:
                    continue

                benchmark_return = (bench_exit - bench_entry) / bench_entry

                # Filter extreme benchmark returns too
                if abs(benchmark_return) > 0.3:
                    continue

                result_row = row.to_dict()
                result_row.update({
                    'bench_entry': bench_entry,
                    'bench_exit': bench_exit,
                    'benchmark_return': benchmark_return,
                    'relative_return': result_row['future_return'] - benchmark_return
                })

                benchmark_results.append(result_row)

            if benchmark_results:
                merged = pd.DataFrame(benchmark_results)
                return_col = 'relative_return'
                self.logger.info(f"Using market-neutral returns (relative to benchmark)")
            else:
                self.logger.warning("Benchmark processing failed, using absolute returns")
                return_col = 'future_return'
        else:
            return_col = 'future_return'

        # Apply labeling strategy
        if self.config.prediction_target == "direction":
            result = self._create_balanced_labels(merged, return_col)
        else:
            merged['label'] = merged[return_col]
            result = merged

        # Final cleanup with validation
        result = result.dropna(subset=['label', 'entry_price', 'exit_price'])

        # Additional quality checks
        valid_prices = (result['entry_price'] > 0) & (result['exit_price'] > 0)
        result = result[valid_prices]

        self.logger.info(f"Final labeled data shape: {result.shape}")

        return result

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        if X.shape[0] < 10:
            self.logger.error("Too few training samples")
            return 0.0

        # Very strict overfitting warnings
        feature_ratio = X.shape[1] / X.shape[0]
        if feature_ratio > 0.1:  # Lowered threshold
            self.logger.error(f"SEVERE overfitting risk! Feature-to-sample ratio: {feature_ratio:.2f}")
        elif feature_ratio > 0.05:
            self.logger.warning(f"High overfitting risk! Feature-to-sample ratio: {feature_ratio:.2f}")

        preprocessor = self._build_preprocessor(X)
        model = self._get_model()

        self.model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Conservative cross-validation
        n_splits = max(2, min(3, X.shape[0] // 25))  # Larger splits for stability
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Disable hyperparameter tuning for small datasets to prevent overfitting
        if self.config.use_hyperparam_tuning and X.shape[0] > 50:  # Raised threshold
            param_grid = self.config.load_param_grid()
            if param_grid:
                self.logger.info(f"Running conservative GridSearchCV...")
                tuner = GridSearchCV(
                    estimator=self.model_pipeline,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring=self.config.tuner_scoring_metric,
                    n_jobs=1
                )

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    tuner.fit(X, y)

                self.model_pipeline = tuner.best_estimator_
                self.logger.info(f"Best params: {tuner.best_params_}")
                return tuner.best_score_
        else:
            if self.config.use_hyperparam_tuning:
                self.logger.info("Disabling hyperparameter tuning due to small dataset size")

        # Standard training with conservative parameters
        self.logger.info("Fitting model with conservative regularization...")
        self.model_pipeline.fit(X, y)
        return self._cross_validate(X, y, tscv)

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, cv) -> float:
        from sklearn.model_selection import cross_val_score

        try:
            scores = cross_val_score(
                self.model_pipeline, X, y,
                cv=cv, scoring=self.config.tuner_scoring_metric
            )
            self.logger.info(f"CV scores: {scores} (mean: {scores.mean():.4f}, std: {scores.std():.4f})")
            return scores.mean()
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            return 0.0

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_pipeline is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        is_classification = self.config.prediction_target == "direction"

        try:
            if is_classification:
                probs = self.model_pipeline.predict_proba(X)
                if probs.shape[1] == 1:
                    labels = np.zeros(X.shape[0])
                    probs_out = np.full(X.shape[0], 0.5)
                else:
                    labels = (probs[:, 1] > 0.5).astype(int)
                    probs_out = probs[:, 1]
                return probs_out, labels
            else:
                predictions = self.model_pipeline.predict(X)
                return predictions, predictions
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.full(X.shape[0], 0.5), np.zeros(X.shape[0])

    def save(self, path: str = None):
        if path is None:
            path = Path(self.config.output_dir) / "model.pkl"
        joblib.dump({
            'model_pipeline': self.model_pipeline,
            'labeling_strategy': self.labeling_strategy,
            'config': self.config.dict()
        }, path)
        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        config = Config(**data.get('config', {}))
        instance = cls(config)
        instance.model_pipeline = data['model_pipeline']
        instance.labeling_strategy = data.get('labeling_strategy', 'median_split')
        instance.logger.info(f"Model loaded from {path}")
        return instance