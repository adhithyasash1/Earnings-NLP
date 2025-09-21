# ============================================
# alpha_model.py - CORRECTED WITH ROBUST SORTING
# ============================================
"""Alpha signal generation with tuning and regression support"""
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from pathlib import Path

from config import Config
from utils import setup_logging


class AlphaModel:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.model_pipeline: Pipeline = None

    def _get_model(self) -> Any:
        is_classification = self.config.prediction_target == "direction"
        if self.config.alpha_model == "logistic":
            if not is_classification:
                self.logger.warning("Logistic model is for classification. Using Ridge regression instead.")
                return Ridge(random_state=self.config.random_seed)
            return LogisticRegression(max_iter=1000, random_state=self.config.random_seed)
        if self.config.alpha_model == "ridge":
            if is_classification:
                self.logger.warning("Ridge model is for regression. Using Logistic regression instead.")
                return LogisticRegression(max_iter=1000, random_state=self.config.random_seed)
            return Ridge(random_state=self.config.random_seed)
        if self.config.alpha_model == "xgboost":
            if is_classification:
                return xgb.XGBClassifier(random_state=self.config.random_seed)
            else:
                return xgb.XGBRegressor(random_state=self.config.random_seed)
        raise ValueError(f"Unknown alpha_model: {self.config.alpha_model}")

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        emb_cols = [c for c in X.columns if c.startswith('emb_')]
        linguistic_cols = [c for c in X.columns if not c.startswith('emb_')]
        emb_pipeline_steps = [('scaler', StandardScaler())]
        if self.config.use_pca and emb_cols:
            n_components = min(self.config.pca_n_components, len(emb_cols))
            self.logger.info(f"Applying PCA with {n_components} components.")
            emb_pipeline_steps.append(
                ('pca', PCA(n_components=n_components, random_state=self.config.random_seed))
            )
        emb_pipeline = Pipeline(emb_pipeline_steps)
        linguistic_pipeline = Pipeline([('scaler', StandardScaler())])
        transformers = []
        if emb_cols:
            transformers.append(('embeddings', emb_pipeline, emb_cols))
        if linguistic_cols:
            transformers.append(('linguistic', linguistic_pipeline, linguistic_cols))
        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _ensure_sorted_for_merge_asof(self, df: pd.DataFrame, sort_cols: List[str]) -> pd.DataFrame:
        """
        Ensure DataFrame is properly sorted for merge_asof operations.
        This function handles edge cases that can cause sorting issues.
        """
        # Make a clean copy
        df_clean = df.copy()

        # Ensure all sort columns are present
        for col in sort_cols:
            if col not in df_clean.columns:
                raise ValueError(f"Column {col} not found in DataFrame")

        # Only convert columns that are actually date-related to datetime
        datetime_cols = [col for col in sort_cols if 'date' in col.lower() or 'time' in col.lower()]
        for col in datetime_cols:
            if col in df_clean.columns:
                # Only convert if it's not already a datetime type
                if not pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    df_clean[col] = pd.to_datetime(df_clean[col])

        # Sort and reset index
        df_clean = df_clean.sort_values(sort_cols).reset_index(drop=True)

        # Remove any duplicate entries that could cause issues
        if len(sort_cols) > 1:
            df_clean = df_clean.drop_duplicates(subset=sort_cols, keep='first')

        return df_clean

    def prepare_labels(self, features_df: pd.DataFrame, prices_df: pd.DataFrame,
                       benchmark_df: pd.DataFrame = None,
                       holding_period: int = 5) -> pd.DataFrame:

        self.logger.info(f"Preparing labels with holding period: {holding_period}")
        self.logger.info(f"Features shape: {features_df.shape}, Prices shape: {prices_df.shape}")

        # Create copies to avoid modifying original data
        features_df = features_df.copy()
        prices_df = prices_df.copy()

        # Ensure datetime columns are properly formatted
        features_df['date'] = pd.to_datetime(features_df['date'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

        # Create entry and exit dates
        features_df['entry_date'] = features_df['date'] + pd.Timedelta(days=1)
        features_df['exit_date'] = features_df['date'] + pd.Timedelta(days=holding_period + 1)

        # Remove any rows with NaN values in key columns
        features_df = features_df.dropna(subset=['ticker', 'entry_date', 'exit_date'])
        prices_df = prices_df.dropna(subset=['ticker', 'timestamp', 'close'])

        self.logger.debug(f"After cleanup - Features: {features_df.shape}, Prices: {prices_df.shape}")

        # ALTERNATIVE APPROACH: Use regular merge with manual date filtering
        # This avoids the strict sorting requirements of merge_asof

        results = []

        for _, feature_row in features_df.iterrows():
            ticker = feature_row['ticker']
            entry_date = feature_row['entry_date']
            exit_date = feature_row['exit_date']

            # Get prices for this ticker
            ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()

            if ticker_prices.empty:
                continue

            # Find entry price (first price on or after entry_date)
            entry_prices = ticker_prices[ticker_prices['timestamp'] >= entry_date]
            if entry_prices.empty:
                continue
            entry_price = entry_prices.iloc[0]['close']

            # Find exit price (last price on or before exit_date)
            exit_prices = ticker_prices[ticker_prices['timestamp'] <= exit_date]
            if exit_prices.empty:
                continue
            exit_price = exit_prices.iloc[-1]['close']

            # Create result row
            result_row = feature_row.to_dict()
            result_row['entry_price'] = entry_price
            result_row['exit_price'] = exit_price
            result_row['future_return'] = (exit_price - entry_price) / entry_price

            results.append(result_row)

        if not results:
            self.logger.error("No valid price matches found!")
            return pd.DataFrame()

        merged = pd.DataFrame(results)
        self.logger.info(f"After price matching: {merged.shape}")

        # Handle benchmark data if market neutral
        if self.config.use_market_neutral and benchmark_df is not None:
            self.logger.debug("Processing benchmark data...")

            benchmark_df = benchmark_df.copy()
            benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
            benchmark_df = benchmark_df.dropna(subset=['timestamp', 'close'])

            # Add benchmark returns using the same manual approach
            benchmark_results = []

            for _, row in merged.iterrows():
                entry_date = row['entry_date']
                exit_date = row['exit_date']

                # Find benchmark entry price
                bench_entry_prices = benchmark_df[benchmark_df['timestamp'] >= entry_date]
                if bench_entry_prices.empty:
                    continue
                bench_entry = bench_entry_prices.iloc[0]['close']

                # Find benchmark exit price
                bench_exit_prices = benchmark_df[benchmark_df['timestamp'] <= exit_date]
                if bench_exit_prices.empty:
                    continue
                bench_exit = bench_exit_prices.iloc[-1]['close']

                # Add benchmark data to row
                result_row = row.to_dict()
                result_row['bench_entry'] = bench_entry
                result_row['bench_exit'] = bench_exit
                result_row['benchmark_return'] = (bench_exit - bench_entry) / bench_entry
                result_row['relative_return'] = result_row['future_return'] - result_row['benchmark_return']

                benchmark_results.append(result_row)

            merged = pd.DataFrame(benchmark_results)

            # Set labels based on relative returns
            if self.config.prediction_target == "direction":
                merged['label'] = (merged['relative_return'] > 0).astype(int)
            else:
                merged['label'] = merged['relative_return']
        else:
            # Set labels based on absolute returns
            if self.config.prediction_target == "direction":
                merged['label'] = (merged['future_return'] > 0).astype(int)
            else:
                merged['label'] = merged['future_return']

        # Final cleanup
        result = merged.dropna(subset=['label', 'entry_price', 'exit_price'])
        self.logger.info(f"Final labeled data shape: {result.shape}")

        return result

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        preprocessor = self._build_preprocessor(X)
        model = self._get_model()
        self.model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        tscv = TimeSeriesSplit(n_splits=3)
        if self.config.use_hyperparam_tuning:
            param_grid = self.config.load_param_grid()
            if not param_grid:
                self.logger.warning(
                    "Hyperparameter tuning enabled but no param grid found. Fitting with default params.")
                self.model_pipeline.fit(X, y)
                return -1.0
            self.logger.info(f"Running GridSearchCV with {self.config.tuner_scoring_metric}...")
            tuner = GridSearchCV(
                estimator=self.model_pipeline,
                param_grid=param_grid,
                cv=tscv,
                scoring=self.config.tuner_scoring_metric,
                n_jobs=-1
            )
            tuner.fit(X, y)
            self.model_pipeline = tuner.best_estimator_
            self.logger.info(f"Best params: {tuner.best_params_}")
            return tuner.best_score_
        else:
            self.logger.info("Fitting model with default parameters...")
            self.model_pipeline.fit(X, y)
            self.logger.warning("No cross-validation score calculated (tuning is disabled).")
            return -1.0

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_pipeline is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        is_classification = self.config.prediction_target == "direction"
        if is_classification:
            probs = self.model_pipeline.predict_proba(X)[:, 1]
            labels = (probs > 0.5).astype(int)
            return probs, labels
        else:
            predictions = self.model_pipeline.predict(X)
            return predictions, predictions

    def save(self, path: str = None):
        if path is None:
            path = Path(self.config.output_dir) / "model.pkl"
        joblib.dump({
            'model_pipeline': self.model_pipeline,
            'config': self.config.dict()
        }, path)
        self.logger.info(f"Full model pipeline saved to {path}")

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        config = Config(**data.get('config', {}))
        instance = cls(config)
        instance.model_pipeline = data['model_pipeline']
        instance.logger.info(f"Model pipeline loaded from {path}")
        return instance