# ============================================
# alpha_model.py - BALANCED LABELING STRATEGIES
# ============================================
"""Alpha signal generation with multiple labeling strategies to handle class imbalance"""
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
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
        self.labeling_strategy = "relative_quantile"  # New default strategy

    def _get_model(self) -> Any:
        is_classification = self.config.prediction_target == "direction"
        if self.config.alpha_model == "logistic":
            if not is_classification:
                self.logger.warning("Logistic model is for classification. Using Ridge regression instead.")
                return Ridge(random_state=self.config.random_seed, alpha=1.0)
            return LogisticRegression(max_iter=1000, random_state=self.config.random_seed,
                                      C=0.1, penalty='l2')
        if self.config.alpha_model == "ridge":
            if is_classification:
                self.logger.warning("Ridge model is for regression. Using Logistic regression instead.")
                return LogisticRegression(max_iter=1000, random_state=self.config.random_seed,
                                          C=0.1, penalty='l2')
            return Ridge(random_state=self.config.random_seed, alpha=10.0)
        if self.config.alpha_model == "xgboost":
            if is_classification:
                return xgb.XGBClassifier(
                    random_state=self.config.random_seed,
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=1.0
                )
            else:
                return xgb.XGBRegressor(
                    random_state=self.config.random_seed,
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=1.0
                )
        raise ValueError(f"Unknown alpha_model: {self.config.alpha_model}")

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        emb_cols = [c for c in X.columns if c.startswith('emb_')]
        linguistic_cols = [c for c in X.columns if not c.startswith('emb_')]

        # Reduce PCA components significantly
        max_components = min(10, len(emb_cols) // 10, X.shape[0] // 20) if emb_cols else 0

        emb_pipeline_steps = [('scaler', StandardScaler())]
        if self.config.use_pca and emb_cols and max_components > 0:
            self.logger.info(f"Applying PCA with {max_components} components (heavily reduced to prevent overfitting)")
            emb_pipeline_steps.append(
                ('pca', PCA(n_components=max_components, random_state=self.config.random_seed))
            )

        linguistic_pipeline_steps = [('scaler', StandardScaler())]
        if linguistic_cols:
            max_linguistic_features = min(5, len(linguistic_cols), X.shape[0] // 30)
            if max_linguistic_features > 0:
                score_func = f_classif if self.config.prediction_target == "direction" else f_regression
                linguistic_pipeline_steps.append(
                    ('selector', SelectKBest(score_func=score_func, k=max_linguistic_features))
                )
                self.logger.info(f"Selecting top {max_linguistic_features} linguistic features")

        emb_pipeline = Pipeline(emb_pipeline_steps)
        linguistic_pipeline = Pipeline(linguistic_pipeline_steps)

        transformers = []
        if emb_cols and max_components > 0:
            transformers.append(('embeddings', emb_pipeline, emb_cols))
        if linguistic_cols:
            transformers.append(('linguistic', linguistic_pipeline, linguistic_cols))

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _create_balanced_labels(self, merged: pd.DataFrame, return_col: str) -> pd.DataFrame:
        """
        Create balanced labels using multiple strategies
        """
        returns = merged[return_col].copy()

        self.logger.info(f"Return statistics: mean={returns.mean():.4f}, std={returns.std():.4f}")
        self.logger.info(f"Return range: {returns.min():.4f} to {returns.max():.4f}")

        # Strategy 1: Quantile-based labeling (most robust)
        if self.labeling_strategy == "relative_quantile":
            # Use top 40% as positive, bottom 40% as negative, discard middle 20%
            q_upper = returns.quantile(0.6)
            q_lower = returns.quantile(0.4)

            self.logger.info(f"Quantile thresholds: lower={q_lower:.4f}, upper={q_upper:.4f}")

            # Filter to only high-confidence predictions
            filtered_mask = (returns <= q_lower) | (returns >= q_upper)
            merged_filtered = merged[filtered_mask].copy()

            if len(merged_filtered) < 20:
                self.logger.warning("Quantile filtering resulted in too few samples, using median split")
                return self._median_split_labels(merged, return_col)

            merged_filtered['label'] = (merged_filtered[return_col] >= q_upper).astype(int)

            label_dist = merged_filtered['label'].value_counts()
            self.logger.info(f"Quantile-based label distribution: {label_dist.to_dict()}")

            return merged_filtered

        # Strategy 2: Median split (fallback)
        elif self.labeling_strategy == "median_split":
            return self._median_split_labels(merged, return_col)

        # Strategy 3: Time-based relative (comparing to past performance)
        elif self.labeling_strategy == "time_relative":
            return self._time_relative_labels(merged, return_col)

        else:
            raise ValueError(f"Unknown labeling strategy: {self.labeling_strategy}")

    def _median_split_labels(self, merged: pd.DataFrame, return_col: str) -> pd.DataFrame:
        """Median split labeling"""
        median_return = merged[return_col].median()
        merged['label'] = (merged[return_col] >= median_return).astype(int)

        label_dist = merged['label'].value_counts()
        self.logger.info(f"Median split (threshold={median_return:.4f}) label distribution: {label_dist.to_dict()}")

        return merged

    def _time_relative_labels(self, merged: pd.DataFrame, return_col: str) -> pd.DataFrame:
        """Time-based relative labeling"""
        merged = merged.sort_values('date').copy()

        # Calculate rolling 30-day median return
        merged['rolling_median'] = merged[return_col].rolling(window=30, min_periods=10).median()

        # Label as 1 if above recent median, 0 otherwise
        valid_mask = merged['rolling_median'].notna()
        merged_valid = merged[valid_mask].copy()

        if len(merged_valid) < 20:
            self.logger.warning("Time-relative filtering resulted in too few samples, using median split")
            return self._median_split_labels(merged, return_col)

        merged_valid['label'] = (merged_valid[return_col] >= merged_valid['rolling_median']).astype(int)

        label_dist = merged_valid['label'].value_counts()
        self.logger.info(f"Time-relative label distribution: {label_dist.to_dict()}")

        return merged_valid

    def prepare_labels(self, features_df: pd.DataFrame, prices_df: pd.DataFrame,
                       benchmark_df: pd.DataFrame = None,
                       holding_period: int = 5) -> pd.DataFrame:

        self.logger.info(f"Preparing labels with holding period: {holding_period}")
        self.logger.info(f"Features shape: {features_df.shape}, Prices shape: {prices_df.shape}")

        # Create copies and prevent leakage (same as before)
        features_df = features_df.copy()
        prices_df = prices_df.copy()

        features_df['date'] = pd.to_datetime(features_df['date'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

        # 2-day buffer to prevent leakage
        features_df['entry_date'] = features_df['date'] + pd.Timedelta(days=2)
        features_df['exit_date'] = features_df['entry_date'] + pd.Timedelta(days=holding_period)

        features_df = features_df.dropna(subset=['ticker', 'entry_date', 'exit_date'])
        prices_df = prices_df.dropna(subset=['ticker', 'timestamp', 'close'])

        # Price matching (same logic as before)
        results = []

        for _, feature_row in features_df.iterrows():
            ticker = feature_row['ticker']
            earnings_date = feature_row['date']
            entry_date = feature_row['entry_date']
            exit_date = feature_row['exit_date']

            ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
            if ticker_prices.empty:
                continue

            valid_prices = ticker_prices[ticker_prices['timestamp'] != earnings_date]

            entry_prices = valid_prices[valid_prices['timestamp'] >= entry_date]
            if entry_prices.empty:
                continue
            entry_price = entry_prices.iloc[0]['close']
            entry_actual_date = entry_prices.iloc[0]['timestamp']

            exit_prices = valid_prices[valid_prices['timestamp'] <= exit_date]
            if exit_prices.empty:
                continue
            exit_price = exit_prices.iloc[-1]['close']
            exit_actual_date = exit_prices.iloc[-1]['timestamp']

            actual_holding_days = (exit_actual_date - entry_actual_date).days
            if actual_holding_days < 1:
                continue

            result_row = feature_row.to_dict()
            result_row['entry_price'] = entry_price
            result_row['exit_price'] = exit_price
            result_row['entry_actual_date'] = entry_actual_date
            result_row['exit_actual_date'] = exit_actual_date
            result_row['actual_holding_days'] = actual_holding_days
            result_row['future_return'] = (exit_price - entry_price) / entry_price

            results.append(result_row)

        if not results:
            self.logger.error("No valid price matches found!")
            return pd.DataFrame()

        merged = pd.DataFrame(results)
        self.logger.info(f"After price matching: {merged.shape}")

        # Handle benchmark (same as before)
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

                bench_entry_prices = valid_benchmark[valid_benchmark['timestamp'] >= entry_actual_date]
                if bench_entry_prices.empty:
                    continue
                bench_entry = bench_entry_prices.iloc[0]['close']

                bench_exit_prices = valid_benchmark[valid_benchmark['timestamp'] <= exit_actual_date]
                if bench_exit_prices.empty:
                    continue
                bench_exit = bench_exit_prices.iloc[-1]['close']

                result_row = row.to_dict()
                result_row['bench_entry'] = bench_entry
                result_row['bench_exit'] = bench_exit
                result_row['benchmark_return'] = (bench_exit - bench_entry) / bench_entry
                result_row['relative_return'] = result_row['future_return'] - result_row['benchmark_return']

                benchmark_results.append(result_row)

            merged = pd.DataFrame(benchmark_results)
            return_col = 'relative_return'
        else:
            return_col = 'future_return'

        # NEW: Apply balanced labeling strategy
        if self.config.prediction_target == "direction":
            result = self._create_balanced_labels(merged, return_col)
        else:
            merged['label'] = merged[return_col]
            result = merged

        # Final cleanup
        result = result.dropna(subset=['label', 'entry_price', 'exit_price'])
        self.logger.info(f"Final labeled data shape: {result.shape}")

        return result

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        if X.shape[0] < 20:
            self.logger.warning("Very few training samples - results may be unreliable")

        if X.shape[1] > X.shape[0] / 2:
            self.logger.warning(f"High feature-to-sample ratio ({X.shape[1]}/{X.shape[0]}) - overfitting risk")

        preprocessor = self._build_preprocessor(X)
        model = self._get_model()

        self.model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        tscv = TimeSeriesSplit(n_splits=min(3, X.shape[0] // 15))

        if self.config.use_hyperparam_tuning:
            param_grid = self.config.load_param_grid()
            if not param_grid:
                self.logger.warning("No param grid found. Using default parameters.")
                self.model_pipeline.fit(X, y)
                return self._cross_validate(X, y, tscv)

            self.logger.info(f"Running GridSearchCV...")
            tuner = GridSearchCV(
                estimator=self.model_pipeline,
                param_grid=param_grid,
                cv=tscv,
                scoring=self.config.tuner_scoring_metric,
                n_jobs=1
            )

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                tuner.fit(X, y)

            self.model_pipeline = tuner.best_estimator_
            self.logger.info(f"Best params: {tuner.best_params_}")
            return tuner.best_score_
        else:
            self.logger.info("Fitting model with regularized parameters...")
            self.model_pipeline.fit(X, y)
            return self._cross_validate(X, y, tscv)

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, cv) -> float:
        from sklearn.model_selection import cross_val_score

        try:
            scores = cross_val_score(
                self.model_pipeline, X, y,
                cv=cv, scoring=self.config.tuner_scoring_metric
            )
            return scores.mean()
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            return -1.0

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_pipeline is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        is_classification = self.config.prediction_target == "direction"

        if is_classification:
            try:
                probs = self.model_pipeline.predict_proba(X)
                if probs.shape[1] == 1:
                    labels = np.zeros(X.shape[0])
                    probs_out = np.full(X.shape[0], 0.5)
                else:
                    labels = (probs[:, 1] > 0.5).astype(int)
                    probs_out = probs[:, 1]
                return probs_out, labels
            except Exception as e:
                self.logger.warning(f"Prediction failed: {e}")
                return np.full(X.shape[0], 0.5), np.zeros(X.shape[0])
        else:
            predictions = self.model_pipeline.predict(X)
            return predictions, predictions

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
        instance.labeling_strategy = data.get('labeling_strategy', 'relative_quantile')
        instance.logger.info(f"Model loaded from {path}")
        return instance