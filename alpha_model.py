# ============================================
# alpha_model.py - CORRECTED
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

from config import Config
from utils import setup_logging


class AlphaModel:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.model_pipeline: Pipeline = None

    def _get_model(self) -> Any:
        # ... (this part of the code is correct and unchanged)
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
        # ... (this part of the code is correct and unchanged)
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

    def prepare_labels(self, features_df: pd.DataFrame, prices_df: pd.DataFrame,
                       benchmark_df: pd.DataFrame = None,
                       holding_period: int = 5) -> pd.DataFrame:

        # --- FIX: Sort both DataFrames by ticker and date before merging ---
        features_df = features_df.sort_values(['ticker', 'date'])
        prices_df = prices_df.sort_values(['ticker', 'timestamp'])
        # -----------------------------------------------------------------

        features_df['entry_date'] = features_df['date'] + pd.Timedelta(days=1)
        features_df['exit_date'] = features_df['date'] + pd.Timedelta(days=holding_period + 1)

        # Merge entry prices
        merged = pd.merge_asof(
            features_df.sort_values('entry_date'),
            prices_df.rename(columns={'timestamp': 'entry_date', 'close': 'entry_price'}),
            on='entry_date', by='ticker', direction='forward')

        # Merge exit prices
        merged = pd.merge_asof(
            merged.sort_values('exit_date'),
            prices_df.rename(columns={'timestamp': 'exit_date', 'close': 'exit_price'}),
            on='exit_date', by='ticker', direction='backward')

        # ... (rest of the function is correct and unchanged) ...
        merged['future_return'] = (merged['exit_price'] - merged['entry_price']) / merged['entry_price']
        if self.config.use_market_neutral and benchmark_df is not None:
            benchmark_df = benchmark_df.sort_values('timestamp')
            merged = pd.merge_asof(
                merged.sort_values('entry_date'),
                benchmark_df.rename(columns={'timestamp': 'entry_date', 'close': 'bench_entry'}),
                on='entry_date', direction='forward')
            merged = pd.merge_asof(
                merged.sort_values('exit_date'),
                benchmark_df.rename(columns={'timestamp': 'exit_date', 'close': 'bench_exit'}),
                on='exit_date', direction='backward')
            merged['benchmark_return'] = (merged['bench_exit'] - merged['bench_entry']) / merged['bench_entry']
            merged['relative_return'] = merged['future_return'] - merged['benchmark_return']
            if self.config.prediction_target == "direction":
                merged['label'] = (merged['relative_return'] > 0).astype(int)
            else:
                merged['label'] = merged['relative_return']
        else:
            if self.config.prediction_target == "direction":
                merged['label'] = (merged['future_return'] > 0).astype(int)
            else:
                merged['label'] = merged['future_return']
        return merged.dropna(subset=['label', 'entry_price', 'exit_price'])

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        # ... (this part of the code is correct and unchanged)
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
        # ... (this part of the code is correct and unchanged)
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
        # ... (this part of the code is correct and unchanged)
        if path is None:
            path = Path(self.config.output_dir) / "model.pkl"
        joblib.dump({
            'model_pipeline': self.model_pipeline,
            'config': self.config.dict()
        }, path)
        self.logger.info(f"Full model pipeline saved to {path}")

    @classmethod
    def load(cls, path: str):
        # ... (this part of the code is correct and unchanged)
        data = joblib.load(path)
        config = Config(**data.get('config', {}))
        instance = cls(config)
        instance.model_pipeline = data['model_pipeline']
        instance.logger.info(f"Model pipeline loaded from {path}")
        return instance