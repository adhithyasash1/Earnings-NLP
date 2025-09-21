"""Configuration"""
from pydantic_settings import BaseSettings
from typing import List, Dict
import yaml
import json
from pathlib import Path


class Config(BaseSettings):
    transcript_source: str = "seeking_alpha"
    price_source: str = "yfinance"
    transcript_dir: str = "./transcript_data"
    embedding_model: str = "philschmid/bge-base-financial-matryoshka"
    alpha_model: str = "xgboost"
    use_pca: bool = False
    holding_periods: List[int] = [5]
    prediction_target: str = "direction"
    use_market_neutral: bool = True
    benchmark_ticker: str = "SPY"
    labeling_strategy: str = "median_split"  # or "quantile"
    train_test_split_date: str = "2023-01-01"
    random_seed: int = 42
    use_hyperparam_tuning: bool = True
    tuner_param_grid_path: str = "./configs/params.json"
    tuner_scoring_metric: str = "accuracy"
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    log_level: str = "INFO"

    class Config:
        env_file = '.env'

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def setup_directories(self):
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.cache_dir).mkdir(exist_ok=True)

    def load_param_grid(self) -> Dict:
        if not self.tuner_param_grid_path:
            return {}
        with open(self.tuner_param_grid_path, 'r') as f:
            all_grids = json.load(f)
        return all_grids.get(self.alpha_model, {})