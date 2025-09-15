# ============================================
# config.py - Configuration with pydantic
# ============================================
"""Configuration settings with validation and YAML support"""
from pydantic_settings import BaseSettings
from pydantic import Field, FilePath
from typing import Optional, List, Dict, Any
import yaml
import json
from pathlib import Path


class Config(BaseSettings):
    # Data sources
    transcript_source: str = "seeking_alpha"
    price_source: str = "yfinance"
    transcript_dir: str = "./transcript_data"

    # Model settings
    embedding_model: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    alpha_model: str = "xgboost"  # or "xgboost", "ridge"
    use_pca: bool = True
    pca_n_components: int = 50

    # Trading parameters
    holding_periods: List[int] = Field(default_factory=lambda: [1, 5, 20])
    prediction_target: str = "direction"  # or "returns"
    use_market_neutral: bool = True
    benchmark_ticker: str = "SPY"

    # Training settings
    train_test_split_date: str = "2023-01-01"
    random_seed: int = 42

    # --- NEW: Hyperparameter Tuning ---
    use_hyperparam_tuning: bool = True
    tuner_param_grid_path: Optional[str] = "./configs/params.json"
    tuner_scoring_metric: str = "accuracy"  # e.g., 'accuracy' or 'neg_mean_squared_error'

    # Output settings
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    log_level: str = "INFO"

    # API keys
    alpaca_key: Optional[str] = None
    alpaca_secret: Optional[str] = None

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def setup_directories(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def load_param_grid(self) -> Dict[str, Any]:
        """Loads the hyperparameter grid from the JSON path"""
        if not self.tuner_param_grid_path:
            return {}

        path = Path(self.tuner_param_grid_path)
        if not path.exists():
            logging.warning(f"Param grid file not found: {path}")
            return {}

        with open(path, 'r') as f:
            all_grids = json.load(f)
            # Return the grid for the specific model
            return all_grids.get(self.alpha_model, {})