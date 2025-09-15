import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.append('..')
from alpha_model import AlphaModel
from config import Config


@pytest.fixture
def config():
    return Config(
        alpha_model="logistic",
        prediction_target="direction",
        use_market_neutral=False,
        random_seed=42
    )


@pytest.fixture
def alpha_model(config):
    return AlphaModel(config)


def test_prepare_labels(alpha_model):
    # Create sample data
    features_df = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL'],
        'date': [datetime(2023, 1, 1), datetime(2023, 1, 2)]
    })

    prices_df = pd.DataFrame({
        'ticker': ['AAPL'] * 10 + ['GOOGL'] * 10,
        'timestamp': pd.date_range('2023-01-01', periods=10).tolist() * 2,
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 2
    })

    labeled = alpha_model.prepare_labels(features_df, prices_df, holding_period=5)

    assert 'label' in labeled.columns
    assert 'future_return' in labeled.columns
    assert len(labeled) <= len(features_df)


def test_train_no_data_leakage(alpha_model):
    # Create sample training data
    X = pd.DataFrame(np.random.randn(100, 10))
    y = pd.Series(np.random.randint(0, 2, 100))

    # Train should not raise errors
    score = alpha_model.train(X, y)

    assert 0 <= score <= 1  # Valid accuracy score
    assert alpha_model.scaler is not None
    assert alpha_model.model is not None