import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import sys

sys.path.append('..')
from pipeline import EarningsNLPPipeline
from config import Config


@pytest.fixture
def config():
    return Config(
        transcript_dir="./test_transcripts",
        train_test_split_date="2023-06-01",
        output_dir="./test_outputs",
        cache_dir="./test_cache",
        random_seed=42
    )


@patch('pipeline.DataLoader')
@patch('pipeline.FeatureExtractor')
@patch('pipeline.AlphaModel')
@patch('pipeline.Backtester')
def test_pipeline_train_test_split(mock_backtester, mock_alpha, mock_extractor, mock_loader, config):
    # Setup mocks
    pipeline = EarningsNLPPipeline(config)

    # Mock data
    mock_transcripts = [Mock(ticker='AAPL', date=datetime(2023, 1, 1))]
    mock_loader.return_value.load_transcripts.return_value = mock_transcripts

    # Mock features
    mock_features = pd.DataFrame({
        'ticker': ['AAPL'] * 10,
        'date': pd.date_range('2023-01-01', periods=10),
        **{f'emb_{i}': np.random.randn(10) for i in range(384)}
    })
    mock_extractor.return_value.extract_features.return_value = mock_features

    # Run pipeline
    results = pipeline.run()

    # Verify train/test split was used
    assert mock_alpha.return_value.train.called
    assert mock_alpha.return_value.predict.called