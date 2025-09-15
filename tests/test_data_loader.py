import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
import sys

sys.path.append('..')
from data_loader import DataLoader
from config import Config


@pytest.fixture
def config():
    return Config(
        transcript_dir="./test_transcripts",
        price_source="alpaca",
        alpaca_key="test_key",
        alpaca_secret="test_secret"
    )


@pytest.fixture
def data_loader(config):
    with patch('data_loader.StockHistoricalDataClient'):
        return DataLoader(config)


def test_load_transcripts(data_loader, tmp_path):
    # Create test transcript files
    test_file = tmp_path / "AAPL_2023-01-15.txt"
    test_file.write_text("This is a test earnings call transcript with sufficient length " * 50)

    data_loader.config.transcript_dir = str(tmp_path)
    transcripts = data_loader.load_transcripts()

    assert len(transcripts) == 1
    assert transcripts[0].ticker == "AAPL"
    assert transcripts[0].date == datetime(2023, 1, 15)


def test_skip_short_transcripts(data_loader, tmp_path):
    # Create short transcript that should be skipped
    test_file = tmp_path / "AAPL_2023-01-15.txt"
    test_file.write_text("Too short")

    data_loader.config.transcript_dir = str(tmp_path)
    transcripts = data_loader.load_transcripts()

    assert len(transcripts) == 0  # Should skip short file