# ============================================
# data_loader.py - Enhanced Data Collection
# ============================================
"""Data loading with caching and batch fetching"""
import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import re
from pathlib import Path

# --- ADD THESE IMPORTS ---
from typing import List, Dict, Set, Tuple
from config import Config
from utils import setup_logging, parse_transcript_sections, cache_result
from models import Transcript  # <-- This is the main fix


# -------------------------

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self._init_apis()

    def _init_apis(self):
        """Initialize API clients"""
        if self.config.price_source == "alpaca":
            from alpaca.data import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            self.stock_client = StockHistoricalDataClient(
                self.config.alpaca_key,
                self.config.alpaca_secret
            )
            self.TimeFrame = TimeFrame
            self.StockBarsRequest = StockBarsRequest

    def load_transcripts(self) -> List[Transcript]:
        """Load and validate earnings call transcripts"""
        transcripts = []
        pattern = os.path.join(self.config.transcript_dir, "*.txt")

        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            match = re.match(r"([A-Z]+)_(\d{4}-\d{2}-\d{2})\.txt", filename)

            if match:
                ticker, date_str = match.groups()
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Validate transcript
                if len(text) < 1000:  # Skip very short files
                    self.logger.warning(f"Skipping short transcript: {filename}")
                    continue

                # Parse sections
                sections = parse_transcript_sections(text)

                transcripts.append(Transcript(
                    ticker=ticker,
                    date=datetime.strptime(date_str, "%Y-%m-%d"),
                    text=text,
                    filename=filename,
                    sections=sections
                ))

        self.logger.info(f"Loaded {len(transcripts)} valid transcripts")
        return transcripts

    def fetch_prices_batch(self, tickers: Set[str], start: datetime,
                           end: datetime) -> pd.DataFrame:
        """Fetch prices for multiple tickers in batch"""
        cache_path = Path(self.config.cache_dir) / f"prices_{start.date()}_{end.date()}.parquet"

        if cache_path.exists():
            self.logger.info(f"Loading cached prices from {cache_path}")
            return pd.read_parquet(cache_path)

        all_data = []

        if self.config.price_source == "alpaca":
            ticker_list = list(tickers)
            for i in range(0, len(ticker_list), 10):
                ticker_batch = ticker_list[i:i + 10]
                request = self.StockBarsRequest(
                    symbol_or_symbols=ticker_batch,
                    timeframe=self.TimeFrame.Day,
                    start=start,
                    end=end
                )
                bars = self.stock_client.get_stock_bars(request)
                df = bars.df.reset_index()
                df.columns = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
                all_data.append(df[['timestamp', 'ticker', 'close', 'volume']])

        if not all_data:
            self.logger.warning("No price data returned from API.")
            return pd.DataFrame(columns=['timestamp', 'ticker', 'close', 'volume'])

        result = pd.concat(all_data, ignore_index=True)
        result['timestamp'] = pd.to_datetime(result['timestamp'])

        # Cache the result
        Path(self.config.cache_dir).mkdir(exist_ok=True)
        result.to_parquet(cache_path)

        return result

    def fetch_benchmark_prices(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch benchmark (SPY) prices"""
        return self.fetch_prices_batch({self.config.benchmark_ticker}, start, end)