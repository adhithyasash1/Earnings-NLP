# ============================================
# data_loader.py - Updated for yfinance
# ============================================
"""Data loading with caching and yfinance"""
import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import re
from pathlib import Path
import yfinance as yf  # <-- ADD THIS IMPORT

# --- ADD THESE IMPORTS ---
from typing import List, Dict, Set, Tuple
from config import Config
from utils import setup_logging, parse_transcript_sections, cache_result
from models import Transcript


# -------------------------

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        # We don't need _init_apis() for yfinance
        # self._init_apis()

    def _init_apis(self):
        """Initialize API clients (No longer needed for yfinance)"""
        pass

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
        """Fetch prices for multiple tickers in batch using yfinance"""
        cache_path = Path(self.config.cache_dir) / f"prices_{start.date()}_{end.date()}.parquet"

        if cache_path.exists():
            self.logger.info(f"Loading cached prices from {cache_path}")
            return pd.read_parquet(cache_path)

        all_data = []

        if self.config.price_source == "yfinance":
            ticker_list_str = " ".join(list(tickers))
            try:
                # yfinance 'end' is exclusive, add one day to be inclusive
                end_date_str = (end + timedelta(days=1)).strftime('%Y-%m-%d')
                start_date_str = start.strftime('%Y-%m-%d')

                data = yf.download(
                    tickers=ticker_list_str,
                    start=start_date_str,
                    end=end_date_str,
                    progress=False
                )

                if data.empty:
                    self.logger.warning(f"No price data returned from yfinance for {ticker_list_str}")
                    return pd.DataFrame(columns=['timestamp', 'ticker', 'close', 'volume'])

                # Format the data into the long format expected by the application
                # Stack the multi-level columns (e.g., 'Close', 'Volume') and tickers
                df_long = data.stack(level=1).reset_index()

                # Rename columns to match the required format
                df_long = df_long.rename(columns={
                    'Date': 'timestamp',
                    'level_1': 'ticker',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                # Ensure timestamp is datetime
                df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])

                # Select only the columns we need
                result = df_long[['timestamp', 'ticker', 'close', 'volume']]

                # Cache the result
                Path(self.config.cache_dir).mkdir(exist_ok=True)
                result.to_parquet(cache_path)
                return result

            except Exception as e:
                self.logger.error(f"yfinance download failed: {e}")
                return pd.DataFrame(columns=['timestamp', 'ticker', 'close', 'volume'])

        elif self.config.price_source == "alpaca":
            self.logger.error("Alpaca client not initialized. Set price_source to yfinance.")
            # ... (original alpaca code would be here, but is now removed)
            return pd.DataFrame(columns=['timestamp', 'ticker', 'close', 'volume'])

        else:
            raise ValueError(f"Unknown price_source: {self.config.price_source}")

    def fetch_benchmark_prices(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch benchmark (SPY) prices"""
        # This function works as-is, it just calls the new yfinance batch fetcher
        return self.fetch_prices_batch({self.config.benchmark_ticker}, start, end)