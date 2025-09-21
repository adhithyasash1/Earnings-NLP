# ============================================
# data_loader.py - CORRECTED for Flat Structure
# ============================================
"""Data loading with caching and yfinance"""
import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import re
from pathlib import Path
import yfinance as yf

from typing import List, Dict, Set, Tuple
from config import Config
from utils import setup_logging, parse_transcript_sections, cache_result
from models import Transcript


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)

    def load_transcripts(self) -> List[Transcript]:
        """Load and validate earnings call transcripts from a single flat directory."""
        transcripts = []
        # --- MODIFICATION: Search directly in the root transcript_dir ---
        pattern = os.path.join(self.config.transcript_dir, "*.txt")

        self.logger.info(f"Searching for transcripts with pattern: {pattern}")

        filepaths = glob.glob(pattern)
        if not filepaths:
            self.logger.error(
                f"No transcript files found in '{self.config.transcript_dir}'. Check the path in your config YAML.")
            return []  # Return empty list if no files found

        for filepath in filepaths:
            filename = os.path.basename(filepath)

            # The regex for "YYYY-Mon-DD-TICKER.txt" is still correct
            match = re.match(r"(\d{4})-([A-Za-z]{3})-(\d{2})-([A-Z]+)\.txt", filename)

            if match:
                year_str, month_str, day_str, ticker = match.groups()
                date_str = f"{year_str}-{month_str}-{day_str}"

                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

                if len(text) < 1000:
                    self.logger.warning(f"Skipping short transcript: {filename}")
                    continue

                sections = parse_transcript_sections(text)

                transcripts.append(Transcript(
                    ticker=ticker,
                    date=datetime.strptime(date_str, "%Y-%b-%d"),
                    text=text,
                    filename=filename,
                    sections=sections
                ))
            else:
                self.logger.warning(f"Skipping file with incorrect format: {filename}")

        self.logger.info(f"Loaded {len(transcripts)} valid transcripts")
        return transcripts

    def fetch_prices_batch(self, tickers: Set[str], start: datetime,
                           end: datetime) -> pd.DataFrame:
        """Fetch prices for multiple tickers in batch using yfinance"""
        cache_path = Path(self.config.cache_dir) / f"prices_{start.date()}_{end.date()}.parquet"

        if cache_path.exists():
            self.logger.info(f"Loading cached prices from {cache_path}")
            return pd.read_parquet(cache_path)

        if self.config.price_source == "yfinance":
            ticker_list_str = " ".join(list(tickers))
            try:
                end_date_str = (end + timedelta(days=1)).strftime('%Y-%m-%d')
                start_date_str = start.strftime('%Y-%m-%d')

                data = yf.download(
                    tickers=ticker_list_str,
                    start=start_date_str,
                    end=end_date_str,
                    progress=False,
                    group_by='ticker'  # More robust for multiple tickers
                )

                if data.empty:
                    self.logger.warning(f"No price data returned from yfinance for {ticker_list_str}")
                    return pd.DataFrame(columns=['timestamp', 'ticker', 'close', 'volume'])

                # Format the data into the long format
                df_long_list = []
                for ticker in tickers:
                    if len(tickers) > 1:
                        ticker_df = data[ticker].copy()
                    else:  # yfinance doesn't create a multi-index for a single ticker
                        ticker_df = data.copy()

                    ticker_df = ticker_df.dropna(subset=['Close', 'Volume'])
                    if not ticker_df.empty:
                        ticker_df['ticker'] = ticker
                        df_long_list.append(ticker_df)

                if not df_long_list:
                    self.logger.warning(f"All price data was NaN for tickers: {ticker_list_str}")
                    return pd.DataFrame(columns=['timestamp', 'ticker', 'close', 'volume'])

                df_long = pd.concat(df_long_list).reset_index()

                df_long = df_long.rename(columns={
                    'Date': 'timestamp',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])
                result = df_long[['timestamp', 'ticker', 'close', 'volume']]

                Path(self.config.cache_dir).mkdir(exist_ok=True)
                result.to_parquet(cache_path)
                return result

            except Exception as e:
                self.logger.error(f"yfinance download failed: {e}")
                return pd.DataFrame(columns=['timestamp', 'ticker', 'close', 'volume'])
        else:
            raise ValueError(f"Unknown price_source: {self.config.price_source}")

    def fetch_benchmark_prices(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch benchmark (SPY) prices"""
        return self.fetch_prices_batch({self.config.benchmark_ticker}, start, end)