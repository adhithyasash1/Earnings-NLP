"""Data loading"""
import glob
import pandas as pd
from datetime import datetime, timedelta
import re
from pathlib import Path
import yfinance as yf

from typing import List, Set
from config import Config
from utils import setup_logging, parse_transcript_sections
from models import Transcript


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)

    def load_transcripts(self) -> List[Transcript]:
        transcripts = []
        transcript_dir = Path(self.config.transcript_dir)

        if not transcript_dir.exists():
            self.logger.error(f"Transcript directory {transcript_dir} does not exist")
            return transcripts

        pattern = transcript_dir / "*.txt"
        filepaths = glob.glob(str(pattern))

        self.logger.info(f"Found {len(filepaths)} transcript files")

        for filepath in filepaths:
            try:
                filename = Path(filepath).name
                self.logger.debug(f"Processing file: {filename}")

                # Try to parse filename in format: YYYY-Mon-DD-TICKER.txt
                match = re.match(r"(\d{4})-([A-Za-z]{3})-(\d{2})-([A-Z]+)\.txt", filename)
                if not match:
                    self.logger.warning(f"Skipping file with invalid format: {filename}")
                    continue

                year, month, day, ticker = match.groups()
                try:
                    date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%b-%d")
                except ValueError as e:
                    self.logger.warning(f"Invalid date format in {filename}: {e}")
                    continue

                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()

                if len(text) < 1000:
                    self.logger.warning(f"Skipping {filename}: text too short ({len(text)} chars)")
                    continue

                sections = parse_transcript_sections(text)
                transcript = Transcript(
                    ticker=ticker,
                    date=date,
                    text=text,
                    filename=filename,
                    sections=sections
                )
                transcripts.append(transcript)
                self.logger.debug(f"Added transcript: {ticker} on {date.date()}")

            except Exception as e:
                self.logger.error(f"Error processing {filepath}: {e}")
                continue

        self.logger.info(f"Successfully loaded {len(transcripts)} transcripts")
        return transcripts

    def fetch_prices_batch(self, tickers: Set[str], start: datetime, end: datetime) -> pd.DataFrame:
        cache_path = Path(self.config.cache_dir) / f"prices_{start.date()}_{end.date()}.parquet"

        # Try loading from cache first
        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                # Check if cached data contains all required tickers
                cached_tickers = set(cached_df['ticker'].unique())
                if tickers.issubset(cached_tickers):
                    self.logger.info(f"Loaded prices from cache: {cache_path}")
                    return cached_df[cached_df['ticker'].isin(tickers)]
                else:
                    self.logger.info("Cache doesn't contain all required tickers, fetching fresh data")
            except Exception as e:
                self.logger.warning(f"Error reading cache {cache_path}: {e}")

        # Ensure cache directory exists
        Path(self.config.cache_dir).mkdir(exist_ok=True)

        tickers_list = list(tickers)
        self.logger.info(f"Fetching prices for tickers: {tickers_list}")

        try:
            # Add a small buffer to end date to ensure we get the data we need
            end_buffered = end + timedelta(days=1)

            # Try downloading each ticker individually as a fallback
            df_list = []

            for ticker in tickers_list:
                try:
                    self.logger.debug(f"Downloading data for {ticker}")
                    ticker_data = yf.download(ticker, start=start, end=end_buffered, auto_adjust=True, progress=False)

                    if ticker_data.empty:
                        self.logger.warning(f"No data found for ticker {ticker}")
                        continue

                    # Debug the structure
                    self.logger.debug(f"Data shape for {ticker}: {ticker_data.shape}")
                    self.logger.debug(f"Data columns for {ticker}: {ticker_data.columns.tolist()}")

                    # Ensure we have the required columns
                    if 'Close' not in ticker_data.columns:
                        self.logger.warning(f"No Close column for {ticker}. Available: {ticker_data.columns.tolist()}")
                        continue

                    # Work directly with the DataFrame - don't reset index yet
                    close_series = ticker_data['Close'].dropna()

                    if close_series.empty:
                        self.logger.warning(f"Empty close data for {ticker}")
                        continue

                    # Create new dataframe using the series index (dates) and values
                    ticker_df = pd.DataFrame(index=close_series.index)
                    ticker_df['timestamp'] = close_series.index
                    ticker_df['close'] = close_series.values
                    ticker_df['ticker'] = ticker

                    # Reset index to make it a regular dataframe
                    ticker_df = ticker_df.reset_index(drop=True)

                    # Remove any invalid data
                    ticker_df = ticker_df.dropna()
                    ticker_df = ticker_df[ticker_df['close'] > 0]

                    if not ticker_df.empty:
                        df_list.append(ticker_df)
                        self.logger.debug(f"Successfully downloaded {len(ticker_df)} records for {ticker}")
                    else:
                        self.logger.warning(f"All data filtered out for {ticker}")

                except Exception as e:
                    self.logger.error(f"Error downloading data for {ticker}: {e}")
                    import traceback
                    self.logger.debug(f"Full traceback for {ticker}: {traceback.format_exc()}")
                    continue

            if not df_list:
                self.logger.error("No valid data downloaded for any ticker")
                return pd.DataFrame()

            df = pd.concat(df_list, ignore_index=True)


            # Clean and validate the data
            df = df.dropna()
            df = df[df['close'] > 0]  # Remove invalid prices

            if df.empty:
                self.logger.warning("All price data was filtered out")
                return df

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by ticker and timestamp
            df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

            self.logger.info(f"Fetched {len(df)} price records for {df['ticker'].nunique()} tickers")

            # Save to cache
            try:
                df.to_parquet(cache_path)
                self.logger.info(f"Cached prices to {cache_path}")
            except Exception as e:
                self.logger.warning(f"Could not cache prices: {e}")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}")
            return pd.DataFrame()

    def fetch_benchmark_prices(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch benchmark prices using the same logic as regular prices"""
        return self.fetch_prices_batch({self.config.benchmark_ticker}, start, end)