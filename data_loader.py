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
        pattern = Path(self.config.transcript_dir) / "*.txt"
        filepaths = glob.glob(str(pattern))
        for filepath in filepaths:
            filename = Path(filepath).name
            match = re.match(r"(\d{4})-([A-Za-z]{3})-(\d{2})-([A-Z]+)\.txt", filename)
            if match:
                year, month, day, ticker = match.groups()
                date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%b-%d")
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                if len(text) < 1000:
                    continue
                sections = parse_transcript_sections(text)
                transcripts.append(Transcript(ticker=ticker, date=date, text=text, filename=filename, sections=sections))
        return transcripts

    def fetch_prices_batch(self, tickers: Set[str], start: datetime, end: datetime) -> pd.DataFrame:
        cache_path = Path(self.config.cache_dir) / f"prices_{start.date()}_{end.date()}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        data = yf.download(" ".join(tickers), start=start, end=end + timedelta(days=1), auto_adjust=True)
        df_list = []
        for ticker in tickers:
            if len(tickers) > 1:
                ticker_df = data[('Close', ticker)]
            else:
                ticker_df = data['Close']
            ticker_df = pd.DataFrame(ticker_df).reset_index()
            ticker_df['ticker'] = ticker
            df_list.append(ticker_df)
        df = pd.concat(df_list).rename(columns={'Date': 'timestamp', 'Close': 'close'})
        df.to_parquet(cache_path)
        return df[['timestamp', 'ticker', 'close']]

    def fetch_benchmark_prices(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self.fetch_prices_batch({self.config.benchmark_ticker}, start, end)