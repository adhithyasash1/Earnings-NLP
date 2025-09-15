"""Data models and schemas"""
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from typing import Optional, Dict

@dataclass
class Transcript:
    ticker: str
    date: datetime
    text: str
    filename: str
    sections: Optional[Dict[str, str]] = None  # prepared_remarks, qa

@dataclass
class Features:
    ticker: str
    date: datetime
    embeddings: np.ndarray
    embeddings_remarks: Optional[np.ndarray] = None
    embeddings_qa: Optional[np.ndarray] = None
    sentiment: Optional[float] = None
    sentiment_remarks: Optional[float] = None
    sentiment_qa: Optional[float] = None
    word_count: Optional[int] = None
    readability: Optional[float] = None

@dataclass
class Signal:
    ticker: str
    date: datetime
    direction: int  # -1, 0, 1
    confidence: float
    predicted_return: Optional[float] = None