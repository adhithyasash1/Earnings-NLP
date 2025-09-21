"""Data models"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict

@dataclass
class Transcript:
    ticker: str
    date: datetime
    text: str
    filename: str
    sections: Optional[Dict[str, str]] = None