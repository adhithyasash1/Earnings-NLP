# ============================================
# utils.py - Helper Functions
# ============================================
"""Common utility functions"""
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import pickle
import re

from typing import Dict  # <-- ADD THIS IMPORT

def setup_logging(level: str = "INFO"):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def cache_result(cache_dir: str, key: str):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create a hashable representation of args/kwargs
            # Using pickle for complex objects, fallback to str
            try:
                arg_bytes = pickle.dumps((args, kwargs))
            except Exception:
                arg_bytes = str((args, kwargs)).encode()

            cache_key = f"{key}_{hashlib.md5(arg_bytes).hexdigest()}.pkl"
            cache_path = Path(cache_dir) / cache_key

            if cache_path.exists():
                logging.info(f"Loading cached result from {cache_path}")
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logging.warning(f"Could not load cache file {cache_path}: {e}. Re-running.")

            result = func(*args, **kwargs)

            Path(cache_dir).mkdir(exist_ok=True)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                 logging.warning(f"Could not write cache file {cache_path}: {e}")

            return result
        return wrapper
    return decorator

def parse_transcript_sections(text: str) -> Dict[str, str]:
    """Parse transcript into sections"""
    sections = {}

    # Common patterns for section breaks
    qa_patterns = [
        r"Questions and Answers",
        r"Q&A Session",
        r"Question-and-Answer Session"
    ]

    for pattern in qa_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sections['prepared_remarks'] = text[:match.start()].strip()
            sections['qa'] = text[match.start():].strip()
            return sections

    # If no Q&A found, entire text is prepared remarks
    sections['prepared_remarks'] = text
    sections['qa'] = ""
    return sections

def calculate_readability(text: str) -> float:
    """Calculate Flesch Reading Ease score"""
    # Handle empty text
    if not text:
        return 0.0

    sentences = len(re.findall(r'[.!?]+', text))
    words = len(text.split())

    # Avoid division by zero
    if sentences == 0 or words == 0:
        return 0.0

    syllables = sum([count_syllables(word) for word in text.split()])

    if syllables == 0: # Avoid division by zero if all words are e.g., numbers
        return 0.0

    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return max(0, min(100, score)) # Clamp score between 0 and 100

def count_syllables(word: str) -> int:
    """Simple syllable counter"""
    word = word.lower().strip(".:,;?!")
    if not word:
        return 0

    count = 0
    vowels = "aeiouy"

    if word[0] in vowels:
        count += 1

    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1

    if word.endswith("e"):
        count -= 1

    if count == 0:
        count = 1

    return count