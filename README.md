# Earnings-NLP

A minimal pipeline for analyzing earnings transcripts using NLP to generate trading signals.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py --config configs/config.yaml`

## Key Components
- Data loading from transcripts and yfinance.
- Feature extraction: Embeddings and sentiment.
- Label preparation with leakage prevention.
- Model training with time-series CV.
- Backtesting with metrics.