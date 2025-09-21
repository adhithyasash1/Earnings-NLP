# README.md (Updated)
# Earnings-NLP

A minimal pipeline for analyzing earnings transcripts using NLP to generate trading signals.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py --config configs/config.yaml`

## Key Components
- Data loading from transcripts and yfinance.
- Feature extraction: Finance-tuned embeddings, full LM sentiment, section analysis.
- Label preparation with vectorization and ternary labels.
- Model training with time-series CV and tuning.
- Backtesting with enhanced metrics.

## Improvements
- Finance-specific embeddings (FinLang/finance-embeddings-investopedia).
- Full Loughran-McDonald lexicon download/integration.
- SSL fix for NLTK on macOS.
- Ternary labels and weighted sections.