#!/usr/bin/env python3
"""
Debug script to identify data quality issues in the earnings NLP pipeline
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append('.')

from config import Config
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from alpha_model import AlphaModel


def debug_pipeline():
    """Debug the entire pipeline step by step"""

    print("EARNINGS NLP PIPELINE DEBUGGING")
    print("=" * 50)

    # Load config
    config = Config.from_yaml("configs/xgboost.yaml")
    print(f"Config loaded: {config.transcript_dir}")

    # Initialize components
    data_loader = DataLoader(config)
    feature_extractor = FeatureExtractor(config)
    alpha_model = AlphaModel(config)

    print("\n1. LOADING TRANSCRIPTS...")
    transcripts = data_loader.load_transcripts()
    print(f"Loaded {len(transcripts)} transcripts")

    if len(transcripts) == 0:
        print("ERROR: No transcripts found!")
        return

    # Show sample transcript info
    sample = transcripts[0]
    print(f"Sample: {sample.ticker} on {sample.date.date()}")
    print(f"Text length: {len(sample.text)} characters")

    print("\n2. EXTRACTING FEATURES...")
    try:
        features_df = feature_extractor.extract_features(transcripts[:5])  # Test with few samples
        print(f"Features extracted: {features_df.shape}")
        print(f"Feature columns: {list(features_df.columns)[:10]}...")
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return

    print("\n3. LOADING PRICE DATA...")
    from datetime import timedelta
    tickers = set(t.ticker for t in transcripts)
    min_date = min(t.date for t in transcripts) - timedelta(days=5)
    max_date = max(t.date for t in transcripts) + timedelta(days=30)

    prices_df = data_loader.fetch_prices_batch(list(tickers)[:5], min_date, max_date)
    print(f"Price data loaded: {prices_df.shape}")

    if len(prices_df) == 0:
        print("ERROR: No price data found!")
        return

    # Sample price data
    sample_ticker = list(tickers)[0]
    sample_prices = prices_df[prices_df['ticker'] == sample_ticker].head()
    print(f"Sample prices for {sample_ticker}:")
    print(sample_prices)

    print("\n4. DEBUGGING RETURN CALCULATION...")

    # Test with one transcript
    test_features = features_df.head(3)

    print("Testing return calculation...")
    labeled_data = alpha_model.prepare_labels(test_features, prices_df, holding_period=5)

    if len(labeled_data) == 0:
        print("ERROR: No labeled data generated!")
        return

    print(f"Labeled data generated: {labeled_data.shape}")

    # Check returns
    if 'future_return' in labeled_data.columns:
        returns = labeled_data['future_return']
        print(f"\nReturn Analysis:")
        print(f"  Count: {len(returns)}")
        print(f"  Mean: {returns.mean():.4f} ({returns.mean() * 100:.2f}%)")
        print(f"  Median: {returns.median():.4f}")
        print(f"  Std: {returns.std():.4f}")
        print(f"  Min: {returns.min():.4f} ({returns.min() * 100:.2f}%)")
        print(f"  Max: {returns.max():.4f} ({returns.max() * 100:.2f}%)")
        print(f"  Positive: {(returns > 0).sum()}/{len(returns)} ({(returns > 0).mean() * 100:.1f}%)")

        # Show detailed breakdown
        print(f"\nDetailed Sample:")
        sample_data = labeled_data[['ticker', 'date', 'entry_price', 'exit_price', 'future_return']].head()
        print(sample_data)

        # Check for data quality issues
        if returns.mean() < -0.1:
            print(f"\n⚠️  WARNING: Average return is {returns.mean() * 100:.1f}% - very negative!")
            print("   Possible issues:")
            print("   - Stock splits not adjusted")
            print("   - Wrong price data")
            print("   - Calculation error")

        if (returns > 0).mean() < 0.3:
            print(f"\n⚠️  WARNING: Only {(returns > 0).mean() * 100:.1f}% positive returns")
            print("   This suggests systematic data issues")

    else:
        print("ERROR: No future_return column found!")

    print("\n5. TESTING MODEL TRAINING...")

    # Prepare simple training data
    feature_cols = [c for c in labeled_data.columns if c.startswith('emb_') or c in ['sentiment', 'word_count']]
    feature_cols = feature_cols[:10]  # Limit features

    if len(feature_cols) == 0:
        print("ERROR: No feature columns found!")
        return

    X = labeled_data[feature_cols].fillna(0)
    y = labeled_data['label'] if 'label' in labeled_data.columns else (labeled_data['future_return'] > 0).astype(int)

    print(f"Training data: X={X.shape}, y={len(y)}")
    print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

    if len(pd.Series(y).value_counts()) < 2:
        print("ERROR: Only one class in labels!")
        return

    try:
        score = alpha_model.train(X, y)
        print(f"Training completed. CV Score: {score:.4f}")

        # Test prediction
        pred_probs, pred_labels = alpha_model.predict(X)
        print(f"Prediction test successful. Unique predictions: {len(np.unique(pred_labels))}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("DEBUGGING COMPLETE")
    print("=" * 50)


def quick_data_check():
    """Quick check of data quality"""

    print("QUICK DATA QUALITY CHECK")
    print("=" * 30)

    # Check if transcript directory exists
    config = Config()
    transcript_dir = Path(config.transcript_dir)

    if not transcript_dir.exists():
        print(f"❌ Transcript directory not found: {transcript_dir}")
        return

    transcript_files = list(transcript_dir.glob("*.txt"))
    print(f"✅ Found {len(transcript_files)} transcript files")

    if len(transcript_files) > 0:
        sample_file = transcript_files[0]
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Sample file readable: {len(content)} characters")

    # Check cache directory
    cache_dir = Path(config.cache_dir)
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.parquet"))
        print(f"✅ Found {len(cache_files)} cached files")
    else:
        print("ℹ️  Cache directory not found (will be created)")

    print("Data check complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick data check only")
    args = parser.parse_args()

    if args.quick:
        quick_data_check()
    else:
        debug_pipeline()