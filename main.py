# ============================================
# main.py - Enhanced CLI Entry Point
# ============================================
"""Main entry point with CLI interface"""
import argparse
from pathlib import Path
import sys
import pandas as pd

from config import Config
from pipeline import EarningsNLPPipeline


def main():
    parser = argparse.ArgumentParser(description='Earnings NLP Alpha Pipeline')
    parser.add_argument('--config', type=str, help='Path to config file (YAML/JSON)')
    parser.add_argument('--transcript-dir', type=str, help='Override transcript directory')
    parser.add_argument('--model', type=str, choices=['logistic', 'xgboost', 'ridge'],
                        help='Override model type')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING'],
                        help='Set logging level')
    parser.add_argument('--split-date', type=str, help='Override train/test split date (YYYY-MM-DD)')
    parser.add_argument('--auto-split', action='store_true',
                        help='Automatically determine split date from data')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config = Config.from_yaml(args.config)
        elif args.config.endswith('.json'):
            config = Config.from_json(args.config)
        else:
            print(f"Unknown config format: {args.config}")
            sys.exit(1)
    else:
        config = Config()  # Use defaults and .env

    # Override with CLI arguments
    if args.transcript_dir:
        config.transcript_dir = args.transcript_dir
    if args.model:
        config.alpha_model = args.model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.log_level:
        config.log_level = args.log_level
    if args.split_date:
        config.train_test_split_date = args.split_date

    # Set auto-split flag (this will be handled by the pipeline)
    if args.auto_split:
        # Force the pipeline to recalculate split date
        config.train_test_split_date = "9999-01-01"

    print(f"\n=== Earnings NLP Pipeline ===")
    print(f"Transcript directory: {config.transcript_dir}")
    print(f"Model: {config.alpha_model}")
    print(f"Prediction target: {config.prediction_target}")
    print(f"Output directory: {config.output_dir}")
    print(f"Split date: {config.train_test_split_date}")
    print(f"Market neutral: {config.use_market_neutral}")
    print("=" * 30)

    try:
        # Run pipeline
        pipeline = EarningsNLPPipeline(config)
        results = pipeline.run()

        print(f"\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print(f"Final test accuracy: {results.get('accuracy', 0):.3f}")
        print(f"Information Coefficient: {results.get('information_coefficient', 0):.3f}")
        print(f"IC P-value: {results.get('ic_pvalue', 1):.4f}")
        print(f"Event Sharpe Ratio: {results.get('event_sharpe', 0):.2f}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"\nResults saved to: {config.output_dir}")

    except Exception as e:
        print(f"\n=== PIPELINE FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()