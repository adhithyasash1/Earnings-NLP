# ============================================
# main.py - CLI Entry Point (Corrected)
# ============================================
"""Main entry point with CLI interface"""
import argparse
from pathlib import Path
import sys

from config import Config  # <-- ADD THIS LINE
from pipeline import EarningsNLPPipeline  # <-- ADD THIS LINE

def main():
    parser = argparse.ArgumentParser(description='Earnings NLP Alpha Pipeline')
    parser.add_argument('--config', type=str, help='Path to config file (YAML/JSON)')
    parser.add_argument('--transcript-dir', type=str, help='Override transcript directory')
    parser.add_argument('--model', type=str, choices=['logistic', 'xgboost', 'ridge'],
                        help='Override model type')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING'],
                        help='Set logging level')

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

    # Run pipeline
    pipeline = EarningsNLPPipeline(config)
    results = pipeline.run()

    print(f"\nPipeline completed successfully!")
    print(f"Final test accuracy: {results.get('accuracy', 0):.3f}")
    print(f"Information Coefficient: {results.get('information_coefficient', 0):.3f}")
    print(f"Event Sharpe: {results.get('event_sharpe', 0):.2f}")

if __name__ == "__main__":
    main()