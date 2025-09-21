"""Entry point"""
import argparse
from pathlib import Path
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import Config
from pipeline import EarningsNLPPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.setup_directories()
    print(f"Running with config: {config.alpha_model}, {config.embedding_model}")

    pipeline = EarningsNLPPipeline(config)
    results = pipeline.run()

    print(f"Accuracy: {results.get('accuracy', 0):.3f}")
    print(f"IC: {results.get('information_coefficient', 0):.3f}")
    print(f"Total Return: {results.get('total_return', 0):.2%}")

if __name__ == "__main__":
    main()