"""Entry point"""
import argparse
from pathlib import Path
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import Config
from pipeline import EarningsNLPPipeline


def main():
    parser = argparse.ArgumentParser(description='Earnings NLP Pipeline')
    parser.add_argument('--config', type=str, default="configs/config.yaml",
                        help='Path to configuration file')
    args = parser.parse_args()

    try:
        # Load and validate configuration
        if not Path(args.config).exists():
            print(f"Error: Configuration file {args.config} not found")
            sys.exit(1)

        config = Config.from_yaml(args.config)
        config.setup_directories()

        print(f"Running Earnings NLP Pipeline")
        print(f"Configuration: {config.alpha_model} model with {config.embedding_model} embeddings")
        print(f"Transcript directory: {config.transcript_dir}")
        print(f"Output directory: {config.output_dir}")
        print("-" * 50)

        # Check if transcript directory exists
        if not Path(config.transcript_dir).exists():
            print(f"Error: Transcript directory {config.transcript_dir} does not exist")
            print("Please ensure you have transcript files in the specified directory")
            sys.exit(1)

        pipeline = EarningsNLPPipeline(config)
        results = pipeline.run()

        # Display key results
        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)

        if config.prediction_target == "direction":
            print(f"Accuracy: {results.get('accuracy', 0):.3f}")
            print(f"Precision: {results.get('precision', 0):.3f}")
            print(f"Recall: {results.get('recall', 0):.3f}")
            print(f"F1 Score: {results.get('f1', 0):.3f}")

        print(f"Information Coefficient: {results.get('information_coefficient', 0):.3f}")
        print(f"IC p-value: {results.get('ic_pvalue', 0):.4f}")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Event Sharpe Ratio: {results.get('event_sharpe', 0):.2f}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")

        print(f"\nDetailed report saved to: {config.output_dir}/backtest_report.txt")
        print(f"Model saved to: {config.output_dir}/model.pkl")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()