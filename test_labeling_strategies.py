#!/usr/bin/env python3
"""
Test different labeling strategies to handle class imbalance
"""
import sys

sys.path.append('.')

from config import Config
from pipeline import EarningsNLPPipeline
from alpha_model import AlphaModel


def test_strategy(strategy_name, config_path="configs/robust_xgboost.yaml"):
    """Test a specific labeling strategy"""
    print(f"\n{'=' * 60}")
    print(f"TESTING STRATEGY: {strategy_name}")
    print(f"{'=' * 60}")

    # Load config
    config = Config.from_yaml(config_path)

    # Create pipeline and modify strategy
    pipeline = EarningsNLPPipeline(config)
    pipeline.alpha_model.labeling_strategy = strategy_name

    try:
        results = pipeline.run()

        print(f"\nSTRATEGY '{strategy_name}' RESULTS:")
        print(f"  Accuracy: {results.get('accuracy', 0):.3f}")
        print(f"  Information Coefficient: {results.get('information_coefficient', 0):.3f}")
        print(f"  Event Sharpe: {results.get('event_sharpe', 0):.2f}")
        print(f"  Total Return: {results.get('total_return', 0):.2%}")
        print(f"  Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"  Train Samples: {results.get('train_samples', 0)}")
        print(f"  Test Samples: {results.get('test_samples', 0)}")

        return results

    except Exception as e:
        print(f"Strategy '{strategy_name}' FAILED: {e}")
        return None


def main():
    """Test all labeling strategies"""
    strategies = [
        "relative_quantile",  # Most robust - uses top/bottom 40%
        "median_split",  # Simple median split
        "time_relative"  # Compare to rolling historical median
    ]

    all_results = {}

    for strategy in strategies:
        result = test_strategy(strategy)
        if result:
            all_results[strategy] = result

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'=' * 60}")

    if all_results:
        print(f"{'Strategy':<20} {'Accuracy':<10} {'IC':<8} {'Sharpe':<8} {'Samples':<10}")
        print("-" * 60)

        for strategy, results in all_results.items():
            accuracy = results.get('accuracy', 0)
            ic = results.get('information_coefficient', 0)
            sharpe = results.get('event_sharpe', 0)
            samples = f"{results.get('train_samples', 0)}/{results.get('test_samples', 0)}"

            print(f"{strategy:<20} {accuracy:<10.3f} {ic:<8.3f} {sharpe:<8.2f} {samples:<10}")
    else:
        print("No strategies succeeded!")

    # Recommendation
    if all_results:
        best_strategy = max(all_results.keys(),
                            key=lambda k: all_results[k].get('accuracy', 0) *
                                          abs(all_results[k].get('information_coefficient', 0)))
        print(f"\nRECOMMENDED STRATEGY: {best_strategy}")

    print(f"\nTo use the best strategy, modify your alpha_model.py:")
    print(f"  self.labeling_strategy = '{best_strategy}'")


if __name__ == "__main__":
    main()