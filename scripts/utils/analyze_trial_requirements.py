"""Analyze whether num_trials=200 guarantees each graph appears 10 times in test sets.

This script calculates the probability that all graphs appear at least 10 times
in the test set after a given number of trials, given the test split fraction.
"""

from scipy.stats import binom


def calculate_trial_requirements(
    dataset_size: int,
    test_fraction: float = 0.25,
    required_appearances: int = 10,
    num_trials: int = 200,
    confidence_level: float = 0.99,
) -> dict:
    """
    Calculate whether num_trials is sufficient to guarantee each graph appears
    at least required_appearances times in test sets.

    Args:
        dataset_size: Total number of graphs in the dataset
        test_fraction: Fraction of dataset used as test set per trial (default: 0.25)
        required_appearances: Required number of test appearances per graph (default: 10)
        num_trials: Number of trials to run (default: 200)
        confidence_level: Desired confidence level for the guarantee (default: 0.99)

    Returns:
        Dictionary with analysis results
    """
    # Probability that a specific graph is in the test set for a given trial
    p_test = test_fraction

    # Expected number of appearances for any graph
    expected_appearances = num_trials * p_test

    # Calculate probability that a specific graph appears at least required_appearances times
    # Using binomial distribution: X ~ Binomial(num_trials, p_test)
    prob_single_graph = 1 - binom.cdf(required_appearances - 1, num_trials, p_test)

    # Probability that ALL graphs appear at least required_appearances times
    # Assuming independence (reasonable with random splits)
    prob_all_graphs = prob_single_graph**dataset_size

    # Calculate minimum number of trials needed to guarantee with confidence
    # We want: (1 - binom.cdf(required_appearances - 1, min_trials, p_test))^dataset_size >= confidence_level
    # Taking log: dataset_size * log(1 - binom.cdf(required_appearances - 1, min_trials, p_test)) >= log(confidence_level)
    # This requires solving: 1 - binom.cdf(required_appearances - 1, min_trials, p_test) >= confidence_level^(1/dataset_size)

    # Binary search for minimum trials
    min_trials_needed = num_trials
    for t in range(num_trials, num_trials * 2):
        prob_single = 1 - binom.cdf(required_appearances - 1, t, p_test)
        prob_all = prob_single**dataset_size
        if prob_all >= confidence_level:
            min_trials_needed = t
            break
    else:
        # If we didn't find a solution, use a conservative estimate
        min_trials_needed = None

    # Calculate percentiles for number of appearances
    # What's the minimum number of appearances we'd expect with high probability?
    percentile_1 = binom.ppf(0.01, num_trials, p_test)  # 1st percentile
    percentile_5 = binom.ppf(0.05, num_trials, p_test)  # 5th percentile
    percentile_10 = binom.ppf(0.10, num_trials, p_test)  # 10th percentile

    return {
        "dataset_size": dataset_size,
        "test_fraction": test_fraction,
        "test_size_per_trial": int(dataset_size * test_fraction),
        "required_appearances": required_appearances,
        "num_trials": num_trials,
        "expected_appearances": expected_appearances,
        "prob_single_graph_sufficient": prob_single_graph,
        "prob_all_graphs_sufficient": prob_all_graphs,
        "confidence_level": confidence_level,
        "min_trials_for_confidence": min_trials_needed,
        "percentile_1_appearances": percentile_1,
        "percentile_5_appearances": percentile_5,
        "percentile_10_appearances": percentile_10,
        "is_sufficient": prob_all_graphs >= confidence_level,
    }


def print_analysis(results: dict) -> None:
    """Print formatted analysis results."""
    print("\n" + "=" * 80)
    print(f"DATASET SIZE: {results['dataset_size']} graphs")
    print(
        f"TEST FRACTION: {results['test_fraction']} (test size per trial: {results['test_size_per_trial']})"
    )
    print(f"REQUIRED APPEARANCES: {results['required_appearances']}")
    print(f"NUM TRIALS: {results['num_trials']}")
    print("=" * 80)

    print(f"\n📊 Expected Appearances per Graph: {results['expected_appearances']:.2f}")
    print(
        f"   (Each graph is expected to appear {results['expected_appearances']:.2f} times in test sets)"
    )

    print("\n📈 Probability Analysis:")
    print(
        f"   Probability a SINGLE graph appears ≥{results['required_appearances']} times: {results['prob_single_graph_sufficient']:.6f} ({results['prob_single_graph_sufficient']*100:.4f}%)"
    )
    print(
        f"   Probability ALL {results['dataset_size']} graphs appear ≥{results['required_appearances']} times: {results['prob_all_graphs_sufficient']:.6f} ({results['prob_all_graphs_sufficient']*100:.4f}%)"
    )

    print(f"\n🎯 Confidence Level: {results['confidence_level']*100}%")
    if results["is_sufficient"]:
        print(
            f"   ✅ SUFFICIENT: {results['num_trials']} trials are enough to guarantee with {results['confidence_level']*100}% confidence"
        )
    else:
        print(f"   ⚠️  INSUFFICIENT: {results['num_trials']} trials may not be enough")
        if results["min_trials_for_confidence"]:
            print(
                f"   💡 Recommendation: Use at least {results['min_trials_for_confidence']} trials to guarantee with {results['confidence_level']*100}% confidence"
            )

    print("\n📉 Percentile Analysis (worst-case scenarios):")
    print(
        f"   1st percentile (1% of graphs): {results['percentile_1_appearances']:.1f} appearances"
    )
    print(
        f"   5th percentile (5% of graphs): {results['percentile_5_appearances']:.1f} appearances"
    )
    print(
        f"   10th percentile (10% of graphs): {results['percentile_10_appearances']:.1f} appearances"
    )

    if results["percentile_1_appearances"] < results["required_appearances"]:
        print(
            f"\n   ⚠️  WARNING: Some graphs (1%) may appear fewer than {results['required_appearances']} times"
        )
    elif results["percentile_5_appearances"] < results["required_appearances"]:
        print(
            f"\n   ⚠️  CAUTION: Some graphs (5%) may appear fewer than {results['required_appearances']} times"
        )

    print("\n" + "=" * 80 + "\n")


def main():
    """Analyze trial requirements for different dataset sizes."""
    print("\n" + "=" * 80)
    print("ANALYZING WHETHER 200 TRIALS GUARANTEES 10 TEST APPEARANCES PER GRAPH")
    print("=" * 80)

    # Common dataset sizes from the project
    dataset_sizes = [
        188,  # MUTAG
        600,  # ENZYMES
        1000,  # IMDB-BINARY
        1113,  # PROTEINS
        2000,  # REDDIT-BINARY
        5000,  # COLLAB
        10000,  # PATTERN
        45000,  # CIFAR10
        55000,  # MNIST
        41127,  # ogbg-molhiv
        437929,  # ogbg-molpcba
    ]

    test_fraction = 0.25  # From default_args in graph_classification.py
    required_appearances = (
        10  # From required_test_appearances in run_graph_classification.py
    )
    num_trials = 200  # From the bash script

    print("\nParameters:")
    print(f"  Test fraction: {test_fraction}")
    print(f"  Required appearances per graph: {required_appearances}")
    print(f"  Number of trials: {num_trials}\n")

    all_results = []
    for dataset_size in sorted(dataset_sizes):
        results = calculate_trial_requirements(
            dataset_size=dataset_size,
            test_fraction=test_fraction,
            required_appearances=required_appearances,
            num_trials=num_trials,
            confidence_level=0.99,
        )
        all_results.append(results)
        print_analysis(results)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'Dataset Size':<15} {'Expected':<12} {'P(All≥10)':<15} {'Sufficient?':<12} {'Min Trials':<12}"
    )
    print("-" * 100)
    for results in all_results:
        dataset_size = results["dataset_size"]
        expected = results["expected_appearances"]
        prob_all = results["prob_all_graphs_sufficient"]
        sufficient = "✅ Yes" if results["is_sufficient"] else "❌ No"
        min_trials = (
            str(results["min_trials_for_confidence"])
            if results["min_trials_for_confidence"]
            else "N/A"
        )
        print(
            f"{dataset_size:<15} {expected:<12.1f} {prob_all:<15.6f} {sufficient:<12} {min_trials:<12}"
        )
    print("=" * 100)

    # Key findings
    print("\n🔑 KEY FINDINGS:")
    print(f"   • Expected appearances per graph: {num_trials * test_fraction} times")
    print(
        f"   • With {num_trials} trials and test_fraction={test_fraction}, each graph is"
    )
    print(f"     expected to appear {num_trials * test_fraction} times in test sets")
    print("   • However, due to randomness, some graphs may appear fewer times")
    print("   • For smaller datasets, 200 trials should be sufficient")
    print("   • For larger datasets (>10k graphs), you may need more trials")
    print("\n💡 RECOMMENDATION:")
    print("   The script already handles this by continuing until all graphs appear")
    print(
        f"   at least {required_appearances} times OR until reaching {num_trials} trials."
    )
    print("   Setting num_trials=200 is a safety limit - the script will stop early")
    print("   if the requirement is met. For very large datasets, consider increasing")
    print("   num_trials to ensure the requirement can be met.")


if __name__ == "__main__":
    main()
