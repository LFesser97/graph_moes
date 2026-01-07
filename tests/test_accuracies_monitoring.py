"""Test suite for accuracy monitoring and per-graph tracking functionality.

This module tests the machinery used to:
- Track correct/incorrect classifications per graph across multiple trials
- Compute average accuracy per graph
- Ensure graphs appear at least 10 times in test sets
- Save and load graph_dict pickle files
- Generate accuracy plots

These tests serve as documentation for how the accuracy monitoring system works.
"""

import os
import pickle
import tempfile

import numpy as np
import pytest

from graph_moes.experiments.track_avg_accuracy import (
    compute_average_per_graph,
    load_and_plot_average_per_graph,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_graph_dict():
    """Create a sample graph_dict for testing.

    Format: {graph_idx: [correctness_values...]}
    - correctness_values are 0 (incorrect) or 1 (correct)
    - Example: {0: [1, 1, 0, 1], 1: [1, 0, 1, 1, 1]} means:
      - Graph 0: correct 3/4 times (75% accuracy)
      - Graph 1: correct 4/5 times (80% accuracy)
    """
    return {
        0: [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],  # 8/10 = 0.8 accuracy
        1: [1, 0, 1, 1, 1, 0, 1, 1, 1, 0],  # 7/10 = 0.7 accuracy
        2: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10/10 = 1.0 accuracy (perfect)
        3: [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],  # 4/10 = 0.4 accuracy
        4: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],  # 9/10 = 0.9 accuracy
    }


@pytest.fixture
def sample_test_appearances():
    """Create a sample test_appearances dictionary.

    Format: {graph_idx: count}
    Tracks how many times each graph has appeared in test sets.
    """
    return {
        0: 10,
        1: 10,
        2: 10,
        3: 10,
        4: 10,
    }


@pytest.fixture
def sample_pickle_data(sample_graph_dict, sample_test_appearances):
    """Create sample pickle file data structure."""
    return {
        "graph_dict": sample_graph_dict,
        "test_appearances": sample_test_appearances,
        "required_test_appearances": 10,
    }


class TestComputeAveragePerGraph:
    """Test the compute_average_per_graph function."""

    def test_compute_average_basic(self, sample_graph_dict):
        """Test basic average computation for classification."""
        graph_indices, average_values = compute_average_per_graph(sample_graph_dict)

        # Should have 5 graphs
        assert len(graph_indices) == 5
        assert len(average_values) == 5

        # Check that indices are sorted
        assert np.array_equal(graph_indices, np.array([0, 1, 2, 3, 4]))

        # Check average values (expected accuracies)
        expected_averages = np.array([0.8, 0.7, 1.0, 0.4, 0.9])
        np.testing.assert_array_almost_equal(
            average_values, expected_averages, decimal=2
        )

    def test_compute_average_perfect_graph(self):
        """Test that a graph with all correct predictions has accuracy 1.0."""
        graph_dict = {0: [1, 1, 1, 1, 1]}
        graph_indices, average_values = compute_average_per_graph(graph_dict)

        assert len(graph_indices) == 1
        assert len(average_values) == 1
        assert average_values[0] == 1.0

    def test_compute_average_always_wrong_graph(self):
        """Test that a graph with all incorrect predictions has accuracy 0.0."""
        graph_dict = {0: [0, 0, 0, 0, 0]}
        graph_indices, average_values = compute_average_per_graph(graph_dict)

        assert len(graph_indices) == 1
        assert average_values[0] == 0.0

    def test_compute_average_empty_list(self):
        """Test that graphs with empty lists are skipped."""
        graph_dict = {
            0: [1, 1, 1],
            1: [],  # Empty - should be skipped
            2: [0, 1, 0],
        }
        graph_indices, average_values = compute_average_per_graph(graph_dict)

        # Should only have 2 graphs (0 and 2)
        assert len(graph_indices) == 2
        assert 1 not in graph_indices

    def test_compute_average_varying_lengths(self):
        """Test that graphs can have different numbers of appearances."""
        graph_dict = {
            0: [1, 1],  # 2 appearances
            1: [1, 0, 1, 1, 0, 1, 1],  # 7 appearances
        }
        graph_indices, average_values = compute_average_per_graph(graph_dict)

        assert len(graph_indices) == 2
        # Graph 0: 2/2 = 1.0
        assert average_values[0] == 1.0
        # Graph 1: 5/7 ≈ 0.714
        assert abs(average_values[1] - 0.714) < 0.01


class TestPickleFileOperations:
    """Test saving and loading of graph_dict pickle files."""

    def test_save_pickle_file(self, temp_dir, sample_pickle_data):
        """Test saving graph_dict data to pickle file."""
        pickle_path = os.path.join(temp_dir, "test_graph_dict.pickle")

        # Save the data
        with open(pickle_path, "wb") as f:
            pickle.dump(sample_pickle_data, f)

        # Verify file exists
        assert os.path.exists(pickle_path)

        # Load and verify contents
        with open(pickle_path, "rb") as f:
            loaded_data = pickle.load(f)

        assert "graph_dict" in loaded_data
        assert "test_appearances" in loaded_data
        assert "required_test_appearances" in loaded_data
        assert loaded_data["required_test_appearances"] == 10
        assert len(loaded_data["graph_dict"]) == 5
        assert len(loaded_data["test_appearances"]) == 5

    def test_load_pickle_file(self, temp_dir, sample_pickle_data):
        """Test loading graph_dict data from pickle file."""
        pickle_path = os.path.join(temp_dir, "test_graph_dict.pickle")

        # Save first
        with open(pickle_path, "wb") as f:
            pickle.dump(sample_pickle_data, f)

        # Load and verify
        with open(pickle_path, "rb") as f:
            loaded_data = pickle.load(f)

        # Check graph_dict structure
        assert loaded_data["graph_dict"][0] == [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        assert loaded_data["graph_dict"][2] == [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]  # Perfect graph

        # Check test_appearances
        assert loaded_data["test_appearances"][0] == 10
        assert all(count == 10 for count in loaded_data["test_appearances"].values())

    def test_pickle_file_format_matches_production(self, temp_dir, sample_pickle_data):
        """Test that saved pickle file has the same format as production files."""
        pickle_path = os.path.join(temp_dir, "test_graph_dict.pickle")

        # Save in production format
        with open(pickle_path, "wb") as f:
            pickle.dump(sample_pickle_data, f)

        # Load and check structure matches what load_and_plot_average_per_graph expects
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        # Should have graph_dict key
        assert isinstance(data, dict)
        assert "graph_dict" in data
        assert isinstance(data["graph_dict"], dict)


class TestTestAppearancesTracking:
    """Test the logic for tracking test set appearances."""

    def test_test_appearances_initialization(self):
        """Test that test_appearances starts at 0 for all graphs."""
        dataset_size = 5
        test_appearances = {i: 0 for i in range(dataset_size)}

        assert len(test_appearances) == dataset_size
        assert all(count == 0 for count in test_appearances.values())

    def test_test_appearances_increment(self):
        """Test incrementing test appearances for graphs in test set."""
        test_appearances = {i: 0 for i in range(5)}

        # Simulate one trial where graphs [1, 3, 4] are in test set
        test_indices = [1, 3, 4]
        for graph_idx in test_indices:
            test_appearances[graph_idx] += 1

        assert test_appearances[0] == 0  # Not in test set
        assert test_appearances[1] == 1  # In test set
        assert test_appearances[2] == 0  # Not in test set
        assert test_appearances[3] == 1  # In test set
        assert test_appearances[4] == 1  # In test set

    def test_test_appearances_multiple_trials(self):
        """Test tracking appearances across multiple trials."""
        test_appearances = {i: 0 for i in range(5)}

        # Simulate multiple trials with different test sets
        trial_test_sets = [
            [0, 1, 2],  # Trial 1
            [1, 2, 3],  # Trial 2
            [0, 2, 4],  # Trial 3
        ]

        for test_indices in trial_test_sets:
            for graph_idx in test_indices:
                test_appearances[graph_idx] += 1

        # Check counts
        assert test_appearances[0] == 2  # In trials 1 and 3
        assert test_appearances[1] == 2  # In trials 1 and 2
        assert test_appearances[2] == 3  # In all 3 trials
        assert test_appearances[3] == 1  # Only in trial 2
        assert test_appearances[4] == 1  # Only in trial 3

    def test_check_required_appearances(self, sample_test_appearances):
        """Test checking if all graphs meet required appearances threshold."""
        required_test_appearances = 10

        # All graphs have exactly 10 appearances
        min_appearances = min(sample_test_appearances.values())
        assert min_appearances >= required_test_appearances

        # Count graphs with sufficient appearances
        graphs_with_sufficient = sum(
            1
            for count in sample_test_appearances.values()
            if count >= required_test_appearances
        )
        assert graphs_with_sufficient == len(sample_test_appearances)

    def test_check_insufficient_appearances(self):
        """Test when graphs don't meet required appearances threshold."""
        test_appearances = {
            0: 8,  # Below threshold
            1: 10,  # At threshold
            2: 12,  # Above threshold
            3: 7,  # Below threshold
        }
        required_test_appearances = 10

        min_appearances = min(test_appearances.values())
        assert min_appearances < required_test_appearances

        graphs_with_sufficient = sum(
            1
            for count in test_appearances.values()
            if count >= required_test_appearances
        )
        assert graphs_with_sufficient == 2  # Only graphs 1 and 2


class TestGraphDictTracking:
    """Test the logic for tracking correct/incorrect classifications per graph."""

    def test_graph_dict_initialization(self):
        """Test that graph_dict starts with empty lists for all graphs."""
        dataset_size = 5
        graph_dict = {i: [] for i in range(dataset_size)}

        assert len(graph_dict) == dataset_size
        assert all(len(values) == 0 for values in graph_dict.values())

    def test_record_correct_classification(self):
        """Test recording a correct classification (value = 1)."""
        graph_dict = {0: [], 1: []}

        # Simulate correct classification for graph 0
        graph_idx = 0
        correctness = 1  # Correct
        graph_dict[graph_idx].append(correctness)

        assert graph_dict[0] == [1]
        assert graph_dict[1] == []

    def test_record_incorrect_classification(self):
        """Test recording an incorrect classification (value = 0)."""
        graph_dict = {0: []}

        # Simulate incorrect classification
        graph_dict[0].append(0)

        assert graph_dict[0] == [0]

    def test_record_multiple_trials(self):
        """Test recording correctness across multiple trials."""
        graph_dict = {0: [], 1: []}

        # Simulate multiple trials for graph 0
        trial_results = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
        for correctness in trial_results:
            graph_dict[0].append(correctness)

        assert graph_dict[0] == trial_results
        assert len(graph_dict[0]) == 10
        assert sum(graph_dict[0]) == 7  # 7 correct out of 10

    def test_compute_accuracy_from_graph_dict(self):
        """Test computing accuracy from graph_dict values."""
        graph_dict = {
            0: [1, 1, 0, 1, 1],  # 4/5 = 80%
            1: [0, 0, 1, 0, 0],  # 1/5 = 20%
            2: [1, 1, 1, 1, 1],  # 5/5 = 100%
        }

        # Compute average accuracy per graph
        graph_indices, average_values = compute_average_per_graph(graph_dict)

        # Check accuracies
        idx_0 = np.where(graph_indices == 0)[0][0]
        idx_1 = np.where(graph_indices == 1)[0][0]
        idx_2 = np.where(graph_indices == 2)[0][0]

        assert abs(average_values[idx_0] - 0.8) < 0.01
        assert abs(average_values[idx_1] - 0.2) < 0.01
        assert abs(average_values[idx_2] - 1.0) < 0.01

    def test_graph_dict_only_tracks_test_set_graphs(self):
        """Test that graph_dict only accumulates values for test set graphs."""
        graph_dict = {i: [] for i in range(5)}
        test_appearances = {i: 0 for i in range(5)}

        # Simulate one trial
        # Dictionary returned from Experiment.run() (correctness per graph)
        trial_dictionary = {
            0: 1,  # Graph 0: correct (but not in test set)
            1: 1,  # Graph 1: correct (in test set)
            2: -1,  # Graph 2: not evaluated (not in test set)
            3: 0,  # Graph 3: incorrect (in test set)
            4: 1,  # Graph 4: correct (in test set)
        }
        test_indices = [1, 3, 4]  # Only these graphs are in test set

        # Track appearances and correctness
        for graph_idx in test_indices:
            test_appearances[graph_idx] += 1
            if graph_idx in trial_dictionary and trial_dictionary[graph_idx] != -1:
                graph_dict[graph_idx].append(trial_dictionary[graph_idx])

        # Check that only test set graphs have values
        assert len(graph_dict[0]) == 0  # Not in test set
        assert graph_dict[1] == [1]  # In test set, correct
        assert len(graph_dict[2]) == 0  # Not in test set
        assert graph_dict[3] == [0]  # In test set, incorrect
        assert graph_dict[4] == [1]  # In test set, correct

        # Check appearances
        assert test_appearances[1] == 1
        assert test_appearances[3] == 1
        assert test_appearances[4] == 1


class TestFullPipeline:
    """Test the complete accuracy monitoring pipeline."""

    def test_complete_tracking_pipeline(self):
        """Test the complete flow: initialize, track, compute averages."""
        # Initialize
        dataset_size = 3
        graph_dict = {i: [] for i in range(dataset_size)}
        test_appearances = {i: 0 for i in range(dataset_size)}
        required_test_appearances = 10

        # Simulate 15 trials to ensure all graphs appear at least 10 times
        # Use a pattern that guarantees each graph appears equally
        for trial in range(15):
            # Rotate through all graphs: trial 0->[0,1], trial 1->[1,2], trial 2->[2,0], repeat
            graph0 = trial % dataset_size
            graph1 = (trial + 1) % dataset_size
            test_indices = [graph0, graph1]

            # Simulate correctness: alternating pattern
            trial_dictionary = {
                idx: 1 if (trial + idx) % 2 == 0 else 0 for idx in test_indices
            }

            # Track
            for graph_idx in test_indices:
                test_appearances[graph_idx] += 1
                if graph_idx in trial_dictionary and trial_dictionary[graph_idx] != -1:
                    graph_dict[graph_idx].append(trial_dictionary[graph_idx])

        # With 15 trials and 2 graphs per trial, each graph should appear 10 times
        # (each graph appears in 2 out of 3 positions, so 15 * 2/3 = 10)
        min_appearances = min(test_appearances.values())
        assert min_appearances >= required_test_appearances, (
            f"Min appearances {min_appearances} < required {required_test_appearances}. "
            f"Actual counts: {test_appearances}"
        )

        # Compute averages
        graph_indices, average_values = compute_average_per_graph(graph_dict)
        assert len(graph_indices) == dataset_size
        assert all(0.0 <= avg <= 1.0 for avg in average_values)

    def test_save_and_load_pipeline(self, temp_dir):
        """Test complete pipeline: track, save, load, compute."""
        # Simulate tracking
        graph_dict = {
            0: [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            1: [1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        }
        test_appearances = {0: 10, 1: 10}
        required_test_appearances = 10

        # Save
        pickle_path = os.path.join(temp_dir, "pipeline_test.pickle")
        with open(pickle_path, "wb") as f:
            pickle.dump(
                {
                    "graph_dict": graph_dict,
                    "test_appearances": test_appearances,
                    "required_test_appearances": required_test_appearances,
                },
                f,
            )

        # Load
        with open(pickle_path, "rb") as f:
            loaded_data = pickle.load(f)

        # Compute averages
        loaded_graph_dict = loaded_data["graph_dict"]
        graph_indices, average_values = compute_average_per_graph(loaded_graph_dict)

        # Verify
        assert len(graph_indices) == 2
        assert abs(average_values[0] - 0.8) < 0.01  # 8/10
        assert abs(average_values[1] - 0.7) < 0.01  # 7/10


class TestPlotGeneration:
    """Test plot generation functionality."""

    def test_load_and_plot_creates_files(self, temp_dir, sample_pickle_data):
        """Test that load_and_plot_average_per_graph creates plot files."""
        pickle_path = os.path.join(temp_dir, "test_plot.pickle")

        # Save pickle file
        with open(pickle_path, "wb") as f:
            pickle.dump(sample_pickle_data, f)

        # Generate plots
        original_path, sorted_path = load_and_plot_average_per_graph(
            pickle_path,
            dataset_name="test_dataset",
            layer_type="GCN",
            encoding=None,
            num_layers=4,
            task_type="classification",
            output_dir=temp_dir,
        )

        # Verify files were created
        assert os.path.exists(original_path)
        assert os.path.exists(sorted_path)
        assert original_path.endswith(".png")
        assert sorted_path.endswith(".png")

    def test_plot_generation_with_encoding(self, temp_dir, sample_pickle_data):
        """Test plot generation with encoding specified."""
        pickle_path = os.path.join(temp_dir, "test_plot_encoding.pickle")

        with open(pickle_path, "wb") as f:
            pickle.dump(sample_pickle_data, f)

        original_path, sorted_path = load_and_plot_average_per_graph(
            pickle_path,
            dataset_name="test_dataset",
            layer_type="GIN",
            encoding="LCP",
            num_layers=5,
            task_type="classification",
            output_dir=temp_dir,
        )

        assert os.path.exists(original_path)
        assert os.path.exists(sorted_path)
        # Verify plot files were created successfully
        assert original_path.endswith("by_index.png")
        assert sorted_path.endswith("by_accuracy.png")
