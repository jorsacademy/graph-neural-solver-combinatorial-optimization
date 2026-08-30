import numpy as np

from gnn_solver.phase3_experiment import aggregate_model_seeds
from gnn_solver.statistics import exact_sign_test, paired_bootstrap_ci


def test_paired_bootstrap_detects_improvement_direction():
    baseline = np.array([5.0, 4.0, 6.0, 3.0])
    candidate = np.array([3.0, 2.0, 4.0, 2.0])
    result = paired_bootstrap_ci(baseline, candidate, samples=200, seed=1)
    assert result["mean_difference"] < 0.0
    assert result["win_rate"] == 1.0


def test_exact_sign_test_is_one_for_all_ties():
    values = np.array([1.0, 2.0, 3.0])
    assert exact_sign_test(values, values) == 1.0


def test_model_seed_aggregation_is_instance_level():
    from gnn_solver.phase3_experiment import Phase3Result

    rows = [
        Phase3Result("test", "gnn_beam_2opt", 10, 0, 8, 2.0, 1.0, 2.0, 3.0, 6.0, True),
        Phase3Result("test", "gnn_beam_2opt", 10, 1, 8, 4.0, 3.0, 4.0, 5.0, 12.0, True),
    ]
    aggregated = aggregate_model_seeds(rows)
    assert len(aggregated) == 1
    assert aggregated[0]["gap_pct"] == 3.0
    assert aggregated[0]["total_ms"] == 9.0
