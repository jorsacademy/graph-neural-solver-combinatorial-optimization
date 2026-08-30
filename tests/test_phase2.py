import numpy as np

from gnn_solver.beam_decoder import beam_edge_decoder, multistart_beam_decoder
from gnn_solver.instance import random_euclidean_instance
from gnn_solver.phase2_experiment import evaluate_split, train_select_model


def test_beam_decoder_returns_feasible_tour():
    instance = random_euclidean_instance(21, n_nodes=8)
    rng = np.random.default_rng(21)
    scores = rng.uniform(size=(8, 8))
    scores = 0.5 * (scores + scores.T)
    tour = beam_edge_decoder(instance, scores, start=0, beam_width=4)
    assert len(tour) == 8
    assert set(tour) == set(range(8))


def test_multistart_refinement_is_not_worse_than_unrefined():
    instance = random_euclidean_instance(22, n_nodes=8)
    rng = np.random.default_rng(22)
    scores = rng.uniform(size=(8, 8))
    scores = 0.5 * (scores + scores.T)
    raw = multistart_beam_decoder(instance, scores, beam_width=3, refine=False)
    refined = multistart_beam_decoder(instance, scores, beam_width=3, refine=True)
    assert instance.tour_cost(refined) <= instance.tour_cost(raw) + 1e-12


def test_validation_selected_model_can_evaluate_unseen_split():
    selection = train_select_model(
        [0, 1, 2],
        [50, 51],
        n_nodes=7,
        model_seeds=(0, 1),
        beam_width=2,
        epochs=2,
    )
    assert selection["selected_seed"] in {0, 1}
    assert set(selection["all_validation_gaps"]) == {0, 1}

    results = evaluate_split(
        selection["model"],
        split="test",
        seeds=[100],
        n_nodes=7,
        beam_width=2,
    )
    assert {result.method for result in results} == {
        "nearest_neighbor",
        "nearest_neighbor_2opt",
        "gnn_multistart_beam",
        "gnn_multistart_beam_2opt",
    }
    assert all(result.feasible for result in results)
