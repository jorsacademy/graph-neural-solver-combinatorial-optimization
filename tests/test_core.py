import numpy as np

from gnn_solver.dataset import build_exact_dataset
from gnn_solver.decoder import greedy_edge_decoder
from gnn_solver.exact import held_karp
from gnn_solver.heuristics import nearest_neighbor, two_opt
from gnn_solver.instance import random_euclidean_instance


def test_instance_generation_is_reproducible():
    first = random_euclidean_instance(7, n_nodes=7)
    second = random_euclidean_instance(7, n_nodes=7)
    assert np.allclose(first.coordinates, second.coordinates)


def test_held_karp_returns_feasible_optimal_tour():
    instance = random_euclidean_instance(3, n_nodes=7)
    result = held_karp(instance)
    assert set(result["tour"]) == set(range(7))
    assert abs(instance.tour_cost(result["tour"]) - result["cost"]) < 1e-10


def test_two_opt_never_worsens_nearest_neighbor():
    instance = random_euclidean_instance(11, n_nodes=9)
    initial = nearest_neighbor(instance)
    improved = two_opt(instance, initial)
    assert instance.tour_cost(improved) <= instance.tour_cost(initial) + 1e-12


def test_dataset_contains_symmetric_optimal_edge_targets():
    example = build_exact_dataset([5], n_nodes=7)[0]
    assert example.edge_targets.shape == (7, 7)
    assert np.allclose(example.edge_targets, example.edge_targets.T)
    assert np.all(example.edge_targets.sum(axis=1) == 2.0)


def test_decoder_always_returns_a_permutation():
    instance = random_euclidean_instance(13, n_nodes=8)
    rng = np.random.default_rng(13)
    scores = rng.normal(size=(8, 8))
    scores = 0.5 * (scores + scores.T)
    tour = greedy_edge_decoder(instance, scores)
    assert len(tour) == 8
    assert set(tour) == set(range(8))
