import numpy as np

from gnn_solver.dataset import build_exact_dataset
from gnn_solver.decoder import greedy_edge_decoder
from gnn_solver.instance import random_euclidean_instance
from gnn_solver.train import TrainConfig, fit_edge_gnn, predict_edge_scores


def test_neural_training_and_decoding_smoke():
    examples = build_exact_dataset([0, 1, 2, 3], n_nodes=6)
    model, history = fit_edge_gnn(
        examples,
        TrainConfig(hidden_dim=16, layers=1, epochs=2, seed=0),
    )
    assert len(history) == 2
    assert np.isfinite(history[-1])

    scores = predict_edge_scores(model, examples[0])
    instance = random_euclidean_instance(0, n_nodes=6)
    tour = greedy_edge_decoder(instance, scores)
    assert set(tour) == set(range(6))
