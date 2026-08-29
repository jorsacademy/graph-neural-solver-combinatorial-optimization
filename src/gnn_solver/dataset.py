from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact import held_karp
from .features import graph_features
from .instance import random_euclidean_instance


@dataclass(frozen=True)
class TSPExample:
    seed: int
    node_features: np.ndarray
    edge_features: np.ndarray
    edge_targets: np.ndarray
    optimal_cost: float


def tour_adjacency(tour: tuple[int, ...]) -> np.ndarray:
    n = len(tour)
    target = np.zeros((n, n), dtype=np.float32)
    for index, node in enumerate(tour):
        nxt = tour[(index + 1) % n]
        target[node, nxt] = 1.0
        target[nxt, node] = 1.0
    return target


def build_exact_dataset(seeds: list[int], n_nodes: int = 9) -> list[TSPExample]:
    examples: list[TSPExample] = []
    for seed in seeds:
        instance = random_euclidean_instance(seed=seed, n_nodes=n_nodes)
        solution = held_karp(instance)
        node_features, edge_features = graph_features(instance)
        examples.append(
            TSPExample(
                seed=seed,
                node_features=node_features,
                edge_features=edge_features,
                edge_targets=tour_adjacency(solution["tour"]),
                optimal_cost=float(solution["cost"]),
            )
        )
    return examples
