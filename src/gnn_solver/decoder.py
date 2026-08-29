from __future__ import annotations

import numpy as np

from .heuristics import two_opt
from .instance import TSPInstance


def greedy_edge_decoder(
    instance: TSPInstance,
    edge_scores: np.ndarray,
    start: int = 0,
) -> tuple[int, ...]:
    """Construct a Hamiltonian tour by choosing the best feasible outgoing edge."""
    scores = np.asarray(edge_scores, dtype=float)
    n = instance.n_nodes
    if scores.shape != (n, n):
        raise ValueError("edge_scores must have shape [n_nodes, n_nodes]")
    if not 0 <= start < n:
        raise ValueError("invalid start node")

    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    distance = instance.distance_matrix

    while unvisited:
        nxt = max(
            unvisited,
            key=lambda node: (scores[current, node], -distance[current, node], -node),
        )
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tuple(tour)


def neural_guided_two_opt(
    instance: TSPInstance,
    edge_scores: np.ndarray,
    start: int = 0,
) -> tuple[int, ...]:
    return two_opt(instance, greedy_edge_decoder(instance, edge_scores, start=start))
