from __future__ import annotations

from .instance import TSPInstance


def nearest_neighbor(instance: TSPInstance, start: int = 0) -> tuple[int, ...]:
    if not 0 <= start < instance.n_nodes:
        raise ValueError("invalid start node")
    distance = instance.distance_matrix
    unvisited = set(range(instance.n_nodes))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda node: (distance[current, node], node))
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tuple(tour)


def two_opt(instance: TSPInstance, tour: tuple[int, ...]) -> tuple[int, ...]:
    """Deterministic best-improvement 2-opt local search."""
    best = list(tour)
    best_cost = instance.tour_cost(best)
    improved = True
    while improved:
        improved = False
        candidate_best = best
        candidate_cost = best_cost
        n = len(best)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                cost = instance.tour_cost(candidate)
                if cost < candidate_cost - 1e-12:
                    candidate_best = candidate
                    candidate_cost = cost
        if candidate_cost < best_cost - 1e-12:
            best = candidate_best
            best_cost = candidate_cost
            improved = True
    return tuple(best)
