from __future__ import annotations

from itertools import combinations
from math import inf

from .instance import TSPInstance


def held_karp(instance: TSPInstance) -> dict[str, object]:
    """Solve symmetric TSP exactly with Held-Karp dynamic programming.

    Node 0 is fixed as the canonical start to remove rotational symmetry.
    Suitable for small research-label instances, not large-scale deployment.
    """
    n = instance.n_nodes
    distance = instance.distance_matrix
    dp: dict[tuple[int, int], tuple[float, int]] = {}

    for node in range(1, n):
        dp[(1 << node, node)] = (float(distance[0, node]), 0)

    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            mask = sum(1 << node for node in subset)
            for last in subset:
                prev_mask = mask ^ (1 << last)
                best_cost = inf
                best_prev = -1
                for prev in subset:
                    if prev == last:
                        continue
                    prev_cost = dp[(prev_mask, prev)][0]
                    candidate = prev_cost + float(distance[prev, last])
                    if candidate < best_cost:
                        best_cost = candidate
                        best_prev = prev
                dp[(mask, last)] = (best_cost, best_prev)

    full_mask = sum(1 << node for node in range(1, n))
    best_cost = inf
    best_last = -1
    for last in range(1, n):
        candidate = dp[(full_mask, last)][0] + float(distance[last, 0])
        if candidate < best_cost:
            best_cost = candidate
            best_last = last

    reversed_path = [best_last]
    mask = full_mask
    last = best_last
    while True:
        _, prev = dp[(mask, last)]
        mask ^= 1 << last
        if prev == 0:
            break
        reversed_path.append(prev)
        last = prev

    tour = (0, *reversed(reversed_path))
    return {"tour": tour, "cost": float(best_cost)}
