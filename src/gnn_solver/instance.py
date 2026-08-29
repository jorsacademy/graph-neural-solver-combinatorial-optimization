from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TSPInstance:
    coordinates: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def distance_matrix(self) -> np.ndarray:
        delta = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=2))

    def tour_cost(self, tour: tuple[int, ...] | list[int]) -> float:
        if len(tour) != self.n_nodes:
            raise ValueError("tour must contain every node exactly once")
        if set(tour) != set(range(self.n_nodes)):
            raise ValueError("tour must be a permutation of node ids")
        distance = self.distance_matrix
        return float(
            sum(distance[tour[i], tour[(i + 1) % self.n_nodes]] for i in range(self.n_nodes))
        )


def random_euclidean_instance(seed: int, n_nodes: int = 10) -> TSPInstance:
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least 3")
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 1.0, size=(n_nodes, 2))
    return TSPInstance(coordinates=coordinates)
