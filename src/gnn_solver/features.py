from __future__ import annotations

import numpy as np

from .instance import TSPInstance


def graph_features(instance: TSPInstance) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized node coordinates and pairwise edge features."""
    coords = np.asarray(instance.coordinates, dtype=np.float32)
    centered = coords - coords.mean(axis=0, keepdims=True)
    scale = max(float(np.std(centered)), 1e-6)
    node_features = centered / scale

    distance = instance.distance_matrix.astype(np.float32)
    n = instance.n_nodes
    edge_features = np.zeros((n, n, 3), dtype=np.float32)
    edge_features[..., 0] = distance
    edge_features[..., 1] = 1.0 / (1.0 + distance)
    edge_features[..., 2] = 1.0 - np.eye(n, dtype=np.float32)
    return node_features, edge_features
