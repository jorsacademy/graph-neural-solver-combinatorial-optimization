from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .dataset import TSPExample
from .model import EdgeGNN


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 64
    layers: int = 3
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 0


def fit_edge_gnn(
    examples: list[TSPExample],
    config: TrainConfig | None = None,
) -> tuple[EdgeGNN, list[float]]:
    if not examples:
        raise ValueError("examples must not be empty")
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = EdgeGNN(hidden_dim=config.hidden_dim, layers=config.layers)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[float] = []

    for _ in range(config.epochs):
        epoch_loss = 0.0
        for example in examples:
            node = torch.as_tensor(example.node_features, dtype=torch.float32)
            edge = torch.as_tensor(example.edge_features, dtype=torch.float32)
            target = torch.as_tensor(example.edge_targets, dtype=torch.float32)
            logits = model(node, edge)
            n = logits.shape[0]
            upper = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
            loss = loss_fn(logits[upper], target[upper])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        history.append(epoch_loss / len(examples))
    return model, history


def predict_edge_scores(model: EdgeGNN, example: TSPExample) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.as_tensor(example.node_features, dtype=torch.float32),
            torch.as_tensor(example.edge_features, dtype=torch.float32),
        )
        return torch.sigmoid(logits).cpu().numpy()
