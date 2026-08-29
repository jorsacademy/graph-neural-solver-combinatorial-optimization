from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without neural extra
    raise ImportError("Install neural dependencies with pip install -e '.[neural]'") from exc


class EdgeGNN(nn.Module):
    """Small permutation-equivariant GNN that scores undirected TSP edges."""

    def __init__(self, hidden_dim: int = 64, layers: int = 3):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_features):
        """Return symmetric edge logits for one complete graph.

        Shapes: node_features ``[n,2]`` and edge_features ``[n,n,3]``.
        """
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)
        n = h.shape[0]
        mask = 1.0 - torch.eye(n, dtype=h.dtype, device=h.device)

        for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
            hi = h[:, None, :].expand(n, n, -1)
            hj = h[None, :, :].expand(n, n, -1)
            messages = message_mlp(torch.cat([hi, hj, e], dim=-1))
            aggregate = (messages * mask[..., None]).sum(dim=1) / max(n - 1, 1)
            h = h + update_mlp(torch.cat([h, aggregate], dim=-1))

        hi = h[:, None, :].expand(n, n, -1)
        hj = h[None, :, :].expand(n, n, -1)
        pair = torch.cat([hi + hj, torch.abs(hi - hj), e], dim=-1)
        logits = self.edge_head(pair).squeeze(-1)
        logits = 0.5 * (logits + logits.T)
        return logits
