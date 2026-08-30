from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from .heuristics import two_opt
from .instance import TSPInstance


@dataclass(frozen=True)
class BeamState:
    tour: tuple[int, ...]
    unvisited: frozenset[int]
    score: float


def _edge_log_score(edge_scores: np.ndarray, i: int, j: int) -> float:
    probability = float(np.clip(edge_scores[i, j], 1e-8, 1.0 - 1e-8))
    return log(probability)


def beam_edge_decoder(
    instance: TSPInstance,
    edge_scores: np.ndarray,
    *,
    start: int = 0,
    beam_width: int = 8,
) -> tuple[int, ...]:
    """Decode a Hamiltonian tour with beam search over feasible partial tours."""
    scores = np.asarray(edge_scores, dtype=float)
    n = instance.n_nodes
    if scores.shape != (n, n):
        raise ValueError("edge_scores must have shape [n_nodes, n_nodes]")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if not 0 <= start < n:
        raise ValueError("invalid start node")

    beam = [BeamState((start,), frozenset(set(range(n)) - {start}), 0.0)]
    while beam[0].unvisited:
        candidates: list[BeamState] = []
        for state in beam:
            current = state.tour[-1]
            for nxt in state.unvisited:
                candidates.append(
                    BeamState(
                        state.tour + (nxt,),
                        state.unvisited - {nxt},
                        state.score + _edge_log_score(scores, current, nxt),
                    )
                )
        candidates.sort(key=lambda state: (-state.score, state.tour))
        beam = candidates[:beam_width]

    completed = [
        (
            state.score + _edge_log_score(scores, state.tour[-1], state.tour[0]),
            state.tour,
        )
        for state in beam
    ]
    completed.sort(key=lambda item: (-item[0], item[1]))
    return completed[0][1]


def multistart_beam_decoder(
    instance: TSPInstance,
    edge_scores: np.ndarray,
    *,
    beam_width: int = 8,
    starts: tuple[int, ...] | None = None,
    refine: bool = False,
) -> tuple[int, ...]:
    """Run beam decoding from multiple starts and keep the lowest-cost valid tour."""
    if starts is None:
        starts = tuple(range(instance.n_nodes))
    if not starts:
        raise ValueError("starts must not be empty")

    candidates = []
    for start in starts:
        tour = beam_edge_decoder(instance, edge_scores, start=start, beam_width=beam_width)
        if refine:
            tour = two_opt(instance, tour)
        candidates.append((instance.tour_cost(tour), tour))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]
