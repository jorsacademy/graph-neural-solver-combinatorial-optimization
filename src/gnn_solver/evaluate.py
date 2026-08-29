from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .dataset import TSPExample, build_exact_dataset
from .decoder import greedy_edge_decoder, neural_guided_two_opt
from .exact import held_karp
from .heuristics import nearest_neighbor, two_opt
from .instance import random_euclidean_instance
from .train import TrainConfig, fit_edge_gnn, predict_edge_scores


@dataclass(frozen=True)
class Result:
    method: str
    seed: int
    n_nodes: int
    cost: float
    optimal_cost: float
    gap_pct: float
    latency_ms: float
    feasible: bool


def _record(method, seed, instance, tour, optimal_cost, elapsed) -> Result:
    try:
        cost = instance.tour_cost(tour)
        feasible = True
    except ValueError:
        cost = float("inf")
        feasible = False
    gap = 100.0 * (cost - optimal_cost) / optimal_cost if feasible else float("inf")
    return Result(method, seed, instance.n_nodes, cost, optimal_cost, gap, elapsed * 1000.0, feasible)


def evaluate_model(model, seeds: list[int], n_nodes: int) -> list[Result]:
    results: list[Result] = []
    examples = build_exact_dataset(seeds, n_nodes=n_nodes)
    by_seed: dict[int, TSPExample] = {example.seed: example for example in examples}
    for seed in seeds:
        instance = random_euclidean_instance(seed, n_nodes=n_nodes)
        optimal = held_karp(instance)

        start = perf_counter()
        nn_tour = nearest_neighbor(instance)
        results.append(_record("nearest_neighbor", seed, instance, nn_tour, optimal["cost"], perf_counter() - start))

        start = perf_counter()
        opt_tour = two_opt(instance, nn_tour)
        results.append(_record("nearest_neighbor_2opt", seed, instance, opt_tour, optimal["cost"], perf_counter() - start))

        scores = predict_edge_scores(model, by_seed[seed])
        start = perf_counter()
        neural_tour = greedy_edge_decoder(instance, scores)
        results.append(_record("gnn_greedy", seed, instance, neural_tour, optimal["cost"], perf_counter() - start))

        start = perf_counter()
        refined = neural_guided_two_opt(instance, scores)
        results.append(_record("gnn_2opt", seed, instance, refined, optimal["cost"], perf_counter() - start))
    return results


def summarize(results: list[Result]) -> list[dict[str, float | str]]:
    summary = []
    for method in sorted({row.method for row in results}):
        rows = [row for row in results if row.method == method]
        summary.append(
            {
                "method": method,
                "mean_gap_pct": float(np.mean([row.gap_pct for row in rows])),
                "median_gap_pct": float(np.median([row.gap_pct for row in rows])),
                "mean_latency_ms": float(np.mean([row.latency_ms for row in rows])),
                "feasibility_rate": float(np.mean([row.feasible for row in rows])),
            }
        )
    return summary


def main() -> None:
    train_examples = build_exact_dataset(list(range(20)), n_nodes=8)
    model, history = fit_edge_gnn(
        train_examples,
        TrainConfig(hidden_dim=32, layers=2, epochs=20, seed=0),
    )
    print(f"final_training_loss={history[-1]:.6f}")
    results = evaluate_model(model, seeds=[100, 101, 102, 103], n_nodes=8)
    for row in summarize(results):
        print(
            f"{row['method']},gap={row['mean_gap_pct']:.3f}%,"
            f"latency_ms={row['mean_latency_ms']:.3f},"
            f"feasible={row['feasibility_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
