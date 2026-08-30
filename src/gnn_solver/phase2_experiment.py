from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .beam_decoder import multistart_beam_decoder
from .dataset import TSPExample, build_exact_dataset
from .exact import held_karp
from .heuristics import nearest_neighbor, two_opt
from .instance import random_euclidean_instance
from .train import TrainConfig, fit_edge_gnn, predict_edge_scores


@dataclass(frozen=True)
class Phase2Result:
    method: str
    split: str
    seed: int
    n_nodes: int
    gap_pct: float
    latency_ms: float
    feasible: bool


def _gap(instance, tour, optimal_cost: float) -> tuple[float, bool]:
    try:
        cost = instance.tour_cost(tour)
    except ValueError:
        return float("inf"), False
    return 100.0 * (cost - optimal_cost) / optimal_cost, True


def _mean_gap(model, examples: list[TSPExample], beam_width: int) -> float:
    gaps = []
    for example in examples:
        instance = random_euclidean_instance(example.seed, n_nodes=example.node_features.shape[0])
        optimal = held_karp(instance)
        scores = predict_edge_scores(model, example)
        tour = multistart_beam_decoder(instance, scores, beam_width=beam_width, refine=True)
        gap, feasible = _gap(instance, tour, float(optimal["cost"]))
        if not feasible:
            return float("inf")
        gaps.append(gap)
    return float(np.mean(gaps))


def train_select_model(
    train_seeds: list[int],
    validation_seeds: list[int],
    *,
    n_nodes: int,
    model_seeds: tuple[int, ...] = (0, 1, 2),
    beam_width: int = 8,
    epochs: int = 20,
):
    train_examples = build_exact_dataset(train_seeds, n_nodes=n_nodes)
    validation_examples = build_exact_dataset(validation_seeds, n_nodes=n_nodes)
    candidates = []
    for model_seed in model_seeds:
        model, history = fit_edge_gnn(
            train_examples,
            TrainConfig(hidden_dim=32, layers=2, epochs=epochs, seed=model_seed),
            validation_examples=validation_examples,
        )
        validation_gap = _mean_gap(model, validation_examples, beam_width)
        candidates.append((validation_gap, model_seed, model, history[-1]))
    candidates.sort(key=lambda row: (row[0], row[1]))
    return {
        "model": candidates[0][2],
        "selected_seed": candidates[0][1],
        "validation_gap_pct": candidates[0][0],
        "final_training_loss": candidates[0][3],
        "all_validation_gaps": {seed: gap for gap, seed, _, _ in candidates},
    }


def evaluate_split(
    model,
    *,
    split: str,
    seeds: list[int],
    n_nodes: int,
    beam_width: int = 8,
) -> list[Phase2Result]:
    examples = {example.seed: example for example in build_exact_dataset(seeds, n_nodes=n_nodes)}
    results: list[Phase2Result] = []
    for seed in seeds:
        instance = random_euclidean_instance(seed, n_nodes=n_nodes)
        optimal = held_karp(instance)
        optimal_cost = float(optimal["cost"])

        start = perf_counter()
        nn_tour = nearest_neighbor(instance)
        nn_gap, nn_feasible = _gap(instance, nn_tour, optimal_cost)
        results.append(
            Phase2Result(
                "nearest_neighbor",
                split,
                seed,
                n_nodes,
                nn_gap,
                (perf_counter() - start) * 1000.0,
                nn_feasible,
            )
        )

        start = perf_counter()
        opt_tour = two_opt(instance, nn_tour)
        opt_gap, opt_feasible = _gap(instance, opt_tour, optimal_cost)
        results.append(
            Phase2Result(
                "nearest_neighbor_2opt",
                split,
                seed,
                n_nodes,
                opt_gap,
                (perf_counter() - start) * 1000.0,
                opt_feasible,
            )
        )

        scores = predict_edge_scores(model, examples[seed])
        start = perf_counter()
        beam_tour = multistart_beam_decoder(instance, scores, beam_width=beam_width, refine=False)
        beam_gap, beam_feasible = _gap(instance, beam_tour, optimal_cost)
        results.append(
            Phase2Result(
                "gnn_multistart_beam",
                split,
                seed,
                n_nodes,
                beam_gap,
                (perf_counter() - start) * 1000.0,
                beam_feasible,
            )
        )

        start = perf_counter()
        refined = multistart_beam_decoder(instance, scores, beam_width=beam_width, refine=True)
        refined_gap, refined_feasible = _gap(instance, refined, optimal_cost)
        results.append(
            Phase2Result(
                "gnn_multistart_beam_2opt",
                split,
                seed,
                n_nodes,
                refined_gap,
                (perf_counter() - start) * 1000.0,
                refined_feasible,
            )
        )
    return results


def summarize(results: list[Phase2Result]) -> list[dict[str, float | str]]:
    rows = []
    keys = sorted({(result.split, result.method) for result in results})
    for split, method in keys:
        selected = [result for result in results if result.split == split and result.method == method]
        rows.append(
            {
                "split": split,
                "method": method,
                "mean_gap_pct": float(np.mean([row.gap_pct for row in selected])),
                "median_gap_pct": float(np.median([row.gap_pct for row in selected])),
                "mean_latency_ms": float(np.mean([row.latency_ms for row in selected])),
                "feasibility_rate": float(np.mean([row.feasible for row in selected])),
            }
        )
    return rows


def main() -> None:
    selection = train_select_model(
        list(range(20)),
        [50, 51, 52, 53, 54],
        n_nodes=8,
        model_seeds=(0, 1, 2),
        beam_width=6,
        epochs=12,
    )
    print(
        f"selected_model_seed={selection['selected_seed']},"
        f"validation_gap_pct={selection['validation_gap_pct']:.4f}"
    )
    results = []
    results.extend(
        evaluate_split(
            selection["model"],
            split="test",
            seeds=[100, 101, 102, 103],
            n_nodes=8,
            beam_width=6,
        )
    )
    results.extend(
        evaluate_split(
            selection["model"],
            split="ood_size",
            seeds=[200, 201, 202],
            n_nodes=10,
            beam_width=6,
        )
    )
    for row in summarize(results):
        print(
            f"{row['split']},{row['method']},gap={row['mean_gap_pct']:.3f}%,"
            f"latency_ms={row['mean_latency_ms']:.3f},"
            f"feasible={row['feasibility_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
