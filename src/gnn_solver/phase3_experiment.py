from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .beam_decoder import multistart_beam_decoder
from .dataset import build_exact_dataset
from .exact import held_karp
from .heuristics import nearest_neighbor, two_opt
from .instance import random_euclidean_instance
from .statistics import exact_sign_test, paired_bootstrap_ci
from .train import TrainConfig, fit_edge_gnn, predict_edge_scores


@dataclass(frozen=True)
class Phase3Result:
    split: str
    method: str
    instance_seed: int
    model_seed: int | None
    n_nodes: int
    gap_pct: float
    inference_ms: float
    decode_ms: float
    refine_ms: float
    total_ms: float
    feasible: bool


def _gap(instance, tour, optimal_cost: float) -> tuple[float, bool]:
    try:
        cost = instance.tour_cost(tour)
    except ValueError:
        return float("inf"), False
    return 100.0 * (cost - optimal_cost) / optimal_cost, True


def train_models(
    train_seeds: list[int],
    validation_seeds: list[int],
    *,
    n_nodes: int,
    model_seeds: tuple[int, ...] = (0, 1, 2),
    epochs: int = 12,
):
    train_examples = build_exact_dataset(train_seeds, n_nodes=n_nodes)
    validation_examples = build_exact_dataset(validation_seeds, n_nodes=n_nodes)
    models = []
    for seed in model_seeds:
        model, _ = fit_edge_gnn(
            train_examples,
            TrainConfig(hidden_dim=32, layers=2, epochs=epochs, seed=seed),
            validation_examples=validation_examples,
        )
        models.append((seed, model))
    return models


def evaluate_split(
    models,
    *,
    split: str,
    seeds: list[int],
    n_nodes: int,
    beam_width: int = 6,
) -> list[Phase3Result]:
    examples = {item.seed: item for item in build_exact_dataset(seeds, n_nodes=n_nodes)}
    rows: list[Phase3Result] = []
    for seed in seeds:
        instance = random_euclidean_instance(seed, n_nodes=n_nodes)
        optimal_cost = float(held_karp(instance)["cost"])

        start = perf_counter()
        nn_tour = nearest_neighbor(instance)
        nn_elapsed = (perf_counter() - start) * 1000.0
        nn_gap, nn_feasible = _gap(instance, nn_tour, optimal_cost)
        rows.append(
            Phase3Result(
                split,
                "nearest_neighbor",
                seed,
                None,
                n_nodes,
                nn_gap,
                0.0,
                nn_elapsed,
                0.0,
                nn_elapsed,
                nn_feasible,
            )
        )

        start = perf_counter()
        refined = two_opt(instance, nn_tour)
        refine_elapsed = (perf_counter() - start) * 1000.0
        refine_gap, refine_feasible = _gap(instance, refined, optimal_cost)
        rows.append(
            Phase3Result(
                split,
                "nearest_neighbor_2opt",
                seed,
                None,
                n_nodes,
                refine_gap,
                0.0,
                nn_elapsed,
                refine_elapsed,
                nn_elapsed + refine_elapsed,
                refine_feasible,
            )
        )

        for model_seed, model in models:
            infer_start = perf_counter()
            scores = predict_edge_scores(model, examples[seed])
            inference_ms = (perf_counter() - infer_start) * 1000.0

            decode_start = perf_counter()
            tour = multistart_beam_decoder(
                instance,
                scores,
                beam_width=beam_width,
                refine=False,
            )
            decode_ms = (perf_counter() - decode_start) * 1000.0

            refine_start = perf_counter()
            tour_refined = two_opt(instance, tour)
            refine_ms = (perf_counter() - refine_start) * 1000.0
            gap, feasible = _gap(instance, tour_refined, optimal_cost)
            rows.append(
                Phase3Result(
                    split,
                    "gnn_beam_2opt",
                    seed,
                    model_seed,
                    n_nodes,
                    gap,
                    inference_ms,
                    decode_ms,
                    refine_ms,
                    inference_ms + decode_ms + refine_ms,
                    feasible,
                )
            )
    return rows


def aggregate_model_seeds(rows: list[Phase3Result]) -> list[dict[str, float | int | str]]:
    output: list[dict[str, float | int | str]] = []
    keys = sorted({(row.split, row.method, row.instance_seed) for row in rows})
    for split, method, seed in keys:
        selected = [
            row
            for row in rows
            if row.split == split and row.method == method and row.instance_seed == seed
        ]
        output.append(
            {
                "split": split,
                "method": method,
                "instance_seed": seed,
                "n_nodes": selected[0].n_nodes,
                "gap_pct": float(np.mean([row.gap_pct for row in selected])),
                "inference_ms": float(np.mean([row.inference_ms for row in selected])),
                "decode_ms": float(np.mean([row.decode_ms for row in selected])),
                "refine_ms": float(np.mean([row.refine_ms for row in selected])),
                "total_ms": float(np.mean([row.total_ms for row in selected])),
                "feasible": float(np.mean([row.feasible for row in selected])),
            }
        )
    return output


def summarize(aggregated: list[dict[str, float | int | str]]):
    summaries = []
    keys = sorted({(row["split"], row["method"]) for row in aggregated})
    for split, method in keys:
        rows = [row for row in aggregated if row["split"] == split and row["method"] == method]
        summaries.append(
            {
                "split": split,
                "method": method,
                "mean_gap_pct": float(np.mean([float(row["gap_pct"]) for row in rows])),
                "mean_inference_ms": float(
                    np.mean([float(row["inference_ms"]) for row in rows])
                ),
                "mean_decode_ms": float(np.mean([float(row["decode_ms"]) for row in rows])),
                "mean_refine_ms": float(np.mean([float(row["refine_ms"]) for row in rows])),
                "mean_total_ms": float(np.mean([float(row["total_ms"]) for row in rows])),
                "feasibility_rate": float(np.mean([float(row["feasible"]) for row in rows])),
            }
        )
    return summaries


def paired_comparisons(aggregated: list[dict[str, float | int | str]]):
    comparisons = []
    for split in sorted({str(row["split"]) for row in aggregated}):
        baseline_rows = sorted(
            [
                row
                for row in aggregated
                if row["split"] == split and row["method"] == "nearest_neighbor_2opt"
            ],
            key=lambda row: int(row["instance_seed"]),
        )
        candidate_rows = sorted(
            [row for row in aggregated if row["split"] == split and row["method"] == "gnn_beam_2opt"],
            key=lambda row: int(row["instance_seed"]),
        )
        if not baseline_rows or len(baseline_rows) != len(candidate_rows):
            continue
        baseline = np.asarray([float(row["gap_pct"]) for row in baseline_rows])
        candidate = np.asarray([float(row["gap_pct"]) for row in candidate_rows])
        stats = paired_bootstrap_ci(baseline, candidate, samples=1000, seed=17)
        comparisons.append(
            {
                "split": split,
                **stats,
                "sign_test_p": exact_sign_test(baseline, candidate),
                "n": len(baseline),
            }
        )
    return comparisons


def main() -> None:
    models = train_models(
        list(range(20)),
        [50, 51, 52, 53, 54],
        n_nodes=8,
        model_seeds=(0, 1, 2),
        epochs=8,
    )
    rows = []
    rows.extend(evaluate_split(models, split="test", seeds=[100, 101, 102, 103], n_nodes=8))
    rows.extend(evaluate_split(models, split="ood_10", seeds=[200, 201, 202], n_nodes=10))
    rows.extend(evaluate_split(models, split="ood_12", seeds=[300, 301], n_nodes=12))
    aggregated = aggregate_model_seeds(rows)
    for row in summarize(aggregated):
        print(
            f"{row['split']},{row['method']},gap={row['mean_gap_pct']:.3f}%,"
            f"infer_ms={row['mean_inference_ms']:.3f},decode_ms={row['mean_decode_ms']:.3f},"
            f"refine_ms={row['mean_refine_ms']:.3f},total_ms={row['mean_total_ms']:.3f},"
            f"feasible={row['feasibility_rate']:.3f}"
        )
    for row in paired_comparisons(aggregated):
        print(
            f"paired,{row['split']},mean_diff={row['mean_difference']:.4f},"
            f"ci95=[{row['ci_low']:.4f},{row['ci_high']:.4f}],"
            f"win_rate={row['win_rate']:.3f},p={row['sign_test_p']:.4f},n={row['n']}"
        )


if __name__ == "__main__":
    main()
