# Experimental Protocol

## Goal

Evaluate whether graph-neural edge scores improve combinatorial search quality relative to transparent OR baselines while preserving feasibility and controlling inference cost.

## Data split

Synthetic Euclidean TSP instances are generated from deterministic seeds. Training, validation, final test and size-shift seeds must remain disjoint. Final test instances must not be used for architecture or hyperparameter decisions.

## Exact supervision

Held-Karp dynamic programming provides globally optimal tours on small instances. Those tours are converted into symmetric binary edge labels. The exact solver is used as a labeling oracle and as a final-test reference wherever tractable.

## Baselines

Every learned result must be compared against:

1. nearest neighbor;
2. nearest neighbor followed by deterministic 2-opt;
3. the exact Held-Karp optimum where tractable.

The primary learned variants are GNN-guided greedy decoding and GNN-guided decoding followed by the same 2-opt refinement used by the classical baseline.

## Feasibility

The neural network never directly declares a tour feasible. It emits edge scores. A deterministic decoder maintains the set of unvisited cities and appends exactly one feasible city at each step, so every completed result is a Hamiltonian tour unless the decoder itself fails.

## Metrics

Primary metric: percentage optimality gap versus Held-Karp.

Secondary metrics:

- absolute tour cost;
- median optimality gap;
- feasibility rate;
- end-to-end inference/solve latency;
- model training loss;
- performance under node-count shift.

## Interpretation

A GNN is not promoted merely for beating nearest neighbor. The main reference is nearest-neighbor + 2-opt because it is fast, deterministic and substantially stronger. If GNN guidance followed by 2-opt does not improve quality or latency enough to justify model complexity, the negative result is retained.

## Scope

The first benchmark intentionally studies a clean Euclidean TSP. CVRP, learned branching, neural cut generation, scheduling graphs and industrial datasets are later research questions and should not be mixed into the initial experimental claim.
