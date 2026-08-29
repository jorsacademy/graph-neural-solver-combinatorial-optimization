# Graph Neural Solver for Combinatorial Optimization

Research-oriented benchmark for **learning-augmented combinatorial optimization** with graph neural networks and exact/heuristic OR baselines.

## Research question

Can a graph neural model learn structural information from exact small-instance solutions and use it to guide a fast combinatorial decoder without sacrificing feasibility, and when does that guidance outperform classical heuristics after accounting for optimality gap and latency?

## Initial problem: Euclidean TSP

The first benchmark is the symmetric Euclidean Traveling Salesperson Problem (TSP). For cities `V={1,...,n}` and Euclidean edge costs `c_ij`, find a Hamiltonian cycle minimizing

```text
min  sum_(i,j) c_ij x_ij
```

subject to degree and subtour-elimination logic implied by the tour representation.

The repository is deliberately structured so the learning layer is separated from the combinatorial decoder and evaluation protocol. The same architecture can later be extended to CVRP and scheduling-derived graphs.

## Solver stack

The benchmark compares:

- exact Held-Karp dynamic programming on small instances;
- nearest-neighbor construction;
- nearest-neighbor + 2-opt local search;
- GNN edge scoring;
- feasibility-preserving neural-guided tour construction;
- neural-guided construction + 2-opt refinement.

The GNN does **not** directly emit an unconstrained permutation. It scores edges; a combinatorial decoder then constructs a valid Hamiltonian cycle using only feasible partial-tour extensions. This makes feasibility explicit and separates prediction quality from decoding quality.

## Research contract

A learned method is not considered better merely because it uses a GNN. Evaluation must report:

- tour cost;
- optimality gap where an exact oracle is tractable;
- feasibility rate;
- inference/solve latency;
- multiple random seeds;
- in-distribution and size-shift evaluation;
- comparison against nearest-neighbor and 2-opt;
- negative/null results.

## Planned repository layout

```text
src/gnn_solver/
  instance.py          # seeded Euclidean TSP instances
  exact.py             # Held-Karp exact oracle
  heuristics.py        # nearest neighbor and 2-opt
  features.py          # node/edge tensors
  dataset.py           # exact-solution supervision
  model.py             # lightweight message-passing GNN
  decoder.py           # feasibility-preserving neural decoder
  train.py             # reproducible training loop
  evaluate.py          # gap, latency and OOD benchmark
tests/
configs/
docs/
.github/workflows/
```

## Scope boundary

The initial benchmark focuses on Euclidean TSP because exact optimal labels are obtainable for controlled small instances and solution quality is easy to audit. Large-scale VRP, learned branching, neural cut selection and scheduling graphs are follow-up studies rather than claims of the initial version.

## License

MIT
