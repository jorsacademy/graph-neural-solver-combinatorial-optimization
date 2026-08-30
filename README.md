# Graph Neural Solver for Combinatorial Optimization

Research-oriented benchmark for **learning-augmented combinatorial optimization** with graph neural networks and exact/heuristic OR baselines.

## Research question

Can a graph neural model learn structural information from exact small-instance solutions and use it to guide a fast combinatorial decoder without sacrificing feasibility, and when does that guidance outperform classical heuristics after accounting for optimality gap and latency?

## Current status

**Phase 2 implemented: multi-start beam decoding + validation-selected multi-seed GNN training.**

The repository currently includes:

- deterministic Euclidean TSP instance generation;
- exact Held-Karp dynamic programming for small instances;
- nearest-neighbor and deterministic 2-opt baselines;
- exact-tour edge supervision;
- normalized node and edge features;
- a lightweight permutation-equivariant PyTorch message-passing GNN;
- feasibility-preserving greedy decoding;
- beam search over feasible partial tours;
- multi-start beam decoding across all start nodes;
- optional 2-opt refinement after neural decoding;
- validation-based checkpoint selection using edge BCE;
- three independent model seeds with validation tour-gap model selection;
- frozen train/validation/test/OOD-size seed blocks;
- optimality-gap, latency and feasibility reporting;
- unit tests and GitHub Actions CI.

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
- GNN edge scoring + greedy decoding;
- GNN edge scoring + multi-start beam decoding;
- GNN multi-start beam decoding + 2-opt refinement.

The GNN does **not** directly emit an unconstrained permutation. It scores edges; combinatorial decoders then construct only valid Hamiltonian tours. This keeps feasibility explicit and separates prediction quality from decoding quality.

## Validation and model selection

Training, validation, test and OOD blocks are disjoint.

- training seeds: `0-19`;
- validation seeds: `50-54`;
- in-distribution test seeds: `100-103`;
- OOD-size seeds: `200-203` with larger node count.

During training, the checkpoint with the lowest validation edge BCE is retained. Independent model seeds `0`, `1` and `2` are then compared by **validation mean tour optimality gap** under the beam decoder. Test and OOD instances are never used for checkpoint, seed or decoder selection.

## Beam and multi-start decoding

Beam search maintains the highest-scoring feasible partial tours under learned edge probabilities. No repeated node is allowed. The decoder closes the Hamiltonian cycle only after all nodes have been visited.

Multi-start decoding runs the same beam search from several start nodes and keeps the lowest-cost feasible completed tour. Because the objective coefficients are known at decision time, using tour cost to choose among completed feasible candidate tours does not use optimal labels or test-set supervision.

## Exact supervision

For each training seed, Held-Karp returns a globally optimal tour. The tour is converted into a symmetric binary edge-adjacency target. The GNN is trained as an edge scorer rather than as a black-box permutation generator.

## Reproduce

Base OR stack:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q tests/test_core.py
```

Neural Phase 1 benchmark:

```bash
pip install -e '.[dev,neural]'
pytest -q tests/test_core.py tests/test_neural.py
python -m gnn_solver.evaluate
```

Phase 2 benchmark:

```bash
pytest -q tests/test_phase2.py
python -m gnn_solver.phase2_experiment
```

## Research contract

A learned method is not considered better merely because it uses a GNN. Evaluation must report:

- tour cost;
- optimality gap where an exact oracle is tractable;
- feasibility rate;
- inference/solve latency;
- multiple independent model seeds;
- in-distribution and size-shift evaluation;
- comparison against nearest-neighbor and 2-opt;
- negative/null results.

The key reference is **nearest-neighbor + 2-opt**, not nearest neighbor alone. If beam search or GNN guidance does not improve quality enough to justify its added latency and complexity, that result is retained.

## Repository layout

```text
src/gnn_solver/
  instance.py              # seeded Euclidean TSP instances
  exact.py                 # Held-Karp exact oracle
  heuristics.py            # nearest neighbor and 2-opt
  features.py              # node/edge tensors
  dataset.py               # exact-solution supervision
  model.py                 # lightweight message-passing GNN
  decoder.py               # greedy feasible neural decoder
  beam_decoder.py          # beam + multi-start decoding
  train.py                 # reproducible training + validation checkpoints
  evaluate.py              # Phase 1 gap/latency runner
  phase2_experiment.py     # multi-seed selection + test/OOD benchmark
tests/
  test_core.py
  test_neural.py
  test_phase2.py
configs/
  experiment.json
docs/
  experimental_protocol.md
.github/workflows/
  ci.yml
```

## Next research stages

### Phase 3 — statistical and deeper OOD evaluation
Add larger size shifts, paired bootstrap confidence intervals, model-seed aggregation, and end-to-end latency decomposition.

### Phase 4 — CVRP extension
Reuse the graph-learning stack on capacitated vehicle routing with explicit vehicle-capacity state and feasibility-preserving decoding.

## Scope boundary

The TSP benchmark is intended to establish a controlled neural-combinatorial solver methodology. Large-scale VRP, learned branching, neural cut selection and scheduling graphs are separate extensions rather than claims of the current implementation.

## License

MIT
