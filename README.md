# BCN Output Tracking via ILP

This repository contains experimental code for optimal output tracking in state-flipped Boolean control networks (BCNs). The proposed method is implemented with Google OR-Tools, an integer programming solver. Baselines include pruned enumeration, a semi-tensor product (STP) search, and STP with Q-learning.

## Repository layout

```text
.
├── data/                    # Example and benchmark network inputs (JSON)
├── scripts/                 # Generators, solvers, and experiment runners
├── pyproject.toml           # Project metadata and dependencies for uv
├── uv.lock                  # Reproducible dependency lock file
└── README.md
```

## Requirements and setup

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required. From the repository root, install the pinned dependencies with:

```bash
uv sync
```

This creates a local `.venv` and installs NumPy and OR-Tools.

## Quick start

Run commands from the repository root. Generate the six-node example and solve it with CP-SAT:

```bash
uv run python scripts/example_6node_case.py
uv run python scripts/cpsat_solver.py data/6_nodes_data.json
```

Generate a random case, then solve it:

```bash
uv run python scripts/bcn_generator.py --n 10 --m 3 --p 4 --T 10 --seed 0
uv run python scripts/cpsat_solver.py bcn_case_data_h_x.json
```

Other solver entry points:

```bash
uv run python scripts/pe_solver.py data/n_10_t_10.json
uv run python scripts/stp_solver.py data/n_10_t_10.json
uv run python scripts/q_learning_solver.py data/n_10_t_10.json --episodes 1000 --seed 0
```

Run the batch experiments with `uv run python scripts/batch_given_networks.py` or `uv run python scripts/batch_scale_sweep.py`. These scripts run multiple instances and can take substantial time and memory.

## Data format

Inputs use JSON fields `n`, `m`, `p`, `T`, `f_rules`, `h_rules`, `x0`, and `targets`. State and output rules are Boolean expressions supporting `&` (AND), `|` (OR), `^` (XOR), `~` (NOT), and parentheses. Optional ground-truth flip set and schedule fields support validation. See files under `data/` for complete examples.

## Resource notes

The STP solvers materialize transition tables with a size proportional to $2^{n+m}$; memory use grows rapidly with network size. Q-learning also performs reachability analysis and can require substantial time. Start with the six-node or ten-node examples. Large batch sweeps are intended for machines with adequate memory and may not be practical on a laptop.

## Experiments

- `batch_given_networks.py` runs CP-SAT on the supplied benchmark networks.
- `batch_scale_sweep.py` generates networks across a range of sizes and records timing and memory statistics.
- `example_6node_case.py` writes `data/6_nodes_data.json` for the paper's six-node example.

Solver output reports construction and solve times; individual methods may report additional timing details. Results are written by the batch runners to text files in the working directory.
