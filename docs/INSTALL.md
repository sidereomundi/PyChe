# Installation

From the repository root:

```bash
pip install -e .
```

With optional features:

```bash
pip install -e .[plots,mpi]
```

## Requirements

- Python 3.10+
- `numpy` (required)
- `pandas` (for dataframe file output)
- `matplotlib` (for diagnostic plot generation)
- `mpi4py` (for MPI runs)

The required yield/input data are bundled inside the package under `pyche/data/`.
