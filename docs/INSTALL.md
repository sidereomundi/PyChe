# Installation

From the repository root:

```bash
pip install -e .
```

With optional features:

```bash
pip install -e .[plots,mpi]
```

Install Cython support explicitly (recommended if you want to force `backend="cython"`):

```bash
pip install cython
```

## Requirements

- Python 3.10+
- `numpy` (required)
- `pandas` (for dataframe file output)
- `matplotlib` (for diagnostic plot generation)
- `mpi4py` (for MPI runs)
- `cython` (optional, for Cython backend when compiled extensions are present)

The required yield/input data are bundled inside the package under `pyche/data/`.

## Backend selection guidance

- Use `backend="auto"` for robust runs across machines.
- Use `backend="cython"` only if compiled extension modules are available.
- Use `backend="numpy"` for guaranteed pure-Python/NumPy compatibility.
