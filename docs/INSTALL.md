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
python setup.py build_ext --inplace
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

## Cython + MPI quick commands

Single process with Cython backend:

```bash
python setup.py build_ext --inplace
python -c "from pyche import GCEModel; m=GCEModel(); m.GCE(1000,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=False,backend='cython',show_progress=False)"
```

MPI run with Cython backend:

```bash
python setup.py build_ext --inplace
mpiexec -n 8 python -c "from pyche import GCEModel; m=GCEModel(); m.GCE(13700,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=True,backend='cython',show_progress=False)"
```
