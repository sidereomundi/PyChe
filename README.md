# PyChe

`PyChe` is a pip-installable Python package for galactic chemical evolution runs, including required yield tables bundled in the package.

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[all]
```

Useful subsets:

```bash
pip install -e .[mpi]
pip install -e .[plots]
```

Optional Cython build (recommended for speed):

```bash
pip install cython
python setup.py build_ext --inplace
```

## Quick Start

```python
from pyche import GCEModel

m = GCEModel()
result = m.MinGCE(
    endoftime=200,
    sigmat=2000.0,
    sigmah=54.0,
    psfr=0.2,
    pwind=0.0,
    delay=7000,
    time_wind=1000000,
    use_mpi=False,
    show_progress=False,
    backend="auto",
    output_mode="dataframe",
    write_output=False,
    return_results=True,
)

mod = result.mod
fis = result.fis
```

## Core Parameters

`MinGCE` positional arguments map as:

```python
m.MinGCE(
    endoftime,   # total simulated timesteps
    sigmat,      # infall timescale width
    sigmah,      # surface-density normalization
    psfr,        # star-formation efficiency factor
    pwind,       # wind/outflow efficiency factor
    delay,       # infall profile delay/offset
    time_wind,   # timestep after which wind can activate
    ...
)
```

Example:

```python
m.MinGCE(
    endoftime=500,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
    use_mpi=False,
)
```

## Outputs

You can choose file output, in-memory return, or both:

- `write_output=True`, `return_results=False`: write files only.
- `write_output=False`, `return_results=True`: return arrays only.
- `write_output=True`, `return_results=True`: both.

## Backend Notes

- `backend="auto"` is recommended. It falls back safely to NumPy.
- MPI requires `mpi4py` and launching with `mpiexec`.
- `backend="cython"` is optional and speed-oriented, but it requires compiled Cython extensions:
  - `pip install cython`
  - `python setup.py build_ext --inplace`

Cache-stride validation helper (one command):

```bash
python -m pyche.cache_validation --stride-test 4 --endoftime 2000
```

For safer `interp_cache_guard_stride>1`, two adaptive guard triggers are available:

- `interp_cache_guard_force_below_zeta` (default `0.005`): always guard below this metallicity.
- `interp_cache_guard_zeta_trigger` (default `2e-4`): force guard if `zeta` changes quickly.

## Documentation

- `docs/INSTALL.md`
- `docs/TUTORIALS.md`
- `examples/diagnostic_plots.ipynb`
- `examples/mpi_cython_benchmark.ipynb`
