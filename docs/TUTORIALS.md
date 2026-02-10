# PyChe Full Tutorial

This guide is a complete practical tutorial for running PyChe in:
- serial mode
- MPI mode from shell
- MPI mode directly from notebooks with in-memory returned results

It also covers diagnostics, plotting, and performance testing.

## 1) Install

From the repo root:

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e .[all]
```

Cython optional build (recommended for speed):

```bash
pip install cython
python setup.py build_ext --inplace
```

## 2) Core model call and argument mapping

Positional form:

```python
m.MinGCE(500, 3000.0, 50.0, 0.3, 0.0, 10000, 10000, ...)
```

maps to:

1. `endoftime`
2. `sigmat`
3. `sigmah`
4. `psfr`
5. `pwind`
6. `delay`
7. `time_wind`

Use keyword arguments for clarity:

```python
m.MinGCE(
    endoftime=500,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
)
```

## 3) Serial in-memory run

```python
from pyche import GCEModel

m = GCEModel()
res = m.MinGCE(
    endoftime=13700,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
    use_mpi=False,
    show_progress=True,
    backend="auto",
    output_mode="dataframe",
    write_output=False,
    return_results=True,
)

print(res.mod.shape, res.fis.shape)
```

## 4) MPI tutorial

### 4.1 MPI run from shell (production pattern)

```bash
mpiexec -n 4 python -c "from pyche import GCEModel; m=GCEModel(); m.MinGCE(13700,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=True,show_progress=False,backend='auto',output_dir='RISULTATI_MPI4',output_mode='dataframe',df_binary_format='pickle')"
```

### 4.2 MPI + Cython from shell

```bash
python setup.py build_ext --inplace
mpiexec -n 4 python -c "from pyche import GCEModel; m=GCEModel(); m.MinGCE(13700,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=True,show_progress=False,backend='cython',output_dir='RISULTATI_MPI4_CY',output_mode='dataframe',df_binary_format='pickle')"
```

### 4.3 MPI directly from notebook and return results as variables

Use `mpi_subprocess=True`:

```python
from pyche import GCEModel

m = GCEModel()
res = m.MinGCE(
    endoftime=13700,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
    use_mpi=True,
    mpi_subprocess=True,
    mpi_subprocess_ranks=4,
    show_progress=True,
    backend="auto",
    output_mode="dataframe",
    write_output=False,
    return_results=True,
)

print(res.mod.shape, res.fis.shape)
```

Notes:
- `mpi_subprocess=True` is for notebook convenience (in-memory return in the caller process).
- It has overhead vs pure shell `mpiexec` because it serializes results back to the notebook.
- For fastest benchmarking, use shell `mpiexec` + `show_progress=False`.

### 4.4 MPI troubleshooting

- If you see MPI implementation mismatch warnings, rebuild `mpi4py` with the same MPI used by `mpiexec`.
- If notebook output is buffered, use `mpi_subprocess=True` (already handled by parent-side progress polling).
- For clean timing comparisons, keep physics flags identical between serial and MPI runs.

## 5) Output modes

Common patterns:
- `write_output=False, return_results=True`: in-memory only.
- `write_output=True, return_results=False`: files only.
- `write_output=True, return_results=True`: both.

File mode example:

```python
from pyche import GCEModel, read_outputs

out_dir = "RISULTATI_PYCHE"
m = GCEModel()
m.MinGCE(
    endoftime=13700,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
    use_mpi=False,
    output_dir=out_dir,
    output_mode="dataframe",
    df_binary_format="pickle",
    write_output=True,
    return_results=False,
)
payload = read_outputs(out_dir, prefer="dataframe", binary_format="pickle")
print(payload["mod"].shape, payload["fis"].shape, payload["format"])
```

## 6) Diagnostics and plots

### 6.1 Diagnostics from in-memory arrays

```python
from pyche.diagnostics import diagnostics_from_tables

diag = diagnostics_from_tables(res.mod, res.fis)
print(diag)
```

### 6.2 Column names

```python
print("mod columns:", res.mod_columns)
print("fis columns:", res.fis_columns)
```

### 6.3 Full diagnostic plots from saved outputs

```python
from pyche import create_diagnostic_plots

paths = create_diagnostic_plots("RISULTATI_PYCHE", prefer="dataframe", binary_format="pickle")
print(paths)
```

For full plotting code from `res.mod` and `res.fis` (SFR/Zeta, mass budget, [Fe/H], [O/Fe], [Mg/Fe], MDF), use:
- `examples/diagnostic_plots.ipynb`

## 7) Baseline vs optimized comparison

You can compare a no-approx baseline to optimized settings:

```python
res_noapprox = m.MinGCE(
    endoftime=13700,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
    use_mpi=False,
    backend="auto",
    output_mode="dataframe",
    write_output=False,
    return_results=True,
    adaptive_timestep=False,
    interp_cache=False,
    interp_cache_guard=False,
    profile_timing=False,
    spalla_stride=1,
    spalla_inactive_threshold=0.0,
    spalla_lut=False,
)
```

Then compare tracks and MDF in `examples/diagnostic_plots.ipynb`.

## 8) Recommended speed workflow

1. Build Cython:
   - `python setup.py build_ext --inplace`
2. Start from `backend="auto"` (or force `cython` if known-good).
3. Disable progress for benchmarks:
   - `show_progress=False`
4. Use MPI shell mode for clean speed tests.
5. Use `profile_timing=True` to inspect `interp`, `wind`, `mpi_reduce`.

## 9) Cache validation tool

Validate cache stride behavior:

```bash
python -m pyche.cache_validation --stride-test 4 --endoftime 13700
```

If you tune stride:
- keep `interp_cache_guard=True`
- test against stride 1
- inspect O/Fe and Fe/H differences

