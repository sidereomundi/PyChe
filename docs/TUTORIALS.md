# Tutorials

## MinGCE argument mapping

For calls like:

```python
m.MinGCE(500, 3000.0, 50.0, 0.3, 0.0, 10000, 10000, ...)
```

the variables are:

1. `500` -> `endoftime`
2. `3000.0` -> `sigmat`
3. `50.0` -> `sigmah`
4. `0.3` -> `psfr`
5. `0.0` -> `pwind`
6. `10000` -> `delay`
7. `10000` -> `time_wind`

Recommended style is to call with keyword args:

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

## 1) Run and return arrays directly

```python
from pyche import GCEModel

m = GCEModel()
res = m.MinGCE(
    endoftime=500,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
    use_mpi=False,
    show_progress=False,
    output_mode="dataframe",
    write_output=False,
    return_results=True,
)

print(res.mod.shape, res.fis.shape)
```

## 2) Run with files and create diagnostic plots

```python
from pyche import GCEModel, create_diagnostic_plots

out_dir = "RISULTATI_PYCHE"
m = GCEModel()
m.MinGCE(
    endoftime=1000,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
    use_mpi=False,
    show_progress=False,
    output_dir=out_dir,
    output_mode="dataframe",
    df_binary_format="pickle",
    df_write_csv=False,
    write_output=True,
    return_results=False,
)

paths = create_diagnostic_plots(out_dir, prefer="dataframe", binary_format="pickle")
print(paths)
```

## 3) Cython backend example (single process)

```python
from pyche import GCEModel

m = GCEModel()
m.MinGCE(
    endoftime=1000,
    sigmat=3000.0,
    sigmah=50.0,
    psfr=0.3,
    pwind=0.0,
    delay=10000,
    time_wind=10000,
    use_mpi=False,
    show_progress=False,
    backend="cython",  # requires compiled Cython extension support
    output_mode="dataframe",
    df_binary_format="pickle",
    write_output=True,
)
```

If you are unsure whether Cython extensions are available on your machine, use `backend="auto"` instead.

## 4) MPI example (recommended with auto backend)

```bash
mpiexec -n 8 python -c "from pyche import GCEModel; m=GCEModel(); m.MinGCE(13700,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=True,show_progress=False,backend='auto',output_dir='RISULTATI_MPI',output_mode='dataframe',df_binary_format='pickle')"
```

## 5) MPI + Cython example

```bash
mpiexec -n 8 python -c "from pyche import GCEModel; m=GCEModel(); m.MinGCE(13700,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=True,show_progress=False,backend='cython',output_dir='RISULTATI_MPI_CYTHON',output_mode='dataframe',df_binary_format='pickle')"
```
