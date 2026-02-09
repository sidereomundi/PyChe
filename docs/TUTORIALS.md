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

## 2) Compute diagnostics and quick plots from in-memory arrays

```python
from pyche import GCEModel
from pyche.diagnostics import diagnostics_from_tables
import matplotlib.pyplot as plt

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
    write_output=False,
    return_results=True,
)

diag = diagnostics_from_tables(res.mod, res.fis)
print(diag)

fis_cols = {name: i for i, name in enumerate(res.fis_columns)}
t = res.fis[:, fis_cols["time"]]
sfr = res.fis[:, fis_cols["sfr"]]
zeta = res.fis[:, fis_cols["zeta"]]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(t, sfr)
ax[0].set_title("SFR vs Time")
ax[1].plot(t, zeta)
ax[1].set_title("Zeta vs Time")
plt.tight_layout()
```

## 3) Run with files and create diagnostic plots

```python
from pyche import GCEModel, create_diagnostic_plots, read_outputs

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

# Load the saved run back into Python
payload = read_outputs(out_dir, prefer="dataframe", binary_format="pickle")
print(payload["mod"].shape, payload["fis"].shape, payload["format"])
```

## 4) Cython backend example (single process)

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

Before forcing `backend="cython"`, compile extensions:

```bash
pip install cython
python setup.py build_ext --inplace
```

## 5) MPI example (recommended with auto backend)

```bash
mpiexec -n 8 python -c "from pyche import GCEModel; m=GCEModel(); m.MinGCE(13700,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=True,show_progress=False,backend='auto',output_dir='RISULTATI_MPI',output_mode='dataframe',df_binary_format='pickle')"
```

## 6) MPI + Cython example

```bash
python setup.py build_ext --inplace
mpiexec -n 8 python -c "from pyche import GCEModel; m=GCEModel(); m.MinGCE(13700,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=True,show_progress=False,backend='cython',output_dir='RISULTATI_MPI_CYTHON',output_mode='dataframe',df_binary_format='pickle')"
```
