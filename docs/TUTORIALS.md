# Tutorials

## 1) Run and return arrays directly

```python
from pyche import GCEModel

m = GCEModel()
res = m.MinGCE(
    500, 3000.0, 50.0, 0.3, 0.0, 10000, 10000,
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
    1000, 3000.0, 50.0, 0.3, 0.0, 10000, 10000,
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

## 3) MPI example

```bash
mpiexec -n 8 python -c "from pyche import GCEModel; m=GCEModel(); m.MinGCE(13700,3000.0,50.0,0.3,0.0,10000,10000,use_mpi=True,show_progress=False,backend='auto',output_dir='RISULTATI_MPI',output_mode='dataframe',df_binary_format='pickle')"
```
