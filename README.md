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

## Quick Start

```python
from pyche import GCEModel

m = GCEModel()
result = m.MinGCE(
    200, 2000.0, 54.0, 0.2, 0.0, 7000, 1000000,
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

## Outputs

You can choose file output, in-memory return, or both:

- `write_output=True`, `return_results=False`: write files only.
- `write_output=False`, `return_results=True`: return arrays only.
- `write_output=True`, `return_results=True`: both.

## Documentation

- `docs/INSTALL.md`
- `docs/TUTORIALS.md`
- `examples/diagnostic_plots.ipynb`
