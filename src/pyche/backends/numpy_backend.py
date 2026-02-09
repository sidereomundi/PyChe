"""Default NumPy backend for interpolation kernels."""

from __future__ import annotations

from ..interpolation_api import ModelInterpolator
from ..model_tables import ModelTables


def build_numpy_backend(tables: ModelTables):
    return ModelInterpolator(tables)
