from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from pyche.main import GCEModel
from pyche.backends.factory import build_backend


def _has_jax() -> bool:
    return importlib.util.find_spec("jax") is not None


@unittest.skipUnless(_has_jax(), "JAX is not installed")
class TestJaxBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = GCEModel()
        cls.model._initialize_from_fortran_tables(lowmassive=1, mm=0)
        cls.tables = cls.model.tables
        assert cls.tables is not None

    def test_interp_matches_numpy_backend(self) -> None:
        numpy_backend = build_backend("numpy", self.tables)
        jax_backend = build_backend("jax", self.tables)

        points = [
            (0.3, 1.0e-4, 0.0),
            (1.5, 5.0e-4, 0.0),
            (2.2, 2.0e-3, 0.0),
            (8.0, 1.0e-3, 0.0),
            (15.0, 1.0e-2, 15.0),
            (25.0, 1.0e-4, 20.0),
        ]

        for mass, zeta, binmax in points:
            q_np, h_np = numpy_backend.interp(mass, zeta, binmax)
            q_jax, h_jax = jax_backend.interp(mass, zeta, binmax)
            np.testing.assert_allclose(q_jax, q_np, rtol=1.0e-12, atol=1.0e-14)
            self.assertAlmostEqual(h_jax, h_np, places=12)


if __name__ == "__main__":
    unittest.main()
