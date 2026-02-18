from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from pyche.main import GCEModel


def _has_jax() -> bool:
    return importlib.util.find_spec("jax") is not None


@unittest.skipUnless(_has_jax(), "JAX is not installed")
class TestJaxFullBackend(unittest.TestCase):
    def test_gce_jax_full_runs(self) -> None:
        model = GCEModel()
        res = model.GCE(
            endoftime=200,
            sigmat=1000.0,
            sigmah=50.0,
            psfr=0.3,
            pwind=0.0,
            delay=1000,
            time_wind=10000,
            use_mpi=False,
            show_progress=False,
            backend="jax_full",
            output_mode="dataframe",
            write_output=False,
            return_results=True,
            adaptive_timestep=False,
            interp_cache=False,
            interp_cache_guard=False,
            profile_timing=False,
        )
        assert res is not None
        self.assertEqual(res.mod.shape[0], 200)
        self.assertEqual(res.fis.shape[0], 200)
        self.assertTrue(np.isfinite(res.mod).all())
        self.assertTrue(np.isfinite(res.fis).all())
        self.assertGreaterEqual(float(np.min(res.fis[:, 2])), 0.0)


if __name__ == "__main__":
    unittest.main()
