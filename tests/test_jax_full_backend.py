from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from pyche.main import GCEModel
from pyche.backends.factory import build_backend


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

    def test_jax_full_tracks_cython_observables(self) -> None:
        model = GCEModel()
        model._initialize_from_fortran_tables(lowmassive=1, mm=0)
        assert model.tables is not None
        try:
            build_backend("cython", model.tables)
        except Exception:
            self.skipTest("Cython backend is not available")

        kwargs = dict(
            endoftime=600,
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
            adaptive_timestep=False,
            interp_cache=False,
            interp_cache_guard=False,
            spalla_stride=1,
            spalla_inactive_threshold=0.0,
            spalla_lut=False,
            profile_timing=False,
        )

        cy = GCEModel().GCE(backend="cython", **kwargs)
        jx = GCEModel().GCE(backend="jax_full", **kwargs)
        assert cy is not None and jx is not None

        mod_cols = {n: i for i, n in enumerate(cy.mod_columns)}
        eps = 1.0e-30
        o_fe_sun = (16.0 / 56.0) * 10.0 ** (8.69 - 7.50)
        mg_fe_sun = (24.305 / 56.0) * 10.0 ** (7.60 - 7.50)

        fe_cy = cy.mod[:, mod_cols["Fe"]]
        o_cy = cy.mod[:, mod_cols["O16"]]
        mg_cy = cy.mod[:, mod_cols["Mg"]]
        fe_jx = jx.mod[:, mod_cols["Fe"]]
        o_jx = jx.mod[:, mod_cols["O16"]]
        mg_jx = jx.mod[:, mod_cols["Mg"]]

        log_o_fe_cy = np.log10(np.maximum(o_cy / np.maximum(fe_cy, eps), eps) / o_fe_sun)
        log_o_fe_jx = np.log10(np.maximum(o_jx / np.maximum(fe_jx, eps), eps) / o_fe_sun)
        log_mg_fe_cy = np.log10(np.maximum(mg_cy / np.maximum(fe_cy, eps), eps) / mg_fe_sun)
        log_mg_fe_jx = np.log10(np.maximum(mg_jx / np.maximum(fe_jx, eps), eps) / mg_fe_sun)

        self.assertLess(float(np.nanmax(np.abs(log_o_fe_jx - log_o_fe_cy))), 5.0e-6)
        self.assertLess(float(np.nanmax(np.abs(log_mg_fe_jx - log_mg_fe_cy))), 5.0e-6)


if __name__ == "__main__":
    unittest.main()
