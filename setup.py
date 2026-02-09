from __future__ import annotations

import sys

from setuptools import Extension, setup


def _build_extensions():
    import numpy as np

    try:
        from Cython.Build import cythonize
    except Exception as exc:
        raise RuntimeError("Cython is required for building extensions. Install with: pip install cython") from exc

    ext_modules = [
        Extension("pyche._cyinterp", ["src/pyche/_cyinterp.pyx"], include_dirs=[np.get_include()]),
        Extension("pyche._cyengine", ["src/pyche/_cyengine.pyx"], include_dirs=[np.get_include()]),
    ]
    return cythonize(ext_modules, compiler_directives={"language_level": "3"})


# Keep regular pip installs lightweight (no build-time numpy/cython needed).
# Compile extensions only when the user explicitly asks for build_ext.
if "build_ext" in sys.argv:
    setup(ext_modules=_build_extensions())
else:
    setup()
