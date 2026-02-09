"""Simple driver emulating ``src/driver.f90``."""

# When executed as a script ``__package__`` is ``None``.  Use an absolute
# import so that the module can be run both as ``python -m pyche.driver``
# and ``python pyche/driver.py``.
from pyche.main import GCEModel


def main() -> None:
    model = GCEModel()
    model.MinGCE(0, 0.0, 0.0, 0.0, 0.0, 0, 0)


if __name__ == "__main__":
    main()
