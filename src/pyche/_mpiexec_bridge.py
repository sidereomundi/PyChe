"""Internal MPI bridge: run GCE under mpiexec and emit pickled result on rank 0."""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys

from .main import GCEModel

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None


def main() -> int:
    if len(sys.argv) < 2:
        print("missing payload", file=sys.stderr)
        return 2
    payload = sys.argv[1]
    if payload.startswith("@"):
        with open(payload[1:], "r", encoding="utf-8") as f:
            kwargs = json.load(f)
    else:
        kwargs = json.loads(base64.b64decode(payload.encode("ascii")).decode("utf-8"))
    model = GCEModel()
    res = model.GCE(**kwargs)

    rank = 0
    if MPI is not None:
        rank = int(MPI.COMM_WORLD.Get_rank())
    if rank == 0:
        result_path = os.getenv("PYCHE_MPI_RESULT_PATH", "").strip()
        if result_path:
            with open(result_path, "wb") as f:
                pickle.dump(res, f)
        else:
            blob = base64.b64encode(pickle.dumps(res)).decode("ascii")
            print(f"PYCHE_RESULT_BASE64:{blob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
