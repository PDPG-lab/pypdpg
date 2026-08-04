"""CipherArray: a numpy-shaped view over CKKS ciphertext.

Column-packed storage: an (N, d) array is d ckks_vectors, each packing one
column of length N ("slots" packing). Reductions produce one single-slot
vector per column ("scalars" packing).
"""

from __future__ import annotations

import numpy as np
import tenseal as ts

from . import errors
from .context import MAX_ROWS, Context, default_context


class CipherArray:
    def __init__(self, vectors: list, shape: tuple, packing: str, ctx: Context):
        self._vectors = vectors
        self._shape = tuple(shape)
        self._packing = packing  # "slots" | "scalars"
        self._ctx = ctx

    # --- introspection (never leaks slot values) ---

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def context(self) -> Context:
        return self._ctx

    def __repr__(self) -> str:
        key = "present" if self._ctx.has_secret else "absent"
        return f"<CipherArray shape={self._shape} 🙈 CKKS · secret_key={key}>"

    # --- custody boundary ---

    def decrypt(self) -> np.ndarray:
        if not self._ctx.has_secret:
            raise errors.make(
                "E-CUSTODY",
                ".decrypt() was called on a context without the secret key.",
            )
        if self._packing == "slots":
            cols = [np.asarray(v.decrypt()) for v in self._vectors]
            if self.ndim == 1:
                return cols[0]
            return np.column_stack(cols)
        # "scalars": one single-slot vector per value
        vals = np.asarray([v.decrypt()[0] for v in self._vectors])
        if self._shape == ():
            return vals[0]
        return vals


def encrypt(arr, ctx: Context | None = None) -> CipherArray:
    """Encrypt a 1-D or 2-D float array column-by-column."""
    if ctx is None:
        ctx = default_context()
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim not in (1, 2):
        raise ValueError(
            f"pypdpg supports 1-D and 2-D arrays only, got ndim={a.ndim}."
        )
    n = a.shape[0]
    if n > MAX_ROWS:
        raise ValueError(
            f"Array has {n} rows but a ciphertext packs at most {MAX_ROWS} "
            f"SIMD slots. Encrypt in chunks of <= {MAX_ROWS} rows."
        )
    if a.ndim == 1:
        vectors = [ts.ckks_vector(ctx.ts, a.tolist())]
    else:
        vectors = [ts.ckks_vector(ctx.ts, a[:, j].tolist()) for j in range(a.shape[1])]
    return CipherArray(vectors, a.shape, "slots", ctx)
