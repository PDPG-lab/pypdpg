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

_NUMBER = (int, float, np.integer, np.floating)


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

    # --- arithmetic machinery ---

    def _new(self, vectors: list) -> "CipherArray":
        return CipherArray(vectors, self._shape, self._packing, self._ctx)

    def _require_compatible(self, other: "CipherArray") -> None:
        if self._shape != other._shape or self._packing != other._packing:
            raise ValueError(
                f"CipherArray shapes must match exactly: {self._shape} vs "
                f"{other._shape} (encrypted broadcasting is not supported)."
            )
        if self._ctx.fingerprint != other._ctx.fingerprint:
            raise ValueError(
                "These CipherArrays were encrypted under different contexts "
                f"({self._ctx.fingerprint} vs {other._ctx.fingerprint}); "
                "they cannot be combined."
            )

    def _plain_operands(self, other):
        """Map a plaintext operand to one operand per stored vector.

        Returns a list (scalar or list-of-floats per vector), or None when
        the operand type isn't a plaintext we can handle.
        """
        if isinstance(other, _NUMBER):
            return [float(other)] * len(self._vectors)
        if isinstance(other, (list, tuple, np.ndarray)):
            a = np.asarray(other, dtype=np.float64)
        else:
            return None
        if a.shape == ():
            return [float(a)] * len(self._vectors)
        if self._packing == "scalars":
            if a.shape == self._shape:
                return [float(x) for x in a.ravel()]
            raise ValueError(
                f"cannot broadcast plain shape {a.shape} onto encrypted "
                f"reduction result of shape {self._shape}."
            )
        if self.ndim == 1:
            (n,) = self._shape
            if a.shape == (n,):
                return [a.tolist()]
            if a.shape == (1,):
                return [float(a[0])]
        else:
            n, d = self._shape
            if a.shape == (d,):  # numpy row-broadcast: one scalar per column
                return [float(a[j]) for j in range(d)]
            if a.shape in ((n,), (n, 1)):  # per-row: same plain column everywhere
                col = a.ravel().tolist()
                return [col] * d
            if a.shape == (n, d):
                return [a[:, j].tolist() for j in range(d)]
        raise ValueError(
            f"cannot broadcast plain shape {a.shape} onto encrypted "
            f"shape {self._shape}."
        )

    def _binary(self, other, op: str):
        if isinstance(other, CipherArray):
            self._require_compatible(other)
            pairs = zip(self._vectors, other._vectors)
            if op == "add":
                return self._new([a + b for a, b in pairs])
            if op == "sub":
                return self._new([a - b for a, b in pairs])
            if op == "rsub":
                return self._new([b - a for a, b in pairs])
            return self._new([a * b for a, b in pairs])
        operands = self._plain_operands(other)
        if operands is None:
            return NotImplemented
        pairs = zip(self._vectors, operands)
        if op == "add":
            return self._new([v + o for v, o in pairs])
        if op == "sub":
            return self._new([v - o for v, o in pairs])
        if op == "rsub":  # o - v == -(v - o); neg() is depth-free
            return self._new([(v - o).neg() for v, o in pairs])
        return self._new([v * o for v, o in pairs])

    def _divide(self, other):
        if isinstance(other, CipherArray):
            raise errors.for_numpy("true_divide")
        if isinstance(other, _NUMBER):
            return self._binary(1.0 / float(other), "mul")
        if isinstance(other, (list, tuple, np.ndarray)):
            return self._binary(1.0 / np.asarray(other, dtype=np.float64), "mul")
        return NotImplemented

    # --- operator dunders (used when numpy isn't on the left) ---

    def __add__(self, other):
        return self._binary(other, "add")

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, "sub")

    def __rsub__(self, other):
        return self._binary(other, "rsub")

    def __mul__(self, other):
        return self._binary(other, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._divide(other)

    def __rtruediv__(self, other):
        raise errors.for_numpy("true_divide")

    def __neg__(self):
        return self._new([v.neg() for v in self._vectors])

    # --- numpy dispatch ---

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        name = ufunc.__name__
        if method != "__call__" or kwargs.get("out") is not None:
            raise errors.for_numpy(name)
        if name in ("add", "multiply"):
            a, b = inputs
            other = b if a is self else a
            return self._binary(other, "add" if name == "add" else "mul")
        if name == "subtract":
            a, b = inputs
            return self._binary(b, "sub") if a is self else self._binary(a, "rsub")
        if name == "true_divide":
            a, b = inputs
            if a is self:
                return self._divide(b)
            raise errors.for_numpy("true_divide")
        if name == "negative":
            return -self
        raise errors.for_numpy(name)


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
