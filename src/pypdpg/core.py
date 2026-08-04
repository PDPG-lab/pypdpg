"""CipherArray: a numpy-shaped view over CKKS ciphertext.

Column-packed storage: an (N, d) array is d ckks_vectors, each packing one
column of length N ("slots" packing). Reductions produce one single-slot
vector per column ("scalars" packing).
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import tenseal as ts

from . import errors
from .context import MAX_ROWS, Context, default_context

_NUMBER = (int, float, np.integer, np.floating)

_DEPTH_MARKERS = ("scale out of bounds", "end of modulus switching chain")


@contextmanager
def _depth_guard():
    """Translate TenSEAL's out-of-levels error into a teaching error."""
    try:
        yield
    except ValueError as e:
        if any(marker in str(e) for marker in _DEPTH_MARKERS):
            raise errors.make(
                "E-DEPTH",
                "this multiplication exceeded the multiplicative depth (4) "
                "of the CKKS context.",
            ) from None
        raise


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

    def save(self, path) -> None:
        """Write this array as a portable .enc file."""
        from . import io as _io  # late import; io builds on core

        _io.save(self, path)

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
            with _depth_guard():
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
        with _depth_guard():
            return self._new([v * o for v, o in pairs])

    def _divide(self, other):
        if isinstance(other, CipherArray):
            raise errors.for_numpy("true_divide")
        if isinstance(other, _NUMBER):
            return self._binary(1.0 / float(other), "mul")
        if isinstance(other, (list, tuple, np.ndarray)):
            return self._binary(1.0 / np.asarray(other, dtype=np.float64), "mul")
        return NotImplemented

    # --- linalg & stats ---

    def _matmul_plain(self, other):
        if isinstance(other, CipherArray):
            raise errors.make(
                "E-UNSUPPORTED",
                "np.matmul (@) between two encrypted arrays is impossible "
                "on CKKS ciphertexts here.",
            )
        if not isinstance(other, (list, tuple, np.ndarray)):
            return NotImplemented
        if self._packing != "slots":
            raise ValueError("matmul on an already-reduced CipherArray is not supported.")
        with _depth_guard():
            return self._matmul_columns(np.asarray(other, dtype=np.float64))

    def _matmul_columns(self, w: np.ndarray):
        if self.ndim == 1:
            (n,) = self._shape
            if w.shape != (n,):
                raise ValueError(f"matmul shape mismatch: {self._shape} @ {w.shape}")
            return CipherArray([self._vectors[0].dot(w.tolist())], (), "scalars", self._ctx)
        n, d = self._shape
        if w.shape == (d,):
            out = self._vectors[0] * float(w[0])
            for j in range(1, d):
                out = out + self._vectors[j] * float(w[j])
            return CipherArray([out], (n,), "slots", self._ctx)
        if w.ndim == 2 and w.shape[0] == d:
            cols = []
            for c in range(w.shape[1]):
                acc = self._vectors[0] * float(w[0, c])
                for j in range(1, d):
                    acc = acc + self._vectors[j] * float(w[j, c])
                cols.append(acc)
            return CipherArray(cols, (n, w.shape[1]), "slots", self._ctx)
        raise ValueError(f"matmul shape mismatch: {self._shape} @ {w.shape}")

    def square(self) -> "CipherArray":
        with _depth_guard():
            return self._new([v.square() for v in self._vectors])

    def _polyval(self, coeffs) -> "CipherArray":
        """Evaluate a polynomial (lowest-degree coefficient first)."""
        with _depth_guard():
            return self._new([v.polyval(list(coeffs)) for v in self._vectors])

    def sum(self, axis=None) -> "CipherArray":
        if self._packing != "slots":
            raise ValueError("sum() on an already-reduced CipherArray is not supported.")
        if axis not in (None, 0):
            raise ValueError(f"axis={axis} is not supported; use axis=0 or axis=None.")
        col_sums = [v.sum() for v in self._vectors]
        if self.ndim == 1 or axis is None:
            total = col_sums[0]
            for v in col_sums[1:]:
                total = total + v
            return CipherArray([total], (), "scalars", self._ctx)
        return CipherArray(col_sums, (self._shape[1],), "scalars", self._ctx)

    def mean(self, axis=None) -> "CipherArray":
        count = self._shape[0] if axis == 0 else int(np.prod(self._shape))
        return self.sum(axis=axis) * (1.0 / count)

    # --- column slicing ---

    def __getitem__(self, key):
        if (
            self._packing == "slots"
            and self.ndim == 2
            and isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], slice)
            and key[0] == slice(None)
        ):
            n, d = self._shape
            csel = key[1]
            if isinstance(csel, (bool, np.bool_)):
                pass  # fall through to the teaching error
            elif isinstance(csel, (int, np.integer)):
                return CipherArray([self._vectors[csel]], (n,), "slots", self._ctx)
            elif isinstance(csel, slice):
                vecs = self._vectors[csel]
                return CipherArray(vecs, (n, len(vecs)), "slots", self._ctx)
            elif isinstance(csel, (list, tuple, np.ndarray)):
                idx = np.asarray(csel)
                if idx.dtype != bool:
                    vecs = [self._vectors[int(i)] for i in idx]
                    return CipherArray(vecs, (n, len(vecs)), "slots", self._ctx)
        raise errors.make(
            "E-INDEX",
            f"indexing a CipherArray with {key!r} is impossible on CKKS "
            "ciphertexts (columns only).",
        )

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

    def __matmul__(self, other):
        return self._matmul_plain(other)

    def __rmatmul__(self, other):
        raise errors.make(
            "E-ORDER",
            "np.matmul (@) with the plaintext on the left of a CipherArray "
            "is impossible on CKKS ciphertexts.",
        )

    def __pow__(self, n):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise errors.for_numpy("power")
        # square-and-multiply; each squaring or multiply costs one depth level
        e = int(n)
        base, result = self, None
        while e:
            if e & 1:
                result = base if result is None else result * base
            e >>= 1
            if e:
                base = base.square()
        return result

    # --- bare-operator traps ---
    # Pure-Python expressions (enc > 0, abs(enc), bool(enc)) never enter
    # numpy's dispatch; Python calls these dunders directly. Left undefined,
    # __eq__ would silently "succeed" with identity semantics — a wrong
    # answer instead of a refusal.

    def __gt__(self, other):
        raise errors.for_numpy("greater")

    def __lt__(self, other):
        raise errors.for_numpy("less")

    def __ge__(self, other):
        raise errors.for_numpy("greater_equal")

    def __le__(self, other):
        raise errors.for_numpy("less_equal")

    def __eq__(self, other):
        raise errors.for_numpy("equal")

    def __ne__(self, other):
        raise errors.for_numpy("not_equal")

    __hash__ = None  # comparisons raise, so hashing is off the table too

    def __abs__(self):
        raise errors.for_numpy("absolute")

    def __floordiv__(self, other):
        raise errors.for_numpy("floor_divide")

    __rfloordiv__ = __floordiv__

    def __mod__(self, other):
        raise errors.for_numpy("remainder")

    __rmod__ = __mod__

    def __bool__(self):
        raise errors.make(
            "E-COMPARE",
            "bool(X) — truth-testing is impossible on CKKS ciphertexts.",
        )

    def __array__(self, *args, **kwargs):
        raise errors.make(
            "E-COERCE",
            "np.asarray(X) — coercing a CipherArray to a plain ndarray is "
            "impossible on this side of the trust boundary.",
        )

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
        if name == "square":
            return self.square()
        if name == "matmul":
            a, b = inputs
            if a is self:
                return self._matmul_plain(b)
            return self.__rmatmul__(a)
        if name == "power":
            a, b = inputs
            if a is self:
                return self.__pow__(b)
            raise errors.for_numpy("power")
        raise errors.for_numpy(name)

    def __array_function__(self, func, types, args, kwargs):
        if func in (np.dot, np.matmul):
            if args and args[0] is self:
                return self._matmul_plain(args[1])
            return self.__rmatmul__(args[0])
        if func in (np.sum, np.mean):
            axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
            return self.sum(axis=axis) if func is np.sum else self.mean(axis=axis)
        raise errors.for_numpy(getattr(func, "__name__", str(func)))


def encrypt(arr, ctx: Context | None = None):
    """Encrypt a 1-D or 2-D float array column-by-column.

    DataFrame-like inputs (anything with .columns and .to_numpy()) come
    back as a CipherFrame with their column names preserved.
    """
    if hasattr(arr, "columns") and hasattr(arr, "to_numpy"):
        from .pandas import encrypt_frame  # late import; avoids module cycle

        return encrypt_frame(arr, ctx)
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
