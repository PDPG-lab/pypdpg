"""Named-column views over encrypted arrays: a minimal pandas-like facade.

CipherFrame is not a pandas DataFrame — it is a thin layer of column names
over a CipherArray, enough that column-oriented analyst code reads
naturally: select by name, compute, reduce, decrypt to a real labeled
DataFrame. pandas itself is imported only at decrypt time and is not a
dependency of pypdpg.
"""

from __future__ import annotations

import numpy as np

from .core import CipherArray
from .core import encrypt as _encrypt_array


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "decrypting to labeled pandas objects requires pandas; "
            "pip install pandas (or use .values.decrypt() for a plain array)."
        ) from e
    return pd


class CipherSeries:
    """A reduction result: one encrypted scalar per label."""

    def __init__(self, data: CipherArray, index):
        self._data = data
        self._index = [str(i) for i in index]

    @property
    def values(self) -> CipherArray:
        return self._data

    def __repr__(self) -> str:
        key = "present" if self._data.context.has_secret else "absent"
        return (
            f"<CipherSeries [{', '.join(self._index)}] 🙈 CKKS · "
            f"secret_key={key}>"
        )

    def decrypt(self):
        return _pandas().Series(self._data.decrypt(), index=self._index)


class CipherFrame:
    """Encrypted 2-D data with named columns."""

    def __init__(self, data: CipherArray, columns):
        columns = [str(c) for c in columns]
        if data.ndim != 2 or data.shape[1] != len(columns):
            raise ValueError(
                f"need a 2-D CipherArray with one name per column: "
                f"shape {data.shape} vs {len(columns)} names."
            )
        self._data = data
        self._columns = columns

    # --- introspection ---

    @property
    def columns(self) -> list:
        return list(self._columns)

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def values(self) -> CipherArray:
        """The underlying CipherArray (for matmul, sklearn wrappers, …)."""
        return self._data

    def __repr__(self) -> str:
        key = "present" if self._data.context.has_secret else "absent"
        n, d = self._data.shape
        return (
            f"<CipherFrame {n}x{d} [{', '.join(self._columns)}] 🙈 CKKS · "
            f"secret_key={key}>"
        )

    # --- column access ---

    def _loc(self, name: str) -> int:
        try:
            return self._columns.index(name)
        except ValueError:
            raise KeyError(
                f"no column {name!r}; columns are {self._columns}"
            ) from None

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[:, self._loc(key)]
        if isinstance(key, list):
            idx = [self._loc(k) for k in key]
            return CipherFrame(self._data[:, idx], [self._columns[j] for j in idx])
        raise TypeError(
            "select columns by name: frame['col'] or frame[['a', 'b']]."
        )

    def __setitem__(self, name: str, value) -> None:
        if not isinstance(value, CipherArray) or value.ndim != 1:
            raise TypeError(
                "assign a 1-D CipherArray computed from encrypted columns, "
                "e.g. frame['ratio'] = frame['debt'] * 0.001."
            )
        n = self._data.shape[0]
        if value.shape[0] != n:
            raise ValueError(f"length mismatch: {value.shape[0]} vs {n} rows.")
        name = str(name)
        vectors = list(self._data._vectors)
        columns = list(self._columns)
        if name in columns:
            vectors[columns.index(name)] = value._vectors[0]
        else:
            vectors.append(value._vectors[0])
            columns.append(name)
        self._data = CipherArray(
            vectors, (n, len(columns)), "slots", self._data.context
        )
        self._columns = columns

    # --- compute ---

    def sum(self) -> CipherSeries:
        return CipherSeries(self._data.sum(axis=0), self._columns)

    def mean(self) -> CipherSeries:
        return CipherSeries(self._data.mean(axis=0), self._columns)

    def __matmul__(self, other) -> CipherArray:
        return self._data @ other

    def _rewrap(self, data):
        return CipherFrame(data, self._columns)

    def __add__(self, other):
        return self._rewrap(self._data + other)

    __radd__ = __add__

    def __sub__(self, other):
        return self._rewrap(self._data - other)

    def __mul__(self, other):
        return self._rewrap(self._data * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._rewrap(self._data / other)

    def __neg__(self):
        return self._rewrap(-self._data)

    # --- custody boundary & io ---

    def decrypt(self):
        return _pandas().DataFrame(self._data.decrypt(), columns=self._columns)

    def save(self, path) -> None:
        from . import io as _io  # late import; io builds on core

        _io.save(self._data, path, columns=self._columns)


def encrypt_frame(df, ctx=None) -> CipherFrame:
    """Encrypt anything DataFrame-like (has .columns and .to_numpy())."""
    columns = [str(c) for c in df.columns]
    data = _encrypt_array(np.asarray(df.to_numpy(), dtype=np.float64), ctx)
    return CipherFrame(data, columns)
