"""CKKS context wrapper and key custody.

File format (both full and public saves):
    b"CCTX" | u8 version=1 | u32 header_len | JSON header | tenseal context blob
    header: {"fp": <16-hex-char fingerprint>, "private": bool}

The fingerprint is computed once at Context.create() from the serialized
public context and then *carried* in every saved file. It is never
recomputed from re-serialized bytes (SEAL expands seeded keys on load, so
those bytes are not stable across parties).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import tenseal as ts

_MAGIC = b"CCTX"
_VERSION = 1

# CKKS defaults: ~128-bit security, multiplicative depth 4, 8192 SIMD slots.
_POLY_MODULUS_DEGREE = 16384
_COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 40, 60]
_GLOBAL_SCALE = 2**40

MAX_ROWS = _POLY_MODULUS_DEGREE // 2  # 8192 slots per packed vector


class Context:
    """Wraps a TenSEAL context and tracks its identity across parties."""

    def __init__(self, ts_ctx: "ts.Context", fingerprint: str):
        self._ctx = ts_ctx
        self._fp = fingerprint

    @classmethod
    def create(cls) -> "Context":
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=_POLY_MODULUS_DEGREE,
            coeff_mod_bit_sizes=_COEFF_MOD_BIT_SIZES,
        )
        ctx.global_scale = _GLOBAL_SCALE
        ctx.generate_galois_keys()  # required for sum(); must precede save_public
        ctx.generate_relin_keys()
        fp = hashlib.sha256(cls._serialize_public(ctx)).hexdigest()[:16]
        return cls(ctx, fp)

    @staticmethod
    def _serialize_public(ts_ctx) -> bytes:
        return ts_ctx.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True,
        )

    @property
    def has_secret(self) -> bool:
        return self._ctx.is_private()

    @property
    def fingerprint(self) -> str:
        return self._fp

    @property
    def ts(self) -> "ts.Context":
        """The underlying TenSEAL context."""
        return self._ctx

    def save(self, path) -> None:
        """Save the full context, secret key included. The data controller keeps this."""
        blob = self._ctx.serialize(
            save_public_key=True,
            save_secret_key=True,
            save_galois_keys=True,
            save_relin_keys=True,
        )
        self._write(path, blob, private=True)

    def save_public(self, path) -> None:
        """Save the evaluation context (no secret key). Safe to ship out."""
        self._write(path, self._serialize_public(self._ctx), private=False)

    def _write(self, path, blob: bytes, private: bool) -> None:
        header = json.dumps({"fp": self._fp, "private": private}).encode()
        with open(path, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<BI", _VERSION, len(header)))
            f.write(header)
            f.write(blob)

    @classmethod
    def load(cls, path) -> "Context":
        data = Path(path).read_bytes()
        if data[:4] != _MAGIC:
            raise ValueError(
                f"{path} is not a pypdpg context file (bad magic bytes). "
                "Expected a file written by ctx.save() or ctx.save_public()."
            )
        version, header_len = struct.unpack("<BI", data[4:9])
        if version != _VERSION:
            raise ValueError(f"{path}: unsupported context file version {version}")
        header = json.loads(data[9 : 9 + header_len])
        ts_ctx = ts.context_from(data[9 + header_len :])
        return cls(ts_ctx, header["fp"])

    def __repr__(self) -> str:
        key = "present" if self.has_secret else "absent"
        return f"<pypdpg.Context CKKS · fp={self._fp} · secret_key={key}>"


# --- process-global default context (what pdpg.activate sets) ---

_default_context: Context | None = None


def activate(path) -> Context:
    """Load a context file and make it the process-global default."""
    global _default_context
    _default_context = Context.load(path)
    return _default_context


def default_context() -> Context:
    if _default_context is None:
        raise RuntimeError(
            "No active pypdpg context. Call pdpg.activate(<context file>) first "
            '— e.g. pdpg.activate("processor.ctx").'
        )
    return _default_context
