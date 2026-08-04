""".enc file format: portable encrypted arrays.

    b"CENC" | u8 version=1 | u32 header_len | JSON header
           | per-vector: u32 blob_len | tenseal blob
    header: {"shape": [N, d], "packing": "slots"|"scalars",
             "scheme": "ckks", "ctx_fp": <context fingerprint>}

The fingerprint ties a file to the key pair that produced it, so loading
with the wrong context fails at open time instead of decrypting garbage.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import tenseal as ts

from .context import Context, default_context
from .core import CipherArray

_MAGIC = b"CENC"
_VERSION = 1


def save(arr: CipherArray, path) -> None:
    header = json.dumps(
        {
            "shape": list(arr.shape),
            "packing": arr._packing,
            "scheme": "ckks",
            "ctx_fp": arr.context.fingerprint,
        }
    ).encode()
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<BI", _VERSION, len(header)))
        f.write(header)
        for vector in arr._vectors:
            blob = vector.serialize()
            f.write(struct.pack("<I", len(blob)))
            f.write(blob)


def load(path, ctx: Context | None = None) -> CipherArray:
    if ctx is None:
        ctx = default_context()
    data = Path(path).read_bytes()
    if data[:4] != _MAGIC:
        raise ValueError(
            f"{path} is not a pypdpg .enc file (bad magic bytes)."
        )
    version, header_len = struct.unpack("<BI", data[4:9])
    if version != _VERSION:
        raise ValueError(f"{path}: unsupported .enc version {version}")
    offset = 9 + header_len
    header = json.loads(data[9:offset])
    if header.get("ctx_fp") != ctx.fingerprint:
        raise ValueError(
            f"{path} was encrypted under context {header.get('ctx_fp')}, "
            f"but the active context is {ctx.fingerprint}. Activate the "
            "matching context file before loading."
        )
    vectors = []
    while offset < len(data):
        (blob_len,) = struct.unpack("<I", data[offset : offset + 4])
        offset += 4
        vectors.append(ts.ckks_vector_from(ctx.ts, data[offset : offset + blob_len]))
        offset += blob_len
    return CipherArray(vectors, tuple(header["shape"]), header["packing"], ctx)
