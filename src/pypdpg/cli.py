"""Command-line interface: the two-party flow without writing Python.

    pdpg keygen -o keys/
    pdpg encrypt applicants.csv -c keys/controller.key -o data.enc
    pdpg inspect data.enc
    pdpg decrypt result.enc -c keys/controller.key -o result.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np


def _cmd_keygen(args):
    from .context import Context

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context.create()
    ctx.save(out / "controller.key")
    ctx.save_public(out / "processor.ctx")
    print(f"wrote {out / 'controller.key'}  (secret key — data controller only)")
    print(f"wrote {out / 'processor.ctx'}  (evaluation keys — safe to ship)")
    print(f"context fingerprint: {ctx.fingerprint}")


def _read_csv(path):
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    if not rows:
        raise SystemExit(f"{path}: empty file")

    def numeric(row):
        try:
            [float(cell) for cell in row]
            return True
        except ValueError:
            return False

    columns = None
    if not numeric(rows[0]):
        columns, rows = rows[0], rows[1:]
    data = np.array([[float(cell) for cell in row] for row in rows])
    return data, columns


def _cmd_encrypt(args):
    from . import io as _io
    from .context import Context
    from .core import encrypt

    ctx = Context.load(args.context)
    data, columns = _read_csv(args.csv)
    out = args.out or str(Path(args.csv).with_suffix(".enc"))
    _io.save(encrypt(data, ctx), out, columns=columns)
    print(f"encrypted {data.shape[0]} rows x {data.shape[1]} columns -> {out}")


def _cmd_decrypt(args):
    from . import io as _io
    from .context import Context

    ctx = Context.load(args.context)
    obj = _io.load(args.enc, ctx)
    if hasattr(obj, "columns"):  # CipherFrame: keep the header row
        columns, values = obj.columns, obj.values.decrypt()
    else:
        columns, values = None, obj.decrypt()
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(-1, 1)
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        writer = csv.writer(out)
        if columns:
            writer.writerow(columns)
        writer.writerows(values.tolist())
    finally:
        if args.out:
            out.close()
            print(f"decrypted {values.shape[0]} rows -> {args.out}")


def _cmd_inspect(args):
    data = Path(args.file).read_bytes()
    magic = data[:4]
    if magic not in (b"CENC", b"CCTX"):
        raise SystemExit(f"{args.file}: not a pypdpg file (magic {magic!r})")
    version, header_len = struct.unpack("<BI", data[4:9])
    header = json.loads(data[9 : 9 + header_len])
    if magic == b"CENC":
        print(f"pypdpg encrypted array, version {version}")
        print(f"  shape:   {tuple(header['shape'])}")
        print(f"  packing: {header['packing']}")
        if "columns" in header:
            print(f"  columns: {', '.join(header['columns'])}")
        print(f"  context: {header['ctx_fp']}")
        print(f"  size:    {len(data) / 1e6:.1f} MB")
        print("  values:  ciphertext — nothing to show without the key")
    else:
        kind = (
            "FULL context — contains the secret key, do not ship"
            if header["private"]
            else "public evaluation context — safe to ship"
        )
        print(f"pypdpg context file, version {version}")
        print(f"  kind:        {kind}")
        print(f"  fingerprint: {header['fp']}")
        print(f"  size:        {len(data) / 1e6:.1f} MB")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="pdpg",
        description="Encrypted numpy for the controller/processor split: "
        "generate keys, encrypt CSVs, inspect and decrypt .enc files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("keygen", help="create controller.key + processor.ctx")
    p.add_argument("-o", "--out-dir", default=".", help="output directory")
    p.set_defaults(func=_cmd_keygen)

    p = sub.add_parser("encrypt", help="encrypt a CSV into a .enc file")
    p.add_argument("csv", help="input CSV (optional header row)")
    p.add_argument("-c", "--context", required=True, help="context file")
    p.add_argument("-o", "--out", help="output path (default: <csv>.enc)")
    p.set_defaults(func=_cmd_encrypt)

    p = sub.add_parser("decrypt", help="decrypt a .enc file to CSV")
    p.add_argument("enc", help="input .enc file")
    p.add_argument("-c", "--context", required=True, help="context file with secret key")
    p.add_argument("-o", "--out", help="output CSV (default: stdout)")
    p.set_defaults(func=_cmd_decrypt)

    p = sub.add_parser("inspect", help="show a pypdpg file's header (no key needed)")
    p.add_argument("file", help=".enc or context file")
    p.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    from .errors import EncryptedOperationError

    try:
        args.func(args)
    except EncryptedOperationError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
