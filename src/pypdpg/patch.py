"""np.load interception: make `np.load("data.enc")` just work.

pdpg.install() wraps np.load with a 4-byte magic peek — .enc files come
back as CipherArrays, everything else falls through to the real np.load
untouched. pdpg.uninstall() restores the original.
"""

from __future__ import annotations

import os

import numpy as np

from . import io as _io

_real_np_load = None


def install() -> None:
    global _real_np_load
    if _real_np_load is not None:
        return  # already installed
    _real_np_load = np.load

    def _pdpg_np_load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and os.path.isfile(file):
            with open(file, "rb") as f:
                magic = f.read(4)
            if magic == _io._MAGIC:
                return _io.load(file)
        return _real_np_load(file, *args, **kwargs)

    np.load = _pdpg_np_load


def uninstall() -> None:
    global _real_np_load
    if _real_np_load is not None:
        np.load = _real_np_load
        _real_np_load = None
