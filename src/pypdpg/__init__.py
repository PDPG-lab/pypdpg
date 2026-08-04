"""pypdpg — drop-in encrypted numpy.

The data controller encrypts, the data processor computes on ciphertext
with unmodified numpy code, the controller decrypts. Nobody in the middle
can read a thing.
"""

from . import approx, sklearn
from . import pandas  # noqa: F401  (pdpg.pandas — facade, not the real thing)
from .context import Context, activate, default_context
from .core import CipherArray, encrypt
from .errors import EncryptedOperationError
from .io import load
from .pandas import CipherFrame, CipherSeries
from .patch import install, uninstall

__version__ = "0.1.0"

__all__ = [
    "approx",
    "pandas",
    "sklearn",
    "Context",
    "CipherArray",
    "CipherFrame",
    "CipherSeries",
    "EncryptedOperationError",
    "activate",
    "default_context",
    "encrypt",
    "install",
    "load",
    "uninstall",
]
