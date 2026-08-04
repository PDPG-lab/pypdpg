"""pypdpg — drop-in encrypted numpy.

The data controller encrypts, the data processor computes on ciphertext
with unmodified numpy code, the controller decrypts. Nobody in the middle
can read a thing.
"""

from . import approx, sklearn
from .context import Context, activate, default_context
from .core import CipherArray, encrypt
from .errors import EncryptedOperationError
from .io import load
from .patch import install, uninstall

__version__ = "0.1.0"

__all__ = [
    "approx",
    "sklearn",
    "Context",
    "CipherArray",
    "EncryptedOperationError",
    "activate",
    "default_context",
    "encrypt",
    "install",
    "load",
    "uninstall",
]
