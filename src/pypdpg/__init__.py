"""pypdpg — drop-in encrypted numpy.

Org A encrypts, the vendor computes on ciphertext with unmodified numpy
code, Org A decrypts. Nobody in the middle can read a thing.
"""

from .context import Context, activate, default_context
from .core import CipherArray, encrypt
from .errors import EncryptedOperationError

__version__ = "0.1.0"

__all__ = [
    "Context",
    "CipherArray",
    "EncryptedOperationError",
    "activate",
    "default_context",
    "encrypt",
]
