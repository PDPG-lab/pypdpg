"""Teaching errors: every impossible operation explains itself.

One exception class, one catalog. Each entry says what you tried, why it
cannot work on CKKS ciphertexts, and what to do instead.
"""

_SUPPORTED = (
    "Supported here: + - * @ dot sum mean square / scalar, pdpg.approx.sigmoid"
)


class EncryptedOperationError(TypeError):
    """Raised when an operation is impossible (or forbidden) on ciphertext."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


# code -> (why, do_instead). The headline is built per call site so it can
# name the exact op the user tried.
_CATALOG = {
    "E-CUSTODY": (
        "decryption requires the secret key, and this context holds only "
        "public evaluation keys. That's the point.",
        "save the encrypted result and return it to the data owner — only "
        'their context (e.g. pdpg.activate("orga.key")) can decrypt.',
    ),
}


def make(code: str, headline: str) -> EncryptedOperationError:
    why, do = _CATALOG[code]
    message = f"{headline}\nWhy: {why}\nDo instead: {do}\n{_SUPPORTED}"
    return EncryptedOperationError(code, message)
