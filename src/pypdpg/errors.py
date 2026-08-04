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
    "E-DIV": (
        "CKKS has no ciphertext division — computing a reciprocal would "
        "require reading the value, and this party holds no secret key.",
        "multiply by a plain reciprocal (x * (1/c)), or return the "
        "denominator to the data owner to decrypt.",
    ),
    "E-TRANSCEND": (
        "only polynomial functions (adds and multiplies) exist under FHE — "
        "transcendental functions have no homomorphic circuit.",
        "use pdpg.approx.sigmoid(x), or fit your own low-degree polynomial "
        "to the function over your input range.",
    ),
    "E-ORDER": (
        "with the plaintext on the left, numpy leads the computation, and "
        "pypdpg's column packing cannot serve the transposed product.",
        "reorder so the encrypted array comes first: X @ w instead of w @ X.",
    ),
    "E-INDEX": (
        "row selection would have to move or mask individual slots inside a "
        "packed ciphertext, and boolean masks would need comparisons — "
        "impossible without the secret key.",
        "select whole columns instead — X[:, j] or X[:, [i, k]] — or return "
        "the array to the data owner for row-level work.",
    ),
    "E-UNSUPPORTED": (
        "CKKS evaluates additions and multiplications only; this operation "
        "has no homomorphic form here.",
        "return the encrypted result to the data owner, or approximate "
        "with a polynomial (see pdpg.approx).",
    ),
}

# numpy op name -> catalog code. Grows with the teaching-error catalog;
# anything unlisted falls back to E-UNSUPPORTED.
_OP_TO_CODE = {
    "true_divide": "E-DIV",
    "divide": "E-DIV",
    "power": "E-TRANSCEND",
}

# symbol shown next to the op name when there is an operator spelling
_OP_SYMBOL = {
    "true_divide": "/",
    "divide": "/",
}


def make(code: str, headline: str) -> EncryptedOperationError:
    why, do = _CATALOG[code]
    message = f"{headline}\nWhy: {why}\nDo instead: {do}\n{_SUPPORTED}"
    return EncryptedOperationError(code, message)


def for_numpy(name: str) -> EncryptedOperationError:
    """Teaching error for a numpy ufunc/function hitting ciphertext."""
    code = _OP_TO_CODE.get(name, "E-UNSUPPORTED")
    symbol = f" ({_OP_SYMBOL[name]})" if name in _OP_SYMBOL else ""
    headline = f"np.{name}{symbol} is impossible on CKKS ciphertexts."
    return make(code, headline)
