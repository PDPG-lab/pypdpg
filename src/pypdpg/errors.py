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
    "E-COMPARE": (
        "comparing requires reading the value — the party running this code "
        "holds no secret key. That's the point.",
        "return the encrypted result to the data owner for the decision, "
        "or soft-threshold with pdpg.approx.sigmoid(x).",
    ),
    "E-COERCE": (
        "numpy tried to materialize the plaintext; that requires the secret "
        "key on the data owner's side.",
        "keep computing on the CipherArray (supported ops below), or "
        ".decrypt() where the secret key lives.",
    ),
    "E-DEPTH": (
        "each ciphertext multiplication consumes one of the 4 rescaling "
        "levels this context provides, and this chain used them all.",
        "chain fewer multiplications, or return the intermediate result to "
        "the data owner for re-encryption (a fresh ciphertext starts at "
        "full depth).",
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

# numpy op name -> catalog code. Anything unlisted falls back to
# E-UNSUPPORTED.
_COMPARE_OPS = (
    "greater greater_equal less less_equal equal not_equal maximum minimum "
    "fmax fmin absolute fabs sign sort argsort max min amax amin argmax "
    "argmin clip isnan isinf isfinite any all nonzero where"
)
_TRANSCEND_OPS = (
    "exp exp2 expm1 log log2 log10 log1p sqrt cbrt tanh sinh cosh sin cos "
    "tan arcsin arccos arctan arctan2 arcsinh arccosh arctanh power float_power"
)
_DIV_OPS = "true_divide divide floor_divide remainder mod fmod reciprocal"

_OP_TO_CODE = {
    **{op: "E-COMPARE" for op in _COMPARE_OPS.split()},
    **{op: "E-TRANSCEND" for op in _TRANSCEND_OPS.split()},
    **{op: "E-DIV" for op in _DIV_OPS.split()},
}

# symbol shown next to the op name when there is an operator spelling
_OP_SYMBOL = {
    "true_divide": "/",
    "divide": "/",
    "floor_divide": "//",
    "remainder": "%",
    "mod": "%",
    "power": "**",
    "greater": ">",
    "greater_equal": ">=",
    "less": "<",
    "less_equal": "<=",
    "equal": "==",
    "not_equal": "!=",
    "absolute": "abs",
    "matmul": "@",
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
