"""Teaching errors: every refused operation explains itself.

Two kinds of refusal, and the message says which:

- ``backend`` — the *current* backend cannot compute this; a future backend
  can, with the result still encrypted. These carry a "Backend roadmap:"
  line.
- ``design`` — impossible on encrypted arrays under any backend, because it
  would hand this party a plaintext. That's the security model, not a gap.
"""

_SUPPORTED = (
    "Supported here: + - * @ dot sum mean square / scalar, pdpg.approx.sigmoid"
)

# Single source of truth for the active backend name in error messages.
# Becomes dynamic when a second backend exists.
CURRENT_BACKEND = "CKKS"


class EncryptedOperationError(TypeError):
    """Raised when an operation is impossible (or refused) on ciphertext."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


_CATALOG = {
    "E-COMPARE": {
        "kind": "backend",
        "why": "the current backend cannot evaluate comparisons at all — "
        "and even on a comparison-capable backend the result stays "
        "encrypted; reading the verdict needs the key. That's the point.",
        "do": "return the encrypted result to the data owner for the "
        "decision, or soft-threshold with pdpg.approx.sigmoid(x).",
        "roadmap": "TFHE-class backends compute comparisons with encrypted "
        "results; workload code won't change.",
    },
    "E-DIV": {
        "kind": "backend",
        "why": "CKKS has no ciphertext division — computing a reciprocal "
        "would require reading the value.",
        "do": "multiply by a plain reciprocal (x * (1/c)), or return the "
        "denominator to the data owner to decrypt.",
        "roadmap": "backends with iterative reciprocal circuits can divide; "
        "workload code won't change.",
    },
    "E-TRANSCEND": {
        "kind": "backend",
        "why": "only polynomial functions (adds and multiplies) exist under "
        "the current backend — transcendental functions have no circuit "
        "here.",
        "do": "use pdpg.approx.sigmoid(x), or fit your own low-degree "
        "polynomial to the function over your input range.",
        "roadmap": "deeper polynomial budgets and programmable bootstrapping "
        "approximate these; workload code won't change.",
    },
    "E-DEPTH": {
        "kind": "backend",
        "why": "each ciphertext multiplication consumes one of the 4 "
        "rescaling levels this context provides, and this chain used them "
        "all.",
        "do": "chain fewer multiplications, or return the intermediate "
        "result to the data owner for re-encryption (a fresh ciphertext "
        "starts at full depth).",
        "roadmap": "a bootstrapping-capable backend removes the depth limit "
        "entirely; workload code won't change.",
    },
    "E-UNSUPPORTED": {
        "kind": "backend",
        "why": "the current backend evaluates additions and multiplications "
        "only; this operation has no homomorphic form here.",
        "do": "return the encrypted result to the data owner, or "
        "approximate with a polynomial (see pdpg.approx).",
        "roadmap": "backend coverage grows release by release; workload "
        "code won't change when it does.",
    },
    "E-COERCE": {
        "kind": "design",
        "why": "this needs the plaintext value in this process, and that "
        "value exists only where the secret key lives.",
        "do": "keep computing on the CipherArray (supported ops below), or "
        ".decrypt() on the data owner's side.",
    },
    "E-CUSTODY": {
        "kind": "design",
        "why": "decryption requires the secret key, and this context holds "
        "only public evaluation keys. That's the point.",
        "do": "save the encrypted result and return it to the data owner — "
        'only their context (e.g. pdpg.activate("controller.key")) can '
        "decrypt.",
    },
    "E-INDEX": {
        "kind": "design",
        "why": "row selection would have to move or mask individual slots "
        "inside a packed ciphertext, and boolean masks would need readable "
        "comparisons.",
        "do": "select whole columns instead — X[:, j] or X[:, [i, k]] — or "
        "return the array to the data owner for row-level work.",
    },
    "E-ORDER": {
        "kind": "design",
        "why": "with the plaintext on the left, numpy leads the "
        "computation, and pypdpg's column packing cannot serve the "
        "transposed product.",
        "do": "reorder so the encrypted array comes first: X @ w instead "
        "of w @ X.",
    },
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
    entry = _CATALOG[code]
    lines = [headline, f"Why: {entry['why']}", f"Do instead: {entry['do']}"]
    if "roadmap" in entry:
        lines.append(f"Backend roadmap: {entry['roadmap']}")
    lines.append(_SUPPORTED)
    return EncryptedOperationError(code, "\n".join(lines))


def for_numpy(name: str) -> EncryptedOperationError:
    """Teaching error for a numpy ufunc/function hitting ciphertext."""
    code = _OP_TO_CODE.get(name, "E-UNSUPPORTED")
    symbol = f" ({_OP_SYMBOL[name]})" if name in _OP_SYMBOL else ""
    if _CATALOG[code]["kind"] == "backend":
        headline = (
            f"np.{name}{symbol} is not supported by the current backend "
            f"({CURRENT_BACKEND})."
        )
    else:
        headline = f"np.{name}{symbol} is impossible on encrypted arrays — by design."
    return make(code, headline)
