# Design

How a numpy expression ends up as CKKS operations, and why the pieces look
the way they do.

## Backends

pypdpg's public surface — `CipherArray`/`CipherFrame`, the dispatch
protocols, `pdpg.sklearn.wrap`, the `.enc` container — is deliberately
scheme-neutral: nothing in it promises CKKS specifics, and the `.enc`
header records which scheme produced a file. The current (and only)
backend is TenSEAL's CKKS; its calls are inlined in `core.py` today rather
than hidden behind a formal backend interface. Extracting that interface —
roughly: encrypt/decrypt, add/mul (plain and cipher), rotate-sum, polyval,
serialize — is roadmap work that becomes worth doing the moment a second
backend (TFHE-class: exact comparisons, programmable bootstrapping) is
real. The intended selection point is context creation, so two parties
agree on a backend the same way they already agree on a key pair.

## Column packing

A `(N, d)` array is stored as `d` TenSEAL `ckks_vector`s, each packing one
*column* of length N into SIMD slots (`packing="slots"`). This one choice
makes the money operations cheap:

- `X @ w` (plain `w`, shape `(d,)`) is `Σ_j col_j * w[j]` — d plain
  multiplies and free additions, **depth 1**, producing one packed vector of
  all N results.
- `X @ W` (plain `(d, k)`) is k such folds, one per output column.
- Column selection `X[:, j]` / `X[:, [i, k]]` is free — it reuses the stored
  vectors without touching ciphertext.
- Reductions (`sum`/`mean` over axis 0) rotate-and-add within each vector via
  galois keys, yielding one single-slot vector per column
  (`packing="scalars"`). Both packings decrypt transparently.

Row-wise anything, by contrast, would mean moving individual slots — which is
why row indexing raises a teaching error instead.

## Dispatch: three layers

1. **`__array_ufunc__`** catches numpy ufuncs (`np.add`, `np.multiply`,
   `np.matmul`, `np.square`, `np.power`, …) — including when numpy is on the
   *left* of the expression, which is why `w * X` works.
2. **`__array_function__`** catches the non-ufunc API: `np.dot`, `np.sum`,
   `np.mean`, `np.sort`, …
3. **Bare-operator dunders** catch what numpy never sees. `enc > 0`,
   `enc == x`, `abs(enc)`, `bool(enc)`, `enc // 2` go straight to Python's
   protocol, so `CipherArray` defines those dunders to raise teaching errors.
   The nastiest case is `__eq__`: left undefined, Python would silently
   answer `X == X` with identity semantics — a *wrong answer* instead of a
   refusal.

Anything not routed raises an `EncryptedOperationError` from a catalog keyed
by failure class (`E-COMPARE`, `E-TRANSCEND`, `E-DIV`, `E-COERCE`, `E-INDEX`,
`E-ORDER`, `E-CUSTODY`, `E-DEPTH`), each with a *why* and a *do instead*.
`E-DEPTH` is a translation: TenSEAL's `scale out of bounds` becomes
"multiplicative depth (4) exhausted — chain fewer multiplications."

Leak-proofing details: `__array__` raises (numpy cannot silently materialize
plaintext), `__repr__` shows shape and key custody but never slot values,
augmented assignment (`x += 3`) works through the ordinary binary dunders and
stays encrypted.

## Branchless rewrites

Data-dependent *control flow* needs a plaintext boolean — never available
here. Data-dependent *selection* is arithmetic:

```python
gate = pdpg.approx.sigmoid(x)          # soft indicator, ciphertext, depth 2
result = gate * b + (1 - gate) * c     # select, one more level
```

The rules mirror constant-time crypto code: both branches always execute,
loops unroll to fixed worst-case bounds, and result shapes must not depend on
data (mask to full shape instead of filtering — a filtered row count leaks).
Exact 0/1 gates arrive with engine-side comparison circuits; the pattern is
unchanged when they do.

## Key custody and the context fingerprint

`Context.create()` builds the CKKS context (degree 16384,
`coeff_mod_bit_sizes=[60,40,40,40,40,60]`, scale 2^40, galois + relin keys —
~128-bit security, depth 4, 8192 slots). Two save paths:

- `ctx.save("controller.key")` — full context including the secret key. Never ships.
- `ctx.save_public("processor.ctx")` — public + evaluation keys only. Safe to
  ship; `sum()` and relinearization work, `decrypt()` raises `E-CUSTODY`.

Both files carry a **fingerprint**: `sha256(public context bytes)[:16]`,
computed once at creation and *carried* in the file header rather than
recomputed on load. (Recomputing would break: SEAL stores fresh keys in
seeded, compressed form and expands them on deserialization, so the same
context does not re-serialize to the same bytes on the other side.) Every
`.enc` file embeds the fingerprint, and every load verifies it — using the
wrong key pair fails at open time with a clear message, not at decrypt time
with garbage.

## File formats

```
context file:  b"CCTX" | u8 version | u32 header_len | JSON header | tenseal context blob
               header: {"fp": <16 hex>, "private": bool}

array file:    b"CENC" | u8 version | u32 header_len | JSON header
               | per vector: u32 blob_len | tenseal ckks_vector blob
               header: {"shape": [N, d], "packing": "slots"|"scalars",
                        "scheme": "ckks", "ctx_fp": <16 hex>}
```

`pdpg.install()` wraps `np.load` with a 4-byte magic peek: `CENC` files come
back as `CipherArray`s (requiring an activated context), everything else
falls through to the real `np.load` untouched. `pdpg.uninstall()` restores
the original.

## Approximations

`pdpg.approx.sigmoid` evaluates `0.5 + 0.197x − 0.004x³` (the standard CKKS
logistic approximation, coefficients lowest-degree-first into TenSEAL's
`polyval`), valid on roughly [-5, 5]. It accepts plain arrays too, so the
same model code runs on both sides of the trust boundary.
