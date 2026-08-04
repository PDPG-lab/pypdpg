# pypdpg

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/demo.ipynb)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: PDPG Community](https://img.shields.io/badge/license-PDPG%20Community-blueviolet)

**Drop-in encrypted numpy.** Org A encrypts its data with fully homomorphic
encryption and ships it to a vendor. The vendor pip-installs this library, adds
two lines, and their existing numpy pipeline runs unchanged — on ciphertext.
Encrypted in, encrypted out, and only Org A holds the key that can read the
result. Built on [TenSEAL](https://github.com/OpenMined/TenSEAL) (CKKS).

## Quickstart

```python
import pypdpg as pdpg

# ---- Org A (data owner) ----
ctx = pdpg.Context.create()
ctx.save("orga.key")                   # secret key — never leaves Org A
ctx.save_public("vendor.ctx")          # evaluation keys — safe to ship
pdpg.encrypt(X, ctx).save("data.enc")

# ---- Vendor (no secret key) ----
pdpg.activate("vendor.ctx")
pdpg.install()                         # np.load now understands .enc
X = np.load("data.enc")                # -> CipherArray 🙈
scores = X @ w + b                     # unmodified numpy code, computed blind
scores.save("result.enc")

# ---- Org A ----
pdpg.activate("orga.key")
result = pdpg.load("result.enc").decrypt()
```

## What works, what can't, what's coming

| | |
|---|---|
| ✅ **works today** | `+ - * /scalar` · `@` · `dot` · `sum` · `mean` · `square` · `**n` · `pdpg.approx.sigmoid` · column select `X[:, j]` · save/load · `np.load` drop-in |
| ❌ **impossible without interaction** | comparisons · `sort`/`argmax` · `exp`/`log`/`sqrt` · division by ciphertext · row indexing · `bool` — anything that must *read* a value. Each raises a teaching error explaining why, and what to do instead |
| 🔜 **engine-side, code-side nothing** | bootstrapping (unlimited depth) · encrypted@encrypted matmul · GPU acceleration · smaller ciphertexts |

The engine improves; your code doesn't.

## What this is not

Encrypted-with-a-key data is **pseudonymized, not anonymized** — GDPR still
applies to it. pypdpg is a technical measure in the sense of Art. 32, a data
minimization tool, and a supplementary measure for third-country transfers in
the sense of *Schrems II*. We do not claim it makes data non-personal, and
neither should you.

## Prior art

- **[TenSEAL](https://github.com/OpenMined/TenSEAL)** — our engine; we add the
  numpy-native array model, dispatch, key custody, and file format on top.
- **[Zama Concrete](https://github.com/zama-ai/concrete)** — compiles a fixed
  circuit ahead of time; we dispatch at runtime. Zero compilation, true drop-in.
- **[CuPy](https://cupy.dev/) / [Dask](https://www.dask.org/)** — we're the
  encrypted member of the duck-array family: same numpy code, different
  execution substrate.

## Honest limits

- **Multiplicative depth 4.** Each ciphertext multiplication in a chain spends
  one level; the fifth raises a teaching error suggesting re-encryption.
- **Max 8192 rows** per array (one SIMD slot per row at degree 16384).
- **Size.** A ciphertext column is ~1 MB regardless of row count: ~16× overhead
  with full slots, hundreds of times for small demos. The one-time `vendor.ctx`
  (evaluation keys) is ~180 MB.
- **Approximate arithmetic.** CKKS returns ~1e-4-accurate results, not exact
  ones. Fine for scoring and analytics; we print the error rather than hide it.
- Hackathon MVP: 1-D/2-D arrays, CKKS only, no security hardening or
  performance tuning. Not production crypto.

## License

**Source-available under the [PDPG Community License](LICENSE.md).** We build
this so that privacy-preserving computation becomes ordinary — which means it
must be free for the people protecting the public. It is free for individuals,
education, research, non-profits, and any organization under **THB 500M /
USD 15M** annual revenue. Above that line, you're the reason we can afford to
keep it free: [get in touch](https://github.com/PDPG-lab) for an enterprise
license. (Not an OSI-approved open-source license, and we don't claim it is.)
