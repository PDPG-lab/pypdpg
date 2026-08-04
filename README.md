# pypdpg

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/demo.ipynb)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: PDPG Community](https://img.shields.io/badge/license-PDPG%20Community-blueviolet)

**Drop-in encrypted numpy.** Org A encrypts its data with fully homomorphic
encryption and ships it to a vendor. The vendor pip-installs this library, adds
two lines, and their existing numpy pipeline runs unchanged — on ciphertext.
Encrypted in, encrypted out, and only Org A holds the key that can read the
result. Built on [TenSEAL](https://github.com/OpenMined/TenSEAL) (CKKS) by
[PDPG-lab](https://pdpglab.xyz).

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

## What drops in, what needs a rewrite, what's coming

| | |
|---|---|
| ✅ **drop in now** | existing numpy code runs unchanged: `+ - * /scalar` · `@` · `dot` · `sum` · `mean` · `square` · `**n` · `pdpg.approx.sigmoid` · column select `X[:, j]` · save/load · `np.load` |
| 🔁 **needs a rewrite** | data-dependent logic runs today, written branchless — the same discipline as constant-time crypto code: `if/else` → `gate*b + (1-gate)*c` · thresholds → `sigmoid` gates · row filtering → full-shape masking (a filtered row *count* would leak) · `/ cipher` → multiply by a reciprocal |
| 🔜 **waiting on the engine** | exact comparisons and `max`/`sort` (programmable bootstrapping / sign circuits) · ciphertext division · high-precision `exp`/`log`/`sqrt` · unlimited depth via bootstrapping · encrypted@encrypted matmul · GPU acceleration. When the engine lands them, they drop in — your code doesn't change |

And one thing that fits no bucket, ever: *this* party reading the data —
`bool()`, `np.asarray`, peeking at values, decrypting without the key. That's
not a roadmap item; that's the security guarantee. Every refused operation
raises a teaching error pointing you at the right bucket.

### Control flow: rewrite, don't branch

Python's `if` needs a plaintext boolean, so *branching* on encrypted data is
impossible by design. *Selecting* on encrypted data is just arithmetic —
rewrite branches the way constant-time crypto code does: evaluate both sides,
blend with an encrypted gate.

```python
# if x > 0: b else c   —   branchless; gate and both branches stay encrypted
gate = pdpg.approx.sigmoid(x)         # soft indicator in [0, 1], ciphertext
result = gate * b + (1 - gate) * c    # selection by multiplication
```

This runs today (three of the four depth levels, ~1e-4 error, every row lands
on the correct branch). The rules are the ones constant-time programmers
already know: both branches always execute, loops need fixed worst-case
bounds, and result shapes can't depend on data — a filtered row count would
leak. Under CKKS the gate is soft, with a gray zone near the threshold; exact
0/1 gates arrive with the comparison circuits in the 🔜 row above, and this
pattern doesn't change when they do.

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
education, research, non-profits, government and public-sector bodies, and any
organization under **THB 50M** (Thai-registered) or **USD 1M** (elsewhere)
annual revenue. Above that line, you're the reason we can afford to keep it
free: [get in touch](https://pdpglab.xyz) for an enterprise
license. (Not an OSI-approved open-source license, and we don't claim it is.)
