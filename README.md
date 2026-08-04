# pypdpg

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/demo.ipynb)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: PDPG Community](https://img.shields.io/badge/license-PDPG%20Community-blueviolet)

Run numpy code on homomorphically encrypted arrays. The data owner encrypts and keeps
the secret key; another party computes on the ciphertext using ordinary numpy
operations; the owner decrypts the result. A
[PDPG-lab](https://pdpglab.xyz) project.

## Installation

```
pip install git+https://github.com/PDPG-lab/pypdpg
```

Python 3.10+. All dependencies install as prebuilt wheels on Linux x86_64,
macOS arm64, and Windows.

## Usage

```python
import pypdpg as pdpg

# data owner: create keys, encrypt, ship
ctx = pdpg.Context.create()
ctx.save("orga.key")                 # includes secret key, stays with the owner
ctx.save_public("vendor.ctx")        # evaluation keys only
pdpg.encrypt(X, ctx).save("data.enc")

# computing party: no secret key
pdpg.activate("vendor.ctx")
pdpg.install()                       # np.load now recognizes .enc files
X = np.load("data.enc")              # CipherArray, shape (200, 5)
scores = X @ w + b                   # unchanged numpy code
scores.save("result.enc")

# data owner: decrypt the result
pdpg.activate("orga.key")
result = pdpg.load("result.enc").decrypt()
```

`pdpg.install()` patches `np.load` with a magic-byte check; regular files
load exactly as before, `pdpg.uninstall()` restores the original.
`decrypt()` requires a context that holds the secret key.

## Supported operations

| | |
|---|---|
| Works unchanged | `+ - * /scalar` · `@` · `dot` · `sum` · `mean` · `square` · `**n` · `pdpg.approx.sigmoid` · column select `X[:, j]` · save/load · `np.load` |
| Requires rewriting | data-dependent logic, expressed branchless: `if/else` as `gate*b + (1-gate)*c` · thresholds via `sigmoid` gates · row filtering via full-shape masking · division by ciphertext via plain reciprocals |
| Planned (engine-side) | exact comparisons, `max`/`sort` · ciphertext division · `exp`/`log`/`sqrt` · unlimited depth via bootstrapping · encrypted×encrypted matmul · GPU |

Operations that would reveal plaintext to the computing party — comparisons
returning readable booleans, `bool()`, `np.asarray`, decryption without the
key — are excluded by the security model rather than by the roadmap.

## Error messages

Unsupported operations raise `EncryptedOperationError` with an explanation
and an alternative, instead of a backend stack trace:

```
>>> X > 600
EncryptedOperationError: np.greater (>) is impossible on CKKS ciphertexts.
Why: comparing requires reading the value — the party running this code
holds no secret key. That's the point.
Do instead: return the encrypted result to the data owner for the decision,
or soft-threshold with pdpg.approx.sigmoid(x).
Supported here: + - * @ dot sum mean square / scalar, pdpg.approx.sigmoid
```

## Conditional logic

Branching on encrypted values is not possible (it would require a plaintext
boolean). Selection is expressed arithmetically:

```python
gate = pdpg.approx.sigmoid(x)          # encrypted soft indicator
result = gate * b + (1 - gate) * c     # if x > 0: b else c
```

Both branches are always evaluated, loops need fixed bounds, and result
shapes cannot depend on data. See [docs/design.md](docs/design.md) for the
rewriting rules.

## Demo

[demo/demo.ipynb](demo/demo.ipynb) walks through the full two-party flow —
encryption, blind scoring, decryption, and a section of attempted leaks —
and runs top-to-bottom on a fresh Colab runtime.

## Documentation

- [docs/design.md](docs/design.md) — column packing, dispatch, file formats,
  key handling
- [docs/fine-print.md](docs/fine-print.md) — accuracy and size measurements,
  depth budget, threat model, legal notes

## Related projects

- [Zama Concrete](https://github.com/zama-ai/concrete) — ahead-of-time
  circuit compilation; pypdpg dispatches at runtime instead.
- [CuPy](https://cupy.dev/) / [Dask](https://www.dask.org/) — duck-typed
  array libraries; pypdpg implements the same numpy dispatch protocols.
- [TenSEAL](https://github.com/OpenMined/TenSEAL) — the CKKS implementation
  pypdpg currently uses as its backend.

## Limitations

- CKKS arithmetic is approximate (~1e-4 error on the demo workload).
- Multiplicative depth is 4; longer chains raise a depth error.
- Arrays are 1-D/2-D, up to 8192 rows.
- Ciphertext is ~1 MB per column; the evaluation context is ~180 MB.
- Encrypted data remains personal data under GDPR (pseudonymization, not
  anonymization).

Details in [docs/fine-print.md](docs/fine-print.md).

## License

[PDPG Community License](LICENSE.md) (source-available). Free for
individuals, education, research, non-profits, the public sector, and
organizations under THB 50M (Thai-registered) / USD 1M (elsewhere) annual
revenue. Larger organizations require an enterprise license — contact
[pdpglab.xyz](https://pdpglab.xyz).

---

Maintained by [PDPG-lab](https://pdpglab.xyz).
