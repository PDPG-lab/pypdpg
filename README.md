# pypdpg

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/demo.ipynb)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: PDPG Community](https://img.shields.io/badge/license-PDPG%20Community-blueviolet)

**Your numpy pipeline. On ciphertext. Unchanged.**

Someone hands you `data.enc`. You `np.load` it, run the same scoring code you
wrote years ago, and send back `result.enc` — without ever being *able* to
read a row of what you just processed. Fully homomorphic encryption
([TenSEAL](https://github.com/OpenMined/TenSEAL)/CKKS) wearing a numpy skin,
by [PDPG-lab](https://pdpglab.xyz).

```python
import pypdpg as pdpg

# ---- Org A: owns the data, keeps the key ----
ctx = pdpg.Context.create()
ctx.save("orga.key")
ctx.save_public("vendor.ctx")
pdpg.encrypt(X, ctx).save("data.enc")

# ---- Vendor: two new lines, zero new concepts ----
pdpg.activate("vendor.ctx")
pdpg.install()                    # np.load now speaks .enc
X = np.load("data.enc")           # <CipherArray shape=(200, 5) 🙈 CKKS · secret_key=absent>
scores = X @ w + b                # your model, computing blind
scores.save("result.enc")

# ---- Org A: only holder of the key ----
pdpg.activate("orga.key")
result = pdpg.load("result.enc").decrypt()
```

## Errors that teach

Try to peek, and the library explains the cryptography to you:

```
>>> X > 600
EncryptedOperationError: np.greater (>) is impossible on CKKS ciphertexts.
Why: comparing requires reading the value — the party running this code
holds no secret key. That's the point.
Do instead: return the encrypted result to the data owner for the decision,
or soft-threshold with pdpg.approx.sigmoid(x).
Supported here: + - * @ dot sum mean square / scalar, pdpg.approx.sigmoid
```

Every impossible operation answers like this — comparisons, `np.exp`, division
by ciphertext, `np.asarray`, row indexing, `bool()`. Never a bare stack trace.

## What drops in, what needs a rewrite, what's coming

| | |
|---|---|
| ✅ **drop in now** | existing numpy code runs unchanged: `+ - * /scalar` · `@` · `dot` · `sum` · `mean` · `square` · `**n` · `pdpg.approx.sigmoid` · column select `X[:, j]` · save/load · `np.load` |
| 🔁 **needs a rewrite** | data-dependent logic runs today, written branchless — the same discipline as constant-time crypto code: `if/else` → `gate*b + (1-gate)*c` · thresholds → `sigmoid` gates · row filtering → full-shape masking (a filtered row *count* would leak) · `/ cipher` → multiply by a reciprocal |
| 🔜 **waiting on the engine** | exact comparisons and `max`/`sort` (programmable bootstrapping / sign circuits) · ciphertext division · high-precision `exp`/`log`/`sqrt` · unlimited depth via bootstrapping · encrypted@encrypted matmul · GPU acceleration. When the engine lands them, they drop in — your code doesn't change |

And one thing that fits no bucket, ever: *this* party reading the data.
That's not a roadmap item; that's the security guarantee.

## Encrypted if/else

Branching on ciphertext is impossible — a plaintext `bool` is exactly what
the vendor must never have. *Selecting* on ciphertext is just arithmetic:

```python
# if x > 0: b else c   —   gate and both branches stay encrypted
gate = pdpg.approx.sigmoid(x)
result = gate * b + (1 - gate) * c
```

Runs today, three of four depth levels, every row lands on the right branch.
The full rewrite rules live in [docs/design.md](docs/design.md).

## Try it

One click: the [Colab demo](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/demo.ipynb)
plays both parties — encrypt, score blind, decrypt, then a whole act of
trying (and failing) to leak data. Locally:

```
pip install git+https://github.com/PDPG-lab/pypdpg
```

## vs the field

- **[Zama Concrete](https://github.com/zama-ai/concrete)** compiles a fixed
  circuit ahead of time; we dispatch at runtime. Zero compilation, true drop-in.
- **[CuPy](https://cupy.dev/) / [Dask](https://www.dask.org/)** — we're the
  encrypted member of the duck-array family.
- **[TenSEAL](https://github.com/OpenMined/TenSEAL)** is our engine; we're the
  numpy dispatch, key custody model, and file format on top.

## Fine print (really, do read it)

CKKS arithmetic is approximate (~1e-4), multiplication depth is finite (4
chained), arrays cap at 8192 rows, and ciphertext is ~1 MB per column.
Encrypted is **pseudonymized, not anonymized** — GDPR still applies. The
measurements, caveats, and legal framing live in
[docs/fine-print.md](docs/fine-print.md); the architecture (column packing,
dispatch, `.enc` format, custody) in [docs/design.md](docs/design.md).

## License

Free for individuals, education, research, non-profits, government and
public-sector bodies, and organizations under **THB 50M** (Thai-registered) /
**USD 1M** (elsewhere) annual revenue. Bigger than that? You're the reason we
can afford the free part — [pdpglab.xyz](https://pdpglab.xyz). Full terms:
[LICENSE.md](LICENSE.md) (source-available, not OSI open source, and we don't
claim otherwise).
