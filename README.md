# pypdpg

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/getting_started.ipynb)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: PDPG Community](https://img.shields.io/badge/license-PDPG%20Community-blueviolet)

🇹🇭 [อ่านภาษาไทย](README.th.md)

Run Python data workloads on homomorphically encrypted arrays. The data
controller encrypts and keeps the secret key; the data processor computes on
the ciphertext with its ordinary code; the controller decrypts the result.

pypdpg is building up encrypted execution for the common Python data stack,
library by library — numpy at the core, with pandas and scikit-learn layers
on top. Coverage is partial and growing; within each library, the supported
subset is what CKKS can honestly do, and everything else refuses with an
explanation. A [PDPG-lab](https://pdpglab.xyz) project.

## Library coverage

| library | status | what runs on ciphertext |
|---|---|---|
| **numpy** | core | arithmetic, broadcasting, `@`/`dot`, reductions, `np.load` drop-in — via the array dispatch protocols |
| **pandas** | facade | named columns, computed columns, labeled aggregates, `.enc` roundtrip (`CipherFrame`) |
| **scikit-learn** | inference wrappers | fitted linear models, logistic probabilities, scalers, `KMeans` distances, pipelines (`pdpg.sklearn.wrap`) |
| **shell** | CLI | `pdpg keygen / encrypt / inspect / decrypt` |
| statsmodels, PyTorch (linear + square-activation nets), … | planned | — |

## Installation

```
pip install git+https://github.com/PDPG-lab/pypdpg
```

Python 3.10+. All dependencies install as prebuilt wheels on Linux x86_64,
macOS arm64, and Windows.

## Usage

```python
import pypdpg as pdpg

# data controller: create keys, encrypt, ship
ctx = pdpg.Context.create()
ctx.save("controller.key")           # includes secret key, stays with the controller
ctx.save_public("processor.ctx")     # evaluation keys only
pdpg.encrypt(X, ctx).save("data.enc")

# data processor: no secret key
pdpg.activate("processor.ctx")
pdpg.install()                       # np.load now recognizes .enc files
X = np.load("data.enc")              # CipherArray, shape (200, 5)
scores = X @ w + b                   # unchanged numpy code
scores.save("result.enc")

# data controller: decrypt the result
pdpg.activate("controller.key")
result = pdpg.load("result.enc").decrypt()
```

`pdpg.install()` patches `np.load` with a magic-byte check; regular files
load exactly as before, `pdpg.uninstall()` restores the original.
`decrypt()` requires a context that holds the secret key.

## Supported operations

| | |
|---|---|
| Works unchanged | `+ - * /scalar` · `@` · `dot` · `sum` · `mean` · `square` · `**n` · `pdpg.approx.sigmoid` · column select `X[:, j]` · save/load · `np.load` · fitted scikit-learn linear models via `pdpg.sklearn.wrap` · DataFrame-style named columns (`CipherFrame`) · `pdpg` CLI |
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

## scikit-learn models

Fitted linear models run on encrypted arrays through `pdpg.sklearn.wrap`,
which reads the learned parameters and replays them as encrypted arithmetic.
scikit-learn is not a dependency of pypdpg and never touches the ciphertext.

The typical setup: the processor is a score provider that owns the model and
fitted it on its own historical data; the controller sends only encrypted
user records. The model stays in the clear (it's the provider's), the data
never is:

```python
# score provider (data processor): your model, fitted on your own data
pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_hist, y_hist)

# same provider, scoring a client's ciphertext: one extra line
model = pdpg.sklearn.wrap(pipe)
proba = model.predict_proba(X_enc)      # encrypted P(class 1), back to the client
```

Supported: linear regressors including multi-output (`LinearRegression`,
`Ridge`, `Lasso`, …), binary linear classifiers (`decision_function`,
`predict_proba` via the sigmoid approximation), `StandardScaler`, `KMeans`
(`transform_squared` — encrypted squared distances to every centroid; same
ranking as `transform`, square root after decryption), and `Pipeline`s of
those. A scaler + logistic pipeline uses exactly the full multiplicative
depth budget. `predict()` raises a teaching error in every wrapper: hard
labels and nearest-centroid picks are decisions, and decisions belong to
the key holder.

## pandas-style frames

`pdpg.encrypt` accepts DataFrames and returns a `CipherFrame`: named columns
over ciphertext. Column names travel inside the `.enc` file, so the
processor gets them back from `np.load`; `decrypt()` returns real labeled
pandas objects. (A thin facade, not a real DataFrame — pandas is imported
only at decrypt time and is not a dependency.)

```python
enc = pdpg.encrypt(df, ctx)              # CipherFrame [income, debt, age]
enc["debt_ratio_k"] = enc["debt"] * 0.001    # computed column, encrypted
enc[["income", "debt"]].mean()           # encrypted, labels preserved
# ... controller side:
enc.mean().decrypt()                     # pandas Series, index = column names
```

## Command line

The two-party flow without writing Python:

```
pdpg keygen -o keys/
pdpg encrypt applicants.csv -c keys/controller.key -o data.enc
pdpg inspect data.enc          # shape, columns, fingerprint — never values
pdpg decrypt result.enc -c keys/controller.key -o result.csv
```

`inspect` needs no key and shows exactly what an outsider can learn from the
file: the shape, the column names, the size. Decrypting with the public
context prints the custody teaching error and exits nonzero.

## Notebooks

Each runs standalone, top-to-bottom, on a fresh Colab runtime.

| notebook | shows |
|---|---|
| [getting_started](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/getting_started.ipynb) | the two-party flow end to end: encrypt a DataFrame, score blind, decrypt |
| [01 · sklearn models](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/cookbook/01_sklearn_models.ipynb) | wrapped pipelines, encrypted probabilities, KMeans segmentation |
| [02 · encrypted dataframes](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/cookbook/02_encrypted_dataframes.ipynb) | named columns, computed columns, labeled aggregates |
| [03 · branchless logic](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/cookbook/03_branchless_logic.ipynb) | the refusals tour, then the constant-time rewrite patterns |
| [04 · CLI workflow](https://colab.research.google.com/github/PDPG-lab/pypdpg/blob/main/demo/cookbook/04_cli_workflow.ipynb) | keygen / encrypt / inspect / decrypt as shell commands |

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
- Encrypted data remains personal data under GDPR and Thailand's PDPA
  (pseudonymization, not anonymization).

Details in [docs/fine-print.md](docs/fine-print.md).

## License

[PDPG Community License](LICENSE.md) (source-available). Free for
individuals, education, research, non-profits, the public sector, and
organizations under THB 50M (Thai-registered) / USD 1M (elsewhere) annual
revenue. Larger organizations require an enterprise license — contact
[pdpglab.xyz](https://pdpglab.xyz).

---

Maintained by [PDPG-lab](https://pdpglab.xyz).
