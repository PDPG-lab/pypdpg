# pypdpg

Drop-in encrypted numpy. Encrypt your data, ship it to a vendor, and their
existing numpy pipeline runs unchanged — on ciphertext. Encrypted in,
encrypted out, only you can decrypt.

Built on [TenSEAL](https://github.com/OpenMined/TenSEAL) (CKKS).

> Work in progress — full README, demo notebook, and docs landing soon.

```python
import pypdpg as pdpg

# data owner
ctx = pdpg.Context.create()
ctx.save("orga.key")
ctx.save_public("vendor.ctx")
pdpg.encrypt(X, ctx).save("data.enc")

# vendor — existing numpy code, unchanged
pdpg.activate("vendor.ctx")
X = pdpg.load("data.enc")
scores = X @ w + b          # computed on ciphertext
scores.save("result.enc")

# data owner
pdpg.activate("orga.key")
result = pdpg.load("result.enc").decrypt()
```
