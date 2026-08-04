# Fine print

The README sells the idea; this page tells you where the edges are. Both are
true.

## What this is not

Encrypted-with-a-key data is **pseudonymized, not anonymized** — GDPR still
applies to it, and so does Thailand's PDPA, which follows the same data
controller / data processor model. pypdpg is a technical measure in the sense
of GDPR Art. 32 (and the PDPA's security-measure duties), a data minimization
tool, and a supplementary measure for third-country transfers in the sense of
*Schrems II*. We do not claim it makes data non-personal, and neither should
you.

This is a research prototype, not certified production cryptography. The
underlying scheme (CKKS via TenSEAL/Microsoft SEAL) is serious; our packaging
of it has had no independent security review.

## Honest limits, measured

- **Approximate arithmetic.** CKKS returns ~1e-4-accurate results, not exact
  ones. In the demo (linear scoring of 200 applicants), max absolute error is
  ~2.5e-4 on ~600-point scores. Fine for scoring and analytics; we print the
  error rather than hide it.
- **Multiplicative depth 4.** Each ciphertext multiplication in a chain spends
  one level — and multiplying by *plaintext* costs a level too (negation and
  addition are free). The fifth chained multiply raises a teaching error
  (`E-DEPTH`) suggesting you restructure or return the intermediate to the
  data owner for re-encryption. Bootstrapping (unlimited depth) is an
  engine-side upgrade we inherit when TenSEAL ships it.
- **Max 8192 rows** per array: one SIMD slot per row at polynomial degree
  16384. Larger data means chunking (not built in yet).
- **Size.** A ciphertext column is ~1 MB regardless of how many of its slots
  you use: ~16× overhead with all 8192 slots full, hundreds of × for small
  demos (the 200×5 demo array is ~5 MB against 8 kB of plaintext). The
  one-time `processor.ctx` evaluation-key file is ~180 MB — galois keys are
  large at this degree.
- **`pdpg.approx.sigmoid` is a degree-3 polynomial**, accurate to ~1e-1 on
  inputs in roughly [-5, 5] and *divergent* outside that range. Scale your
  inputs; the soft gate has a gray zone near the threshold.
- **Exact zeros are refused.** Subtracting a ciphertext from itself (the same
  object) would produce a "transparent" ciphertext that trivially reveals its
  plaintext, so SEAL raises. Two independent encryptions of equal values
  subtract fine.
- **Scope.** 1-D and 2-D float arrays, CKKS only, one key pair per pipeline.
  No performance tuning beyond "the demo runs in seconds."

## Threat model, briefly

The data processor is honest-but-curious: it runs the agreed computation but
might try to read the data. Against that, ciphertext plus a public evaluation
context reveals nothing usable. An actively malicious processor could return
wrong results (FHE alone doesn't give verifiability) or refuse to compute;
result *shape* and file sizes are visible by design. Key custody is the whole
game: `controller.key` never travels.

Note the asymmetry: FHE protects the controller's **data** from the
processor; it does not protect the processor's **model** from the
controller. The controller decrypts the scores, and decrypted scores permit
standard model extraction — a linear model can be recovered exactly from
d+1 probing queries. This is the same exposure any scoring API has, and it
is a contract matter, not a cryptographic one.
