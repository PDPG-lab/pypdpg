"""Oracle tests: encrypt -> op -> decrypt -> compare against plain numpy."""

import numpy as np
import pytest

import pypdpg as pdpg
from pypdpg.errors import EncryptedOperationError

rng = np.random.default_rng(0)
X_PLAIN = rng.normal(size=(50, 4))
X1D_PLAIN = rng.uniform(-3, 3, size=50)
W_VEC = rng.normal(size=4)
W_MAT = rng.normal(size=(4, 2))
COL = rng.normal(size=(50, 1))
DIV_VEC = np.abs(rng.normal(size=4)) + 1.0  # safe denominators

ATOL = 1e-2


def oracle(ctx, plain, fn, atol=ATOL):
    """encrypt -> fn -> decrypt must match fn on plain numpy."""
    got = fn(pdpg.encrypt(plain, ctx)).decrypt()
    expected = fn(np.array(plain))
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=atol)


@pytest.fixture(scope="session")
def ctx():
    return pdpg.Context.create()


@pytest.fixture(scope="session")
def stranger_ctx():
    """A second, unrelated key pair."""
    return pdpg.Context.create()


@pytest.fixture(scope="session")
def key_files(ctx, tmp_path_factory):
    keys = tmp_path_factory.mktemp("keys")
    orga, vendor = keys / "orga.key", keys / "vendor.ctx"
    ctx.save(orga)
    ctx.save_public(vendor)
    return orga, vendor


@pytest.fixture(scope="session")
def pub_ctx(key_files):
    return pdpg.Context.load(key_files[1])


# ---------------------------------------------------------------- roundtrip

def test_roundtrip_2d(ctx):
    enc = pdpg.encrypt(X_PLAIN, ctx)
    assert enc.shape == X_PLAIN.shape
    assert np.allclose(enc.decrypt(), X_PLAIN, atol=ATOL)


def test_roundtrip_1d(ctx):
    enc = pdpg.encrypt(X1D_PLAIN, ctx)
    assert enc.shape == X1D_PLAIN.shape
    assert np.allclose(enc.decrypt(), X1D_PLAIN, atol=ATOL)


def test_context_save_load_roundtrip(ctx, tmp_path):
    path = tmp_path / "orga.key"
    ctx.save(path)
    loaded = pdpg.Context.load(path)
    assert loaded.has_secret
    assert loaded.fingerprint == ctx.fingerprint
    enc = pdpg.encrypt(X1D_PLAIN, loaded)
    assert np.allclose(enc.decrypt(), X1D_PLAIN, atol=ATOL)


def test_public_context_has_no_secret(ctx, pub_ctx):
    assert ctx.has_secret
    assert not pub_ctx.has_secret
    assert pub_ctx.fingerprint == ctx.fingerprint


def test_too_many_rows_rejected(ctx):
    with pytest.raises(ValueError, match="8192"):
        pdpg.encrypt(np.zeros(8193), ctx)


def test_3d_rejected(ctx):
    with pytest.raises(ValueError, match="ndim"):
        pdpg.encrypt(np.zeros((2, 2, 2)), ctx)


# --------------------------------------------------------------- arithmetic

def test_add_scalar(ctx):
    oracle(ctx, X_PLAIN, lambda a: a + 2)


def test_radd_scalar(ctx):
    oracle(ctx, X_PLAIN, lambda a: 2 + a)


def test_add_cipher_cipher(ctx):
    enc = pdpg.encrypt(X_PLAIN, ctx)
    assert np.allclose((enc + enc).decrypt(), X_PLAIN + X_PLAIN, atol=ATOL)


def test_sub_cipher_cipher(ctx):
    # two independent encryptions: same-object X - X would be an exactly-zero
    # "transparent" ciphertext, which SEAL refuses to produce
    a = pdpg.encrypt(X_PLAIN, ctx)
    b = pdpg.encrypt(X_PLAIN, ctx)
    assert np.allclose((a - b).decrypt(), np.zeros_like(X_PLAIN), atol=ATOL)


def test_rsub_scalar(ctx):
    oracle(ctx, X_PLAIN, lambda a: 5 - a)


def test_mul_scalar(ctx):
    oracle(ctx, X_PLAIN, lambda a: a * 3)


def test_mul_cipher_cipher(ctx):
    enc = pdpg.encrypt(X_PLAIN, ctx)
    assert np.allclose((enc * enc).decrypt(), X_PLAIN * X_PLAIN, atol=ATOL)


def test_mul_broadcast_row_vector(ctx):
    # (N, d) * (d,) — one plain weight per column
    oracle(ctx, X_PLAIN, lambda a: a * W_VEC)


def test_rmul_broadcast_row_vector(ctx):
    oracle(ctx, X_PLAIN, lambda a: W_VEC * a)


def test_mul_broadcast_column(ctx):
    # (N, d) * (N, 1) — one plain factor per row
    oracle(ctx, X_PLAIN, lambda a: a * COL)


def test_mul_same_shape_plain(ctx):
    oracle(ctx, X_PLAIN, lambda a: a * (X_PLAIN + 1.0))


def test_sub_broadcast_row_vector(ctx):
    oracle(ctx, X_PLAIN, lambda a: a - W_VEC)


def test_div_scalar(ctx):
    oracle(ctx, X_PLAIN, lambda a: a / 2)


def test_div_plain_vector(ctx):
    oracle(ctx, X_PLAIN, lambda a: a / DIV_VEC)


def test_neg(ctx):
    oracle(ctx, X_PLAIN, lambda a: -a)


def test_chained_expression(ctx):
    oracle(ctx, X1D_PLAIN, lambda a: (a + 10) * 5)


def test_augmented_assignment(ctx):
    enc = pdpg.encrypt(X1D_PLAIN, ctx)
    plain = X1D_PLAIN.copy()
    enc += 3
    plain += 3
    enc *= 2
    plain *= 2
    assert isinstance(enc, pdpg.CipherArray)  # stayed encrypted
    assert np.allclose(enc.decrypt(), plain, atol=ATOL)


def test_np_ufunc_spellings(ctx):
    enc = pdpg.encrypt(X1D_PLAIN, ctx)
    assert np.allclose(np.add(enc, 1).decrypt(), X1D_PLAIN + 1, atol=ATOL)
    assert np.allclose(
        np.multiply(X1D_PLAIN, enc).decrypt(), X1D_PLAIN * X1D_PLAIN, atol=ATOL
    )
    assert np.allclose(
        np.subtract(X1D_PLAIN, enc).decrypt(), np.zeros_like(X1D_PLAIN), atol=ATOL
    )
    assert np.allclose(np.negative(enc).decrypt(), -X1D_PLAIN, atol=ATOL)


def test_mismatched_context_rejected(ctx, stranger_ctx):
    # a second context has a different fingerprint; combining must fail
    a = pdpg.encrypt(X1D_PLAIN, ctx)
    b = pdpg.encrypt(X1D_PLAIN, stranger_ctx)
    with pytest.raises(ValueError, match="context"):
        a + b


# ------------------------------------------------------------ linalg & stats

def test_matmul_vector(ctx):
    oracle(ctx, X_PLAIN, lambda a: a @ W_VEC)


def test_matmul_matrix(ctx):
    oracle(ctx, X_PLAIN, lambda a: a @ W_MAT)


def test_np_dot(ctx):
    oracle(ctx, X_PLAIN, lambda a: np.dot(a, W_VEC))


def test_np_matmul(ctx):
    oracle(ctx, X_PLAIN, lambda a: np.matmul(a, W_MAT))


def test_dot_1d(ctx):
    oracle(ctx, X1D_PLAIN, lambda a: a @ X1D_PLAIN)


def test_sum_axis0(ctx):
    oracle(ctx, X_PLAIN, lambda a: a.sum(axis=0))


def test_np_sum_all(ctx):
    oracle(ctx, X_PLAIN, lambda a: np.sum(a))


def test_mean_axis0(ctx):
    oracle(ctx, X_PLAIN, lambda a: a.mean(axis=0))


def test_np_mean_axis0(ctx):
    oracle(ctx, X_PLAIN, lambda a: np.mean(a, axis=0))


def test_np_square(ctx):
    oracle(ctx, X_PLAIN, lambda a: np.square(a))


def test_pow_2(ctx):
    oracle(ctx, X1D_PLAIN, lambda a: a**2)


def test_pow_3(ctx):
    oracle(ctx, X1D_PLAIN, lambda a: a**3)


def test_sigmoid(ctx):
    enc = pdpg.approx.sigmoid(pdpg.encrypt(X1D_PLAIN, ctx))
    true = 1.0 / (1.0 + np.exp(-X1D_PLAIN))
    assert np.allclose(enc.decrypt(), true, atol=1e-1)


def test_slice_single_column(ctx):
    oracle(ctx, X_PLAIN, lambda a: a[:, 1])


def test_slice_column_list(ctx):
    oracle(ctx, X_PLAIN, lambda a: a[:, [0, 2]])


def test_sliced_column_still_computes(ctx):
    enc = pdpg.encrypt(X_PLAIN, ctx)
    got = (enc[:, 1] * 2 + 1).decrypt()
    assert np.allclose(got, X_PLAIN[:, 1] * 2 + 1, atol=ATOL)


# ---------------------------------------------------------- teaching errors

@pytest.fixture(scope="session")
def encX(ctx):
    return pdpg.encrypt(X_PLAIN, ctx)


IMPOSSIBLE_OPS = {
    "greater": lambda X: X > 0,
    "equal-must-raise-not-identity": lambda X: X == X,
    "exp": lambda X: np.exp(X),
    "div-by-cipher": lambda X: X / X,
    "rdiv-by-cipher": lambda X: 1.0 / X,
    "sqrt-via-pow": lambda X: X**0.5,
    "pow-zero": lambda X: X**0,
    "abs": lambda X: abs(X),
    "sort": lambda X: np.sort(X),
    "max": lambda X: np.max(X),
    "log": lambda X: np.log(X),
    "bool": lambda X: bool(X),
    "asarray": lambda X: np.asarray(X),
    "row-index": lambda X: X[0],
    "row-slice": lambda X: X[0:10],
    "bool-mask": lambda X: X[np.ones(50, dtype=bool)],
    "plain-matmul-left": lambda X: W_VEC @ X,
    "floordiv": lambda X: X // 2,
    "mod": lambda X: X % 2,
    "ufunc-reduce": lambda X: np.add.reduce(X),
}


@pytest.mark.parametrize("attempt", IMPOSSIBLE_OPS.values(), ids=IMPOSSIBLE_OPS.keys())
def test_impossible_ops_teach(encX, attempt):
    with pytest.raises(EncryptedOperationError) as excinfo:
        attempt(encX)
    message = str(excinfo.value)
    assert "Why:" in message
    assert "Do instead:" in message


def test_error_message_format(encX):
    with pytest.raises(EncryptedOperationError) as excinfo:
        encX > 0
    message = str(excinfo.value)
    assert message.startswith("np.greater (>) is impossible on CKKS ciphertexts.")
    assert "Supported here:" in message


def test_depth_exhausted_teaches(ctx):
    enc = pdpg.encrypt(np.full(4, 1.1), ctx)
    with pytest.raises(EncryptedOperationError, match="depth"):
        for _ in range(6):
            enc = enc * enc


# ------------------------------------------------------------------ custody

def test_decrypt_without_secret_key_raises(pub_ctx):
    enc = pdpg.encrypt(X1D_PLAIN, pub_ctx)
    with pytest.raises(EncryptedOperationError, match="Why:"):
        enc.decrypt()


def test_repr_never_shows_values(ctx, pub_ctx):
    enc = pdpg.encrypt(X_PLAIN, pub_ctx)
    r = repr(enc)
    assert "secret_key=absent" in r
    assert "(50, 4)" in r
    assert "secret_key=present" in repr(pdpg.encrypt(X_PLAIN, ctx))
    # no slot value should ever appear
    assert not any(f"{v:.3f}" in r for v in X_PLAIN[:, 0][:3])

# ----------------------------------------------------------------------- io

def test_enc_roundtrip(ctx, tmp_path):
    enc = pdpg.encrypt(X_PLAIN, ctx)
    enc.save(tmp_path / "data.enc")
    loaded = pdpg.load(tmp_path / "data.enc", ctx)
    assert loaded.shape == X_PLAIN.shape
    assert np.allclose(loaded.decrypt(), X_PLAIN, atol=ATOL)


def test_enc_roundtrip_reduced_packing(ctx, tmp_path):
    enc = pdpg.encrypt(X_PLAIN, ctx).mean(axis=0)
    enc.save(tmp_path / "mean.enc")
    loaded = pdpg.load(tmp_path / "mean.enc", ctx)
    assert np.allclose(loaded.decrypt(), X_PLAIN.mean(axis=0), atol=ATOL)


def test_load_wrong_context_fingerprint(ctx, stranger_ctx, tmp_path):
    pdpg.encrypt(X1D_PLAIN, ctx).save(tmp_path / "d.enc")
    with pytest.raises(ValueError, match="context"):
        pdpg.load(tmp_path / "d.enc", stranger_ctx)


def test_load_rejects_non_enc_file(ctx, tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, X_PLAIN)
    with pytest.raises(ValueError, match="magic"):
        pdpg.load(path, ctx)


def test_two_party_flow(ctx, key_files, tmp_path):
    orga_key, vendor_ctx = key_files
    # Org A encrypts and ships
    pdpg.encrypt(X_PLAIN, ctx).save(tmp_path / "data.enc")
    # Vendor: public context only, unmodified numpy scoring code
    pdpg.activate(vendor_ctx)
    X = pdpg.load(tmp_path / "data.enc")
    scores = X @ W_VEC + 0.7
    with pytest.raises(EncryptedOperationError, match="Why:"):
        scores.decrypt()
    scores.save(tmp_path / "result.enc")
    # Org A gets the result back
    pdpg.activate(orga_key)
    result = pdpg.load(tmp_path / "result.enc").decrypt()
    assert np.allclose(result, X_PLAIN @ W_VEC + 0.7, atol=ATOL)


def test_install_routes_np_load(ctx, key_files, tmp_path):
    enc_path = tmp_path / "data.enc"
    npy_path = tmp_path / "plain.npy"
    pdpg.encrypt(X_PLAIN, ctx).save(enc_path)
    np.save(npy_path, X_PLAIN)
    real_load = np.load
    pdpg.activate(key_files[0])
    pdpg.install()
    try:
        loaded = np.load(enc_path)
        assert isinstance(loaded, pdpg.CipherArray)
        assert np.allclose(loaded.decrypt(), X_PLAIN, atol=ATOL)
        plain = np.load(npy_path)
        assert isinstance(plain, np.ndarray)
        assert np.array_equal(plain, X_PLAIN)
    finally:
        pdpg.uninstall()
    assert np.load is real_load
