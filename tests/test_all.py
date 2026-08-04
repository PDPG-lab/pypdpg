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

ATOL = 1e-2


@pytest.fixture(scope="session")
def ctx():
    return pdpg.Context.create()


@pytest.fixture(scope="session")
def pub_ctx(ctx, tmp_path_factory):
    path = tmp_path_factory.mktemp("keys") / "vendor.ctx"
    ctx.save_public(path)
    return pdpg.Context.load(path)


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
