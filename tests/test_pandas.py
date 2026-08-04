"""Oracle tests for the CipherFrame facade against real pandas."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import pypdpg as pdpg
from pypdpg.errors import EncryptedOperationError

rng = np.random.default_rng(5)
DF = pd.DataFrame(
    {
        "income": rng.normal(50_000, 15_000, 40),
        "debt": rng.normal(20_000, 8_000, 40),
        "age": rng.uniform(20, 70, 40),
    }
)

ATOL = 1e-2


@pytest.fixture(scope="module")
def ctx():
    return pdpg.Context.create()


@pytest.fixture(scope="module")
def encdf(ctx):
    return pdpg.encrypt(DF, ctx)


def test_encrypt_dataframe_roundtrip(ctx, encdf):
    assert isinstance(encdf, pdpg.CipherFrame)
    assert encdf.columns == ["income", "debt", "age"]
    out = encdf.decrypt()
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(DF.columns)
    assert np.allclose(out.to_numpy(), DF.to_numpy(), atol=ATOL)


def test_column_select_and_compute(ctx, encdf):
    got = (encdf["income"] * 12).decrypt()
    assert np.allclose(got, DF["income"].to_numpy() * 12, atol=ATOL)


def test_column_subset(ctx, encdf):
    sub = encdf[["income", "age"]]
    assert sub.columns == ["income", "age"]
    assert np.allclose(
        sub.decrypt().to_numpy(), DF[["income", "age"]].to_numpy(), atol=ATOL
    )


def test_missing_column_raises(encdf):
    with pytest.raises(KeyError, match="salary"):
        encdf["salary"]


def test_mean_returns_labeled_series(ctx, encdf):
    got = encdf.mean().decrypt()
    assert isinstance(got, pd.Series)
    assert list(got.index) == ["income", "debt", "age"]
    assert np.allclose(got.to_numpy(), DF.mean().to_numpy(), atol=ATOL)


def test_sum_returns_labeled_series(ctx, encdf):
    got = encdf.sum().decrypt()
    assert np.allclose(got.to_numpy(), DF.sum().to_numpy(), atol=ATOL)


def test_frame_arithmetic(ctx, encdf):
    got = ((encdf - 100) * 2).decrypt()
    assert np.allclose(got.to_numpy(), (DF - 100).to_numpy() * 2, atol=ATOL)


def test_assign_computed_column(ctx):
    frame = pdpg.encrypt(DF, ctx)
    frame["debt_k"] = frame["debt"] * 0.001
    assert frame.columns[-1] == "debt_k"
    out = frame.decrypt()
    assert np.allclose(out["debt_k"], DF["debt"] * 0.001, atol=ATOL)


def test_division_by_cipher_column_raises(encdf):
    with pytest.raises(EncryptedOperationError, match="Why:"):
        encdf["debt"] / encdf["income"]


def test_matmul_via_values(ctx, encdf):
    w = np.array([0.5, -0.2, 1.0])
    got = (encdf @ w).decrypt()
    assert np.allclose(got, DF.to_numpy() @ w, atol=ATOL)


def test_save_load_preserves_columns(ctx, encdf, tmp_path):
    path = tmp_path / "frame.enc"
    encdf.save(path)
    loaded = pdpg.load(path, ctx)
    assert isinstance(loaded, pdpg.CipherFrame)
    assert loaded.columns == encdf.columns
    assert np.allclose(loaded.decrypt().to_numpy(), DF.to_numpy(), atol=ATOL)


def test_np_load_returns_frame(ctx, encdf, tmp_path, monkeypatch):
    path = tmp_path / "frame.enc"
    encdf.save(path)
    import pypdpg.context as ctx_module

    monkeypatch.setattr(ctx_module, "_default_context", ctx)
    pdpg.install()
    try:
        loaded = np.load(path)
        assert isinstance(loaded, pdpg.CipherFrame)
        assert loaded.columns == ["income", "debt", "age"]
    finally:
        pdpg.uninstall()


def test_repr_shows_columns_never_values(encdf):
    r = repr(encdf)
    assert "income" in r and "🙈" in r
    assert not any(f"{v:.1f}" in r for v in DF["income"].to_numpy()[:3])
