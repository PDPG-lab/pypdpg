"""Oracle tests for pdpg.sklearn.wrap: encrypted inference must match the
fitted model's own plaintext predictions."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pypdpg as pdpg
from pypdpg.errors import EncryptedOperationError

rng = np.random.default_rng(3)
X = rng.normal(size=(60, 4))
y_reg = X @ rng.normal(size=4) + 0.5 + rng.normal(scale=0.1, size=60)
Y_multi = np.column_stack([y_reg, -2.0 * y_reg + 1.0])
y_cls = (X[:, 0] - X[:, 1] > 0).astype(int)

ATOL = 1e-2


@pytest.fixture(scope="module")
def ctx():
    return pdpg.Context.create()


@pytest.fixture(scope="module")
def encX(ctx):
    return pdpg.encrypt(X, ctx)


def test_linear_regression_predict(ctx, encX):
    model = LinearRegression().fit(X, y_reg)
    got = pdpg.sklearn.wrap(model).predict(encX).decrypt()
    assert np.allclose(got, model.predict(X), atol=ATOL)


def test_ridge_predict(ctx, encX):
    model = Ridge(alpha=1.0).fit(X, y_reg)
    got = pdpg.sklearn.wrap(model).predict(encX).decrypt()
    assert np.allclose(got, model.predict(X), atol=ATOL)


def test_multioutput_regression_predict(ctx, encX):
    model = LinearRegression().fit(X, Y_multi)
    got = pdpg.sklearn.wrap(model).predict(encX).decrypt()
    assert got.shape == (60, 2)
    assert np.allclose(got, model.predict(X), atol=ATOL)


def test_logistic_decision_function(ctx, encX):
    model = LogisticRegression().fit(X, y_cls)
    got = pdpg.sklearn.wrap(model).decision_function(encX).decrypt()
    assert np.allclose(got, model.decision_function(X), atol=ATOL)


def test_logistic_predict_proba(ctx, encX):
    # C=0.1 keeps decision values inside the sigmoid approximation's
    # valid range (~[-5, 5]); beyond it the polynomial diverges by design
    model = LogisticRegression(C=0.1).fit(X, y_cls)
    got = pdpg.sklearn.wrap(model).predict_proba(encX).decrypt()
    # oracle: the same logistic approximation evaluated on plaintext —
    # this isolates encryption error from polynomial-approximation error
    expected = pdpg.approx.sigmoid(model.decision_function(X))
    assert np.allclose(got, expected, atol=ATOL)
    # and loosely against sklearn's true probabilities
    assert np.allclose(got, model.predict_proba(X)[:, 1], atol=1e-1)


def test_classifier_predict_raises(encX):
    model = LogisticRegression().fit(X, y_cls)
    with pytest.raises(EncryptedOperationError, match="Why:"):
        pdpg.sklearn.wrap(model).predict(encX)


def test_scaler_transform(ctx, encX):
    scaler = StandardScaler().fit(X)
    got = pdpg.sklearn.wrap(scaler).transform(encX).decrypt()
    assert np.allclose(got, scaler.transform(X), atol=ATOL)


def test_pipeline_scaler_regression(ctx, encX):
    pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y_reg)
    got = pdpg.sklearn.wrap(pipe).predict(encX).decrypt()
    assert np.allclose(got, pipe.predict(X), atol=ATOL)


def test_pipeline_scaler_logistic_proba(ctx, encX):
    # scaler (1 level) + linear (1) + sigmoid (2) = full depth budget
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(X, y_cls)
    got = pdpg.sklearn.wrap(pipe).predict_proba(encX).decrypt()
    expected = pdpg.approx.sigmoid(pipe.decision_function(X))
    assert np.allclose(got, expected, atol=ATOL)


def test_kmeans_transform_squared(ctx, encX):
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    got = pdpg.sklearn.wrap(model).transform_squared(encX).decrypt()
    assert got.shape == (60, 3)
    assert np.allclose(got, model.transform(X) ** 2, atol=ATOL)


def test_kmeans_pipeline(ctx, encX):
    from sklearn.cluster import KMeans

    pipe = make_pipeline(
        StandardScaler(), KMeans(n_clusters=3, random_state=0, n_init=10)
    ).fit(X)
    got = pdpg.sklearn.wrap(pipe).transform_squared(encX).decrypt()
    expected = pipe.named_steps["kmeans"].transform(
        pipe.named_steps["standardscaler"].transform(X)
    ) ** 2
    assert np.allclose(got, expected, atol=ATOL)


def test_kmeans_transform_and_predict_raise(encX):
    from sklearn.cluster import KMeans

    wrapped = pdpg.sklearn.wrap(KMeans(n_clusters=2, random_state=0, n_init=10).fit(X))
    with pytest.raises(EncryptedOperationError, match="transform_squared"):
        wrapped.transform(encX)
    with pytest.raises(EncryptedOperationError, match="Why:"):
        wrapped.predict(encX)


# ------------------------------------------------- targeted runtime hooks

@pytest.fixture()
def hooks():
    pdpg.sklearn.install()
    yield
    pdpg.sklearn.uninstall()


def test_uninstall_restores_identity():
    from sklearn.linear_model import LinearRegression

    original = LinearRegression.predict
    pdpg.sklearn.install()
    assert LinearRegression.predict is not original
    pdpg.sklearn.install()  # idempotent
    pdpg.sklearn.uninstall()
    assert LinearRegression.predict is original


def test_hooked_predict_cipher_and_plain(hooks, ctx, encX):
    model = LinearRegression().fit(X, y_reg)
    plain_before = model.predict(X)
    enc_result = model.predict(encX)
    assert isinstance(enc_result, pdpg.CipherArray)
    assert np.allclose(enc_result.decrypt(), plain_before, atol=ATOL)
    # plain path is untouched by the hook
    assert np.array_equal(model.predict(X), plain_before)


def test_hooked_pipeline_proba(hooks, ctx, encX):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(X, y_cls)
    got = pipe.predict_proba(encX).decrypt()
    expected = pdpg.approx.sigmoid(pipe.decision_function(X))
    assert np.allclose(got, expected, atol=ATOL)
    assert pipe.predict_proba(X).shape == (60, 2)  # plain path still sklearn's


def test_hooked_fit_on_cipher_teaches(hooks, encX):
    with pytest.raises(EncryptedOperationError, match="inference only"):
        LinearRegression().fit(encX, y_reg)


def test_hooked_kmeans_transform_squared_both_worlds(hooks, ctx, encX):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    assert np.allclose(km.transform_squared(X), km.transform(X) ** 2)
    got = km.transform_squared(encX).decrypt()
    assert np.allclose(got, km.transform(X) ** 2, atol=ATOL)
    with pytest.raises(EncryptedOperationError, match="transform_squared"):
        km.transform(encX)


def test_unhooked_estimator_still_refuses(hooks, encX):
    from sklearn.ensemble import RandomForestRegressor

    forest = RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y_reg)
    with pytest.raises(EncryptedOperationError, match="Why:"):
        forest.predict(encX)


def test_multiclass_rejected():
    y3 = rng.integers(0, 3, size=60)
    model = LogisticRegression().fit(X, y3)
    with pytest.raises(ValueError, match="binary"):
        pdpg.sklearn.wrap(model)


def test_unsupported_model_rejected():
    with pytest.raises(TypeError, match="wrap"):
        pdpg.sklearn.wrap(object())
