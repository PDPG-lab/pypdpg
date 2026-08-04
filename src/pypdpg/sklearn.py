"""Run fitted scikit-learn linear models on encrypted arrays.

Duck-typed on fitted attributes (coef_, intercept_, mean_, scale_, steps),
so scikit-learn is not a dependency of pypdpg — any object exposing the
same attributes works. The intended split: the model owner (typically the
data processor — e.g. a credit-score or fraud-score provider) fits on its
own plaintext training data and keeps the model; inference runs blind on
the controller's ciphertext.
"""

from __future__ import annotations

import numpy as np

from . import approx, errors
from .core import CipherArray
from .pandas import CipherFrame


def wrap(model):
    """Wrap a fitted estimator for encrypted inference.

    Supported: linear regressors (``coef_``/``intercept_``), binary linear
    classifiers (``decision_function``/``predict_proba``), StandardScaler,
    and Pipelines composed of the above.
    """
    if hasattr(model, "steps"):
        return _EncryptedPipeline(model)
    if hasattr(model, "cluster_centers_"):
        return _EncryptedKMeans(model)
    if hasattr(model, "scale_") or hasattr(model, "mean_"):
        return _EncryptedScaler(model)
    if hasattr(model, "coef_"):
        if hasattr(model, "classes_"):
            return _EncryptedClassifier(model)
        return _EncryptedRegressor(model)
    raise TypeError(
        f"pdpg.sklearn.wrap: no encrypted equivalent for "
        f"{type(model).__name__}. Supported: linear models with "
        "coef_/intercept_, StandardScaler, and Pipelines of those."
    )


class _EncryptedRegressor:
    def __init__(self, model):
        self._coef = np.asarray(model.coef_, dtype=np.float64)
        self._intercept = np.asarray(model.intercept_, dtype=np.float64)

    def predict(self, X: CipherArray) -> CipherArray:
        if self._coef.ndim == 1:
            return X @ self._coef + float(self._intercept)
        return X @ self._coef.T + self._intercept  # multi-output


class _EncryptedClassifier:
    def __init__(self, model):
        if len(np.asarray(model.classes_)) != 2:
            raise ValueError(
                "pdpg.sklearn.wrap: only binary classifiers are supported — "
                "multiclass probabilities need softmax (exp and division), "
                "which cannot run under CKKS without interaction."
            )
        self._coef = np.asarray(model.coef_, dtype=np.float64).ravel()
        self._intercept = float(np.asarray(model.intercept_).ravel()[0])

    def decision_function(self, X: CipherArray) -> CipherArray:
        return X @ self._coef + self._intercept

    def predict_proba(self, X: CipherArray) -> CipherArray:
        """P(positive class) via the CKKS logistic approximation.

        Accurate for decision values in roughly [-5, 5]. The result stays
        encrypted; the key holder decrypts and applies the threshold.
        """
        return approx.sigmoid(self.decision_function(X))

    def predict(self, X):
        raise errors.make(
            "E-COERCE",
            "predict() — hard class labels are a plaintext decision, "
            "impossible on encrypted arrays by design.",
        )


class _EncryptedKMeans:
    def __init__(self, model):
        self._centers = np.asarray(model.cluster_centers_, dtype=np.float64)

    def transform_squared(self, X: CipherArray) -> CipherArray:
        """Squared euclidean distance to every centroid, shape (N, k).

        Equals ``model.transform(X) ** 2``. Squaring is monotonic, so the
        ranking of centroids is identical; the key holder can take the
        square root after decryption. Costs one depth level:
        ``|x|^2 - 2 x.c + |c|^2`` is a square and a plain multiply on
        parallel branches.
        """
        if X.ndim != 2:
            raise ValueError("KMeans distances need a 2-D CipherArray.")
        squares = X.square()
        row_norm = squares[:, 0]
        for j in range(1, squares.shape[1]):
            row_norm = row_norm + squares[:, j]
        cross = X @ (-2.0 * self._centers.T)          # (N, k), plain matmul
        center_norms = (self._centers**2).sum(axis=1)  # (k,) plain
        columns = [
            (cross[:, j] + row_norm + float(center_norms[j]))._vectors[0]
            for j in range(self._centers.shape[0])
        ]
        return CipherArray(
            columns, (X.shape[0], len(columns)), "slots", X.context
        )

    def transform(self, X):
        raise errors.make(
            "E-TRANSCEND",
            "KMeans.transform() takes a square root — not supported by the "
            f"current backend ({errors.CURRENT_BACKEND}). Use "
            ".transform_squared(): same centroid ranking; take the square "
            "root after decryption.",
        )

    def predict(self, X):
        raise errors.make(
            "E-COMPARE",
            "KMeans.predict() — choosing the nearest centroid needs argmin, "
            f"not supported by the current backend ({errors.CURRENT_BACKEND}).",
        )


class _EncryptedScaler:
    def __init__(self, model):
        self._mean = getattr(model, "mean_", None)
        self._scale = getattr(model, "scale_", None)

    def transform(self, X: CipherArray) -> CipherArray:
        if self._mean is not None:
            X = X - np.asarray(self._mean, dtype=np.float64)
        if self._scale is not None:
            X = X / np.asarray(self._scale, dtype=np.float64)
        return X


class _EncryptedPipeline:
    def __init__(self, pipeline):
        *heads, tail = [wrap(step) for _, step in pipeline.steps]
        self._transforms = heads
        self._final = tail

    def _apply(self, X: CipherArray) -> CipherArray:
        for step in self._transforms:
            X = step.transform(X)
        return X

    def transform(self, X):
        return self._final.transform(self._apply(X))

    def predict(self, X):
        return self._final.predict(self._apply(X))

    def decision_function(self, X):
        return self._final.decision_function(self._apply(X))

    def predict_proba(self, X):
        return self._final.predict_proba(self._apply(X))

    def transform_squared(self, X):
        return self._final.transform_squared(self._apply(X))


# --- targeted runtime hooks -------------------------------------------------
#
# pdpg.sklearn.install() patches the *public* methods of exactly the
# estimator classes wrap() supports, with a type gate at the top:
# encrypted input routes into wrap() (same code, same teaching errors),
# plain input calls the original, untouched. Opt-in, idempotent, and
# reversible with uninstall().

_PATCHED: dict = {}  # (cls, name) -> original entry from cls.__dict__, or None


def _class_attr(cls, name):
    """The raw class attribute (function or descriptor) through the MRO."""
    for base in cls.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    raise AttributeError(name)


def _as_cipher(X):
    return X.values if isinstance(X, CipherFrame) else X


def _gated(cls, name):
    original = _class_attr(cls, name)

    def method(self, X=None, *args, **kwargs):
        if isinstance(X, (CipherArray, CipherFrame)):
            if name == "fit":
                raise errors.make(
                    "E-COERCE",
                    "fit() on encrypted data — training happens on "
                    "plaintext, where the training data's owner sits; "
                    "pypdpg runs inference only.",
                )
            return getattr(wrap(self), name)(_as_cipher(X))
        return original.__get__(self, cls)(X, *args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f"{cls.__name__}.{name}"
    return method


def _transform_squared(self, X):
    """Squared distances to centroids; works on plain and encrypted input."""
    if isinstance(X, (CipherArray, CipherFrame)):
        return wrap(self).transform_squared(_as_cipher(X))
    return self.transform(X) ** 2


def _targets():
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.linear_model import (
        ElasticNet,
        Lasso,
        LinearRegression,
        LogisticRegression,
        Ridge,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    patched = [
        (LinearRegression, ("fit", "predict")),
        (Ridge, ("fit", "predict")),
        (Lasso, ("fit", "predict")),
        (ElasticNet, ("fit", "predict")),
        (LogisticRegression, ("fit", "predict", "predict_proba", "decision_function")),
        (StandardScaler, ("fit", "transform")),
        (KMeans, ("fit", "transform", "predict")),
        (MiniBatchKMeans, ("fit", "transform", "predict")),
        (Pipeline, ("fit", "predict", "predict_proba", "decision_function", "transform")),
    ]
    added = (KMeans, MiniBatchKMeans, Pipeline)  # gain .transform_squared
    return patched, added


def install() -> None:
    """Hook the supported scikit-learn estimators for encrypted inference.

    After this, the same estimator objects accept plain arrays (original
    behavior, byte for byte) and CipherArrays/CipherFrames (routed through
    pdpg.sklearn.wrap). One line on the data processor's side.
    """
    if _PATCHED:
        return  # already installed
    patched, added = _targets()
    for cls, names in patched:
        for name in names:
            _PATCHED[(cls, name)] = cls.__dict__.get(name)
            setattr(cls, name, _gated(cls, name))
    for cls in added:
        _PATCHED[(cls, "transform_squared")] = cls.__dict__.get("transform_squared")
        cls.transform_squared = _transform_squared


def uninstall() -> None:
    """Restore every hooked method exactly as it was."""
    for (cls, name), original in _PATCHED.items():
        if original is None:
            if name in cls.__dict__:
                delattr(cls, name)
        else:
            setattr(cls, name, original)
    _PATCHED.clear()
