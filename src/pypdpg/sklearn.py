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
            "E-COMPARE",
            "predict() — hard class labels are impossible on CKKS "
            "ciphertexts.",
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
            "KMeans.transform() takes a square root — impossible on CKKS "
            "ciphertexts. Use .transform_squared(): same centroid ranking; "
            "take the square root after decryption.",
        )

    def predict(self, X):
        raise errors.make(
            "E-COMPARE",
            "KMeans.predict() — choosing the nearest centroid needs argmin, "
            "impossible on CKKS ciphertexts.",
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
