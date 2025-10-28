# label_wrapper.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class LabelOnlyClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper ให้โมเดลใด ๆ ส่งออก hard label ด้วย threshold=0.5
    - รองรับ predict_proba หรือ decision_function
    - เป็น estimator สมบูรณ์ (clone ได้) และมี classes_
    """
    def __init__(self, base, threshold=0.5):
        self.base = base
        self.threshold = threshold

    def get_params(self, deep=True):
        return {"base": self.base, "threshold": self.threshold}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self.base_ = clone(self.base)
        self.base_.fit(X, y)
        self.classes_ = getattr(self.base_, "classes_", np.array([0, 1], dtype=int))
        return self

    def predict_proba(self, X):
        if hasattr(self.base_, "predict_proba"):
            return self.base_.predict_proba(X)
        if hasattr(self.base_, "decision_function"):
            m = np.ravel(self.base_.decision_function(X))
            p1 = 1.0 / (1.0 + np.exp(-m))
            return np.vstack([1 - p1, p1]).T
        p = np.ravel(self.base_.predict(X)).astype(float)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
