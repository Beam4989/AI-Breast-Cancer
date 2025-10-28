# utils_common.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import inspect

# ----- Paths anchored to this file -----
BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"
PLOT_DIR = BASE_DIR / "plots"
ART_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# ----- Dataset schema (UCI Breast Cancer - Recurrence) -----
COLUMNS = [
    "Class", "age", "menopause", "tumor-size", "inv-nodes",
    "node-caps", "deg-malig", "breast", "breast-quad", "irradiat",
]
TARGET = "Class"

def load_raw(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=COLUMNS, na_values=["?"], dtype=str)
    df["deg-malig"] = pd.to_numeric(df["deg-malig"], errors="coerce")
    return df

def split_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET].map({"no-recurrence-events": 0, "recurrence-events": 1}).fillna(0).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y

def _make_ohe(sparse_flag: bool = True) -> OneHotEncoder:
    sig = inspect.signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_flag)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse_flag)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    num_pipe = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline(steps=[("onehot", _make_ohe(sparse_flag=True))])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)]
    )
    return pre, numeric_cols, categorical_cols

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# ----- Wrapper: output label only (0/1) -----
class LabelOnlyClassifier:
    """
    Wrap base estimator so that predict() returns only 0/1 labels with a fixed threshold (0.5).
    """
    def __init__(self, base_estimator, threshold: float = 0.5):
        self.base_estimator = base_estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X):
        be = self.base_estimator
        if hasattr(be, "predict_proba"):
            import numpy as np
            proba = be.predict_proba(X)[:, 1]
            return (proba >= self.threshold).astype(int)
        if hasattr(be, "decision_function"):
            import numpy as np
            score = be.decision_function(X)
            return (score >= 0).astype(int)
        return be.predict(X)

    def __getstate__(self):
        return {"base_estimator": self.base_estimator, "threshold": self.threshold}

    def __setstate__(self, state):
        self.base_estimator = state["base_estimator"]
        self.threshold = state["threshold"]
