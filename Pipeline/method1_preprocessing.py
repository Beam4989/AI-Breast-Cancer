# method1_preprocessing.py
"""
Method 1 — Data Collection & Preprocessing
- Load: ..\breast+cancer\breast-cancer.data (relative to Pipeline/)
- Clean: impute missing (num=median / cat=most_frequent)
- Encode: OneHotEncoder(handle_unknown=ignore), scale numeric (StandardScaler with_mean=False)
- Split: train/test (stratify)
- Save artifacts: preprocessor.pkl, X_train_sparse.npz, X_test_sparse.npz, y_train.pkl, y_test.pkl, feature_names.pkl
"""
import inspect
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy import sparse
import joblib

from utils_common import ART_DIR, PLOT_DIR, BASE_DIR

# ใช้พาธตามที่กำหนด
RAW_PATH = (BASE_DIR.parent / "breast+cancer" / "breast-cancer.data")
if not RAW_PATH.exists():
    raise FileNotFoundError(f"Data file not found at: {RAW_PATH}")

COLUMNS = [
    "Class", "age", "menopause", "tumor-size", "inv-nodes",
    "node-caps", "deg-malig", "breast", "breast-quad", "irradiat",
]
TARGET = "Class"

def _make_ohe(sparse_flag: bool = True) -> OneHotEncoder:
    sig = inspect.signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_flag)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse_flag)

def load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=COLUMNS, na_values=["?"], dtype=str)
    df["deg-malig"] = pd.to_numeric(df["deg-malig"], errors="coerce")
    return df

def split_target(df: pd.DataFrame):
    y = df[TARGET].map({"no-recurrence-events": 0, "recurrence-events": 1}).fillna(0).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y

def main():
    # 1) Load
    df = load_raw(RAW_PATH)
    X, y = split_target(df)

    # 2) Column types
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # 3) Preprocessors
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_ohe(sparse_flag=True)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5) Fit/transform
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)

    # 6) Save artifacts
    joblib.dump(pre, ART_DIR / "preprocessor.pkl")
    joblib.dump(y_train, ART_DIR / "y_train.pkl")
    joblib.dump(y_test, ART_DIR / "y_test.pkl")
    sparse.save_npz(ART_DIR / "X_train_sparse.npz", X_train_enc)
    sparse.save_npz(ART_DIR / "X_test_sparse.npz", X_test_enc)

    # 7) Save feature names
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    feature_names = numeric_cols + cat_feature_names
    joblib.dump(feature_names, ART_DIR / "feature_names.pkl")

    print("✅ Preprocessing complete. Artifacts saved in ./artifacts")

if __name__ == "__main__":
    main()
