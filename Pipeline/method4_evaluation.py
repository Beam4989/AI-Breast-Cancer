# method4_evaluation.py — sparse-aware, val-from-train, plots all
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import load_npz, issparse, csr_matrix

from utils_common import ART_DIR  # points to Pipeline/artifacts

CLASS_NAMES = ["Negative", "Positive"]
FIG_DPI = 200

print(f"📂 ART_DIR = {ART_DIR}")

# ---------- Load artifacts that exist ----------
# preprocessor.pkl / feature_names.pkl เผื่อไว้ แต่จะไม่ transform ซ้ำถ้า X เป็น sparse ที่เข้ารหัสแล้ว
pre_path = ART_DIR / "preprocessor.pkl"
feat_path = ART_DIR / "feature_names.pkl"
pre = joblib.load(pre_path) if pre_path.exists() else None
feat_names = joblib.load(feat_path) if feat_path.exists() else None

# ---------- Load X/y (sparse) ----------
def load_sparse_or_fail(name: str) -> csr_matrix:
    p = ART_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    X = load_npz(p)
    if not issparse(X):
        X = csr_matrix(X)
    print(f"✅ Loaded sparse: {p.name} shape={X.shape}")
    return X

def load_pkl_or_fail(name: str):
    p = ART_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    y = joblib.load(p)
    print(f"✅ Loaded: {p.name} shape={(len(y),) if hasattr(y,'__len__') else 'unknown'}")
    return np.array(y).ravel()


# มีไฟล์ train/test (ไม่มี val) ⇒ จะ split val จาก train
X_train = load_sparse_or_fail("X_train_sparse.npz")
y_train = load_pkl_or_fail("y_train.pkl")
X_test  = load_sparse_or_fail("X_test_sparse.npz")
y_test  = load_pkl_or_fail("y_test.pkl")

# ---------- Build validation from train (20%) ----------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)
print(f"🧪 Split val from train: train={X_tr.shape}, val={X_val.shape}, test={X_test.shape}")

n_samples, n_features = X_tr.shape
use_dual = True if n_features > n_samples else False

# ---------- Utilities ----------
def safe_y_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return model.predict(X).astype(float)

def densify_if_needed(X):
    return X.toarray() if issparse(X) else X

def plot_cm(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(values_format='d')
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(ART_DIR / f"fig_confusion_matrix_{name}.png", dpi=FIG_DPI)
    plt.close()

def plot_cls_report(y_true, y_pred, name):
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T
    fig, ax = plt.subplots(figsize=(8, 3 + 0.3*len(df)))
    ax.axis('off')
    tbl = ax.table(cellText=np.round(df.values, 3),
                   colLabels=df.columns,
                   rowLabels=df.index,
                   loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.2)
    plt.title(f"Classification Report — {name}")
    plt.tight_layout()
    plt.savefig(ART_DIR / f"fig_classification_report_{name}.png", dpi=FIG_DPI)
    plt.close()

def plot_roc(y_true, y_score, name):
    auc = roc_auc_score(y_true, y_score)
    RocCurveDisplay.from_predictions(y_true, y_score, name=f"{name} (AUC={auc:.3f})")
    plt.title(f"ROC Curve — {name}")
    plt.tight_layout()
    plt.savefig(ART_DIR / f"fig_roc_{name}.png", dpi=FIG_DPI)
    plt.close()

def plot_model_comparison(df):
    metrics = ["AUC","F1","Recall","Precision","Accuracy"]
    fig, ax = plt.subplots(figsize=(10,5))
    df.plot(x="model", y=metrics, kind="bar", ax=ax)
    plt.title("Model Performance Comparison"); plt.ylabel("Score"); plt.ylim(0,1.0)
    plt.legend(title="Metric", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout(); plt.savefig(ART_DIR / "fig_model_comparison.png", dpi=FIG_DPI); plt.close()

def plot_model_table(df):
    data = np.round(df[["AUC","F1","Recall","Precision","Accuracy"]].values, 3)
    fig, ax = plt.subplots(figsize=(10, 0.6*len(df)+2)); ax.axis('off')
    tbl = ax.table(cellText=data,
                   colLabels=["AUC","F1","Recall","Precision","Accuracy"],
                   rowLabels=df["model"].tolist(),
                   loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.2)
    plt.title("Model Performance Summary")
    plt.tight_layout(); plt.savefig(ART_DIR / "fig_model_performance_table.png", dpi=FIG_DPI); plt.close()

def plot_feature_selection_from_csv(csv_path: Path, title: str, filename: str):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    # เดาชื่อคอลัมน์: ใช้คอลัมน์แรกที่เป็น string เป็นชื่อฟีเจอร์, คอลัมน์ตัวเลขสุดท้ายเป็นคะแนน
    feature_col = None
    for c in df.columns:
        if df[c].dtype == object:
            feature_col = c
            break
    if feature_col is None:
        return
    score_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not score_cols:
        return
    score_col = score_cols[-1]
    s = pd.Series(df[score_col].values, index=df[feature_col].values).sort_values(ascending=True)
    plt.figure(figsize=(8, max(2, 0.35*len(s)+2)))
    plt.barh(s.index, s.values)
    plt.title(title); plt.xlabel(score_col)
    plt.tight_layout(); plt.savefig(ART_DIR / filename, dpi=FIG_DPI); plt.close()

def plot_feature_importance(model, feature_names, X_val=None, y_val=None, name="model"):
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = coef[0]
        imp = np.abs(coef.astype(float))
    else:
        if (X_val is not None) and (y_val is not None):
            r = permutation_importance(model, densify_if_needed(X_val), y_val, n_repeats=10, random_state=42)
            imp = r.importances_mean
    if imp is None:
        print(f"[WARN] No feature importance for {name}")
        return
    if feature_names is None or len(feature_names) != imp.shape[0]:
        feature_names_local = [f"f{i}" for i in range(imp.shape[0])]
    else:
        feature_names_local = feature_names
    s = pd.Series(imp, index=feature_names_local).sort_values(ascending=True)
    plt.figure(figsize=(8, max(2, 0.35*len(s)+2)))
    plt.barh(s.index, s.values)
    plt.title(f"Feature Importance — {name}"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(ART_DIR / f"fig_feature_importance_{name}.png", dpi=FIG_DPI); plt.close()

# ---------- Models (sparse-friendly) ----------
models = {
    "LogisticRegression": LogisticRegression(max_iter=200, solver="liblinear"),  # supports CSR
    "LinearSVC": LinearSVC(C=1.0, random_state=42),                               # supports CSR
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42)     # needs dense
}

# Fit
for name, clf in models.items():
    if name == "RandomForest":
        clf.fit(densify_if_needed(X_tr), y_tr)
    else:
        clf.fit(X_tr, y_tr)

# Evaluate + plots
records = []
for name, clf in models.items():
    X_eval = densify_if_needed(X_test) if name == "RandomForest" else X_test
    y_score = safe_y_score(clf, X_eval)
    y_pred  = clf.predict(X_eval)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_score)

    records.append({"model":name, "Accuracy":acc, "Precision":pre, "Recall":rec, "F1":f1, "AUC":auc})

    plot_cm(y_test, y_pred, name)
    plot_cls_report(y_test, y_pred, name)
    plot_roc(y_test, y_score, name)

    # importance (val set for permutation if needed)
    X_val_used = densify_if_needed(X_val) if name == "RandomForest" else X_val
    plot_feature_importance(clf, feat_names, X_val=X_val_used, y_val=y_val, name=name)

res_df = pd.DataFrame(records).sort_values("AUC", ascending=False).reset_index(drop=True)
plot_model_comparison(res_df)
plot_model_table(res_df)

# ---------- Feature Selection (from CSVs) ----------
plot_feature_selection_from_csv(ART_DIR / "feature_MI.csv",
    "Feature Selection (Mutual Information) — All Features", "fig_feature_selection_mi.png")
plot_feature_selection_from_csv(ART_DIR / "feature_SHAP.csv",
    "Feature Selection (SHAP) — All Features", "fig_feature_selection_shap.png")
plot_feature_selection_from_csv(ART_DIR / "permutation_importance.csv",
    "Feature Selection (Permutation Importance) — All Features", "fig_feature_selection_perm.png")
plot_feature_selection_from_csv(ART_DIR / "rf_gini_importance.csv",
    "Feature Selection (RF Gini Importance) — All Features", "fig_feature_selection_rf_gini.png")

print("\n✅ All evaluation figures saved under:", ART_DIR)
