# method5_explainability.py
"""
Method 5 — Explainability (robust)
- RF Gini: bar
- Permutation Importance: bar + error bars, scatter (Gini vs Permutation)
- SHAP: bar (Top-20 mean |SHAP|) + boxplot (Top-10)
"""
import os, warnings
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message="Clustering metrics expects discrete values")

import numpy as np
import pandas as pd
from utils_common import ART_DIR, PLOT_DIR
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib
import matplotlib.pyplot as plt
import joblib

# Thai font (optional)
for _font in ["Noto Sans Thai", "Sarabun", "Tahoma", "Angsana New"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

def save_dir(p): p.parent.mkdir(parents=True, exist_ok=True)

def bar_topk(names, scores, title, out_png, k=20):
    names = np.asarray(names); scores = np.asarray(scores)
    order = np.argsort(scores)[::-1]; top = order[:k]
    plt.figure()
    plt.barh(range(len(top)), scores[top]); plt.yticks(range(len(top)), names[top])
    plt.gca().invert_yaxis(); plt.title(title); plt.tight_layout()
    out = PLOT_DIR / out_png; save_dir(out); plt.savefig(out, dpi=150); plt.close()

def bar_with_error(names, means, stds, title, out_png, k=20):
    names = np.asarray(names); means = np.asarray(means); stds = np.asarray(stds)
    order = np.argsort(means)[::-1]; top = order[:k]
    plt.figure()
    plt.barh(range(len(top)), means[top], xerr=stds[top], capsize=3)
    plt.yticks(range(len(top)), names[top]); plt.gca().invert_yaxis()
    plt.title(title); plt.tight_layout()
    out = PLOT_DIR / out_png; save_dir(out); plt.savefig(out, dpi=150); plt.close()

def shap_boxplot_top10(feature_names, sv, out_png="shap_mean_abs_top10_box.png"):
    feat = np.array([str(x) for x in feature_names])
    sv = np.asarray(sv, dtype=np.float64)
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    data = [sv[:, j] for j in order]
    plt.figure()
    plt.boxplot(data, vert=False, showfliers=False)
    plt.yticks(range(1, len(order)+1), feat[order])
    plt.title("SHAP value distribution (Top 10)")
    plt.tight_layout()
    out = PLOT_DIR / out_png; save_dir(out); plt.savefig(out, dpi=150); plt.close()

def main():
    X_train = sparse.load_npz(ART_DIR / "X_train_sparse.npz")
    y_train = joblib.load(ART_DIR / "y_train.pkl")
    X_test  = sparse.load_npz(ART_DIR / "X_test_sparse.npz")
    y_test  = joblib.load(ART_DIR / "y_test.pkl")
    feature_names = joblib.load(ART_DIR / "feature_names.pkl")

    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)

    # Gini
    gini = rf.feature_importances_
    pd.DataFrame({"feature": feature_names, "rf_importance": gini})\
      .sort_values("rf_importance", ascending=False).to_csv(ART_DIR / "rf_gini_importance.csv", index=False)
    bar_topk(feature_names, gini, "RandomForest Feature Importance (Gini)", "rf_importance_top20.png", k=20)

    # Permutation
    X_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    X_dense = np.asarray(X_dense, dtype=np.float64, order="C")
    n = X_dense.shape[0]
    idx = np.random.RandomState(42).choice(n, size=min(300, n), replace=False) if n > 300 else np.arange(n)
    Xs, ys = X_dense[idx], np.asarray(y_test)[idx]

    perm = permutation_importance(rf, Xs, ys, n_repeats=5, random_state=42, n_jobs=1, scoring="roc_auc")
    pd.DataFrame({"feature": feature_names,
                  "perm_importance_mean": perm.importances_mean,
                  "perm_importance_std":  perm.importances_std})\
      .sort_values("perm_importance_mean", ascending=False).to_csv(ART_DIR / "permutation_importance.csv", index=False)
    bar_with_error(feature_names, perm.importances_mean, perm.importances_std,
                   "Permutation Importance (ROC-AUC)", "perm_importance_top20.png", k=20)

    # Scatter
    plt.figure()
    plt.scatter(gini, perm.importances_mean, s=12, alpha=0.7)
    plt.xlabel("Gini importance (RF)")
    plt.ylabel("Permutation importance (mean ROC-AUC drop)")
    plt.title("Gini vs Permutation Importance"); plt.grid(True, alpha=0.3); plt.tight_layout()
    out_sc = PLOT_DIR / "gini_vs_permutation_scatter.png"; save_dir(out_sc); plt.savefig(out_sc, dpi=150); plt.close()

    # SHAP
    try:
        import shap
        n_shap = min(150, X_dense.shape[0]); Xsh = X_dense[:n_shap]
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(Xsh)
        sv = np.asarray(shap_values[1], dtype=np.float64) if isinstance(shap_values, list) and len(shap_values)==2 \
             else np.asarray(shap_values, dtype=np.float64)

        mean_abs = np.mean(np.abs(sv), axis=0)
        pd.DataFrame({"feature": feature_names, "mean_abs_SHAP": mean_abs})\
          .sort_values("mean_abs_SHAP", ascending=False).to_csv(ART_DIR / "explain_shap_mean_abs.csv", index=False)

        bar_topk(feature_names, mean_abs, "Mean |SHAP| (RF)", "shap_mean_abs_top20.png", k=20)
        shap_boxplot_top10(feature_names, sv, out_png="shap_mean_abs_top10_box.png")
    except Exception as e:
        with open(ART_DIR / "explain_shap_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    print("✅ Explainability plots saved in ./plots and summaries in ./artifacts")

if __name__ == "__main__":
    main()
