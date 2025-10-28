# method2_feature_selection.py
"""
Method 2 — Feature Selection (MI + robust SHAP)
- MI: bar (Top-k) + cumulative
- SHAP (RF-based): bar (Top-k) + boxplot (Top-10)
Artifacts:
  - artifacts/feature_MI.csv
  - artifacts/feature_SHAP.csv  (if SHAP ok)
Plots:
  - plots/mi_top20.png
  - plots/mi_cumulative.png
  - plots/shap_top20.png        (if SHAP ok)
  - plots/shap_box_top10.png    (if SHAP ok)
"""
import os, warnings
os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore", message="Clustering metrics expects discrete values")

from utils_common import ART_DIR, PLOT_DIR
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
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

def plot_mi_topk_and_cum(feature_names, mi_scores, k=20):
    names = np.asarray(feature_names); scores = np.asarray(mi_scores)
    order = np.argsort(scores)[::-1]
    top = order[:k]
    plt.figure()
    plt.barh(range(len(top)), scores[top])
    plt.yticks(range(len(top)), names[top]); plt.gca().invert_yaxis()
    plt.title("Top MI Features"); plt.tight_layout()
    out1 = PLOT_DIR / "mi_top20.png"; save_dir(out1); plt.savefig(out1, dpi=150); plt.close()

    scores_sorted = scores[order]
    cum = np.cumsum(scores_sorted); 
    if cum[-1] > 0: cum = cum / cum[-1]
    plt.figure()
    plt.plot(range(1, len(cum)+1), cum, marker="o", linewidth=1)
    plt.xlabel("#Features (sorted by MI)"); plt.ylabel("Cumulative MI (normalized)")
    plt.title("Cumulative MI by number of features"); plt.grid(True, alpha=0.3); plt.tight_layout()
    out2 = PLOT_DIR / "mi_cumulative.png"; save_dir(out2); plt.savefig(out2, dpi=150); plt.close()

def shap_bar_and_box(feature_names, shap_values, topk_bar=20, topk_box=10):
    feat = np.array([str(x) for x in feature_names])
    sv = np.asarray(shap_values, dtype=np.float64)
    mean_abs = np.mean(np.abs(sv), axis=0)

    order = np.argsort(mean_abs)[::-1]
    top = order[:topk_bar]
    plt.figure()
    plt.barh(range(len(top)), mean_abs[top])
    plt.yticks(range(len(top)), feat[top]); plt.gca().invert_yaxis()
    plt.title("Top |SHAP| Features (RF)"); plt.tight_layout()
    out1 = PLOT_DIR / "shap_top20.png"; save_dir(out1); plt.savefig(out1, dpi=150); plt.close()

    top10 = order[:topk_box]
    data = [sv[:, j] for j in top10]
    plt.figure()
    plt.boxplot(data, vert=False, showfliers=False)
    plt.yticks(range(1, len(top10)+1), feat[top10])
    plt.title("SHAP value distribution (Top 10)"); plt.tight_layout()
    out2 = PLOT_DIR / "shap_box_top10.png"; save_dir(out2); plt.savefig(out2, dpi=150); plt.close()

    return mean_abs

def main():
    X_train = sparse.load_npz(ART_DIR / "X_train_sparse.npz")
    y_train = joblib.load(ART_DIR / "y_train.pkl")
    feature_names = joblib.load(ART_DIR / "feature_names.pkl")

    # MI
    mi = mutual_info_classif(X_train, y_train, discrete_features=True, random_state=42)
    pd.DataFrame({"feature": feature_names, "MI": mi})\
      .sort_values("MI", ascending=False).to_csv(ART_DIR / "feature_MI.csv", index=False)
    plot_mi_topk_and_cum(feature_names, mi, k=20)

    # method2_feature_selection.py  (เฉพาะบล็อค SHAP)
        # --- SHAP (robust) ---
    try:
        import shap
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)

        n_sample = min(300, X_train.shape[0])
        Xs = X_train[:n_sample].toarray() if hasattr(X_train, "toarray") else X_train[:n_sample]
        Xs = np.asarray(Xs, dtype=np.float64, order="C")

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(Xs)

        # --- normalize shapes ---
        if isinstance(shap_values, list):
            # กรณี binary ส่วนใหญ่จะเป็น list ความยาว 2 → ใช้ของคลาส 1 (positive)
            sv = np.asarray(shap_values[1], dtype=np.float64)
        else:
            sv = np.asarray(shap_values, dtype=np.float64)
            # กรณีได้ (n, f, 2) → เลือกคลาส 1
            if sv.ndim == 3 and sv.shape[2] == 2:
                sv = sv[:, :, 1]
            # กรณีได้ (n, f, 1) → บีบแกนสุดท้าย
            elif sv.ndim == 3 and sv.shape[2] == 1:
                sv = sv[:, :, 0]
            # ป้องกันเหตุสุดวิสัยอื่น ๆ
            elif sv.ndim != 2:
                raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

        # ให้จำนวนฟีเจอร์ตรงกัน
        if sv.shape[1] != len(feature_names):
            F = min(sv.shape[1], len(feature_names))
            sv = sv[:, :F]
            feature_names = feature_names[:F]

        # --- สร้างกราฟ ---
        mean_abs = np.mean(np.abs(sv), axis=0)
        order20 = np.argsort(mean_abs)[::-1][:20]

        plt.figure()
        plt.barh(range(len(order20)), mean_abs[order20])
        plt.yticks(range(len(order20)), np.array(feature_names)[order20])
        plt.gca().invert_yaxis()
        plt.title("Top |SHAP| Features (RF)")
        plt.tight_layout()
        out1 = PLOT_DIR / "shap_top20.png"
        save_dir(out1); plt.savefig(out1, dpi=150); plt.close()

        order10 = np.argsort(mean_abs)[::-1][:10]
        data = [sv[:, j] for j in order10]
        plt.figure()
        plt.boxplot(data, vert=False, showfliers=False)
        plt.yticks(range(1, len(order10)+1), np.array(feature_names)[order10])
        plt.title("SHAP value distribution (Top 10)")
        plt.tight_layout()
        out2 = PLOT_DIR / "shap_box_top10.png"
        save_dir(out2); plt.savefig(out2, dpi=150); plt.close()

        # CSV
        pd.DataFrame({"feature": feature_names, "mean_abs_SHAP": mean_abs})\
          .sort_values("mean_abs_SHAP", ascending=False)\
          .to_csv(ART_DIR / "feature_SHAP.csv", index=False)

        # debug shapes
        with open(ART_DIR / "shap_debug_shapes.txt", "w", encoding="utf-8") as f:
            f.write(f"Xs={Xs.shape}, sv={sv.shape}, n_features={len(feature_names)}\n")

        print("✅ MI & SHAP plots saved.")
    except Exception as e:
        with open(ART_DIR / "shap_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
        print("✅ MI saved. ⚠️ SHAP unavailable; see artifacts/shap_error.txt")

if __name__ == "__main__":
    main()
