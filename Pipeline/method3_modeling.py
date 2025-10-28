# method3_modeling.py  (เฉพาะส่วนที่ต่างจากของเดิม)
import numpy as np
from scipy import sparse
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from utils_common import ART_DIR
from label_wrapper import LabelOnlyClassifier   # <<-- NEW
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def cv_auc(model, X, y):
    from numpy import nanmean
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return nanmean(scores)

def main():
    X_train = sparse.load_npz(ART_DIR / "X_train_sparse.npz")
    y_train = joblib.load(ART_DIR / "y_train.pkl")

    candidates = []

    lr = LabelOnlyClassifier(LogisticRegression(max_iter=1000), threshold=0.5)
    candidates.append(("LogReg", lr, cv_auc(lr, X_train, y_train)))

    rf = LabelOnlyClassifier(RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=1), threshold=0.5)
    candidates.append(("RandomForest", rf, cv_auc(rf, X_train, y_train)))

    lsvc = LabelOnlyClassifier(LinearSVC(random_state=42), threshold=0.5)
    candidates.append(("LinearSVC", lsvc, cv_auc(lsvc, X_train, y_train)))

    valid = [(n, m, a) for (n, m, a) in candidates if not np.isnan(a)]
    best_name, best_model, best_auc = (valid or candidates)[0]
    if valid:
        best_name, best_model, best_auc = sorted(valid, key=lambda t: t[2], reverse=True)[0]

    best_model.fit(X_train, y_train)
    joblib.dump(best_model, ART_DIR / "best_model.pkl")
    print(f"✅ Best model saved as label-only: {best_name} (CV ROC-AUC={best_auc:.3f}) -> artifacts/best_model.pkl")

if __name__ == "__main__":
    main()
