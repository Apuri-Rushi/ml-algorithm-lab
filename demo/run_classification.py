"""
Experiment 1 - Classification on Chronic Kidney Disease (KNN, Decision Tree, SVM-RBF).

Local version of classification/classification/classification_ (2).py
(the original is a Google Colab export that mounts Drive and reads /content/).

Run from the repository root:
    python demo/run_classification.py            # full grid search (about 10-20 s)
    python demo/run_classification.py --quick    # skip grid search, use the best params found
    python demo/run_classification.py --patient 5  # also explain the prediction for test patient #5
    python demo/run_classification.py --show     # open the plots in windows as well
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "raw" / "chronic_kidney_disease" / "chronic_kidney_disease.arff"
OUT = ROOT / "demo" / "outputs" / "classification"

NUMERICAL = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
             "hemo", "pcv", "wbcc", "rbcc"]
CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]


def load_ckd(path=DATA):
    """Parse the ARFF file by hand (pandas cannot read ARFF directly)."""
    columns, rows, in_data = [], [], False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                columns.append(line.split()[1].strip("'\""))
            elif line.lower().startswith("@data"):
                in_data = True
            elif in_data:
                values = [v.strip() for v in line.split(",")]
                # a few rows have stray trailing / double commas
                while len(values) > len(columns) and values[-1] == "":
                    values.pop()
                if len(values) > len(columns) and "" in values:
                    values.remove("")
                rows.append(values)
    df = pd.DataFrame(rows, columns=columns)

    # cleaning
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace({"?": np.nan, "": np.nan, "nan": np.nan})
    for col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["appet"] == "no", "appet"] = np.nan   # invalid category -> missing
    df.loc[df["pe"] == "good", "pe"] = np.nan       # invalid category -> missing
    return df


def make_preprocessor():
    """Median-impute + scale numbers, mode-impute + one-hot categories.
    It lives INSIDE each model pipeline, so it is refit on every CV fold (no leakage)."""
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERICAL), ("cat", cat, CATEGORICAL)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip grid search")
    ap.add_argument("--patient", type=int, default=None, help="index (0-79) of a test patient to explain")
    ap.add_argument("--show", action="store_true", help="show plots in windows")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Data --------------------------------------------------------------
    df = load_ckd()
    X = df.drop(columns="class")
    y = df["class"].map({"ckd": 1, "notckd": 0})
    print(f"Data: {df.shape[0]} patients, {X.shape[1]} attributes | CKD={int(y.sum())}, not-CKD={int((y == 0).sum())}")
    print("Most-missing columns:", df.isnull().sum().sort_values(ascending=False).head(4).to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Split: train {len(X_train)} / test {len(X_test)} (stratified, random_state=42)\n")

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Models + hyper-parameter grids ------------------------------------
    specs = {
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__metric": ["euclidean", "manhattan"]},
                {"clf__n_neighbors": 9, "clf__metric": "manhattan"}),
        "Decision Tree": (DecisionTreeClassifier(random_state=42),
                          {"clf__max_depth": [2, 3, 5, 10, None], "clf__min_samples_leaf": [1, 2, 5, 10]},
                          {"clf__max_depth": 3, "clf__min_samples_leaf": 5}),
        "SVM (RBF)": (SVC(kernel="rbf", probability=True, random_state=42),
                      {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]},
                      {"clf__C": 0.1, "clf__gamma": "scale"}),
    }

    models = {}
    for name, (clf, grid, best) in specs.items():
        pipe = Pipeline([("preprocess", make_preprocessor()), ("clf", clf)])
        if args.quick:
            pipe.set_params(**best).fit(X_train, y_train)
            models[name] = pipe
            print(f"{name:14s} fitted with saved best params {best}")
        else:
            gs = GridSearchCV(pipe, grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
            gs.fit(X_train, y_train)
            models[name] = gs.best_estimator_
            print(f"{name:14s} best {gs.best_params_}  CV ROC-AUC = {gs.best_score_:.4f}")

    # 3. Test-set evaluation -----------------------------------------------
    rows = []
    for name, m in models.items():
        pred = m.predict(X_test)
        prob = m.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred),
                     "Recall": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred),
                     "ROC-AUC": roc_auc_score(y_test, prob),
                     "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    res = pd.DataFrame(rows)
    print("\nTEST SET (80 patients: 50 CKD, 30 not-CKD)")
    print(res.round(4).to_string(index=False))
    res.round(4).to_csv(OUT / "test_results.csv", index=False)

    # 4. ROC curves ----------------------------------------------------------
    plt.figure(figsize=(6.5, 5.5))
    for name, m in models.items():
        prob = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
    plt.title("ROC curves - held-out test set"); plt.legend(loc="lower right"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "roc_curves.png", dpi=150)

    # 5. Optional: explain one patient ------------------------------------
    if args.patient is not None:
        i = args.patient
        row = X_test.iloc[[i]]
        print(f"\nTest patient #{i} (true label: {'CKD' if y_test.iloc[i] == 1 else 'not CKD'})")
        print(row[["age", "bp", "sg", "al", "hemo", "sc", "htn", "dm"]].to_string(index=False))
        for name, m in models.items():
            p = m.predict_proba(row)[0, 1]
            print(f"  {name:14s} P(CKD) = {p:.3f} -> {'CKD' if p >= 0.5 else 'not CKD'}")

    print(f"\nSaved: {OUT / 'test_results.csv'} and roc_curves.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
