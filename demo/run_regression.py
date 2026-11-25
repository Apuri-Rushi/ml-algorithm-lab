"""
Experiment 2 - Regression on Graduate Admissions.
Linear regression by batch gradient descent (from scratch), polynomial degrees 1-4,
Ridge and Lasso tuned by 5-fold CV, and the underfit / good-fit / overfit comparison.

Combines regression/02, 03, 04, 06 notebooks into one script.

Run from the repository root:
    python demo/run_regression.py
    python demo/run_regression.py --lr 0.01 --epochs 2000   # try another learning rate live
    python demo/run_regression.py --show
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "raw" / "graduate_admissions" / "Admission_Predict.csv"
OUT = ROOT / "demo" / "outputs" / "regression"


# ---------- Gradient descent written by hand ----------
def predict(X, w, b):
    return X @ w + b


def gradient_descent(X, y, lr=0.05, epochs=5000):
    """Minimise J(w,b) = (1/n) * sum((Xw + b - y)^2)."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    history = []
    for _ in range(epochs):
        y_hat = predict(X, w, b)
        error = y_hat - y
        history.append(np.mean(error ** 2))          # MSE before the update
        dw = (2 / n) * X.T @ error                   # dJ/dw
        db = (2 / n) * error.sum()                   # dJ/db
        w -= lr * dw
        b -= lr * db
    return w, b, history


def metrics(y, y_hat):
    mse = mean_squared_error(y, y_hat)
    return mse, np.sqrt(mse), r2_score(y, y_hat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()             # "LOR " and "Chance of Admit " have trailing spaces
    df = df.drop(columns=["Serial No."])            # an ID, not a feature
    X = df.drop(columns=["Chance of Admit"])
    y = df["Chance of Admit"].to_numpy()
    print(f"Data: {df.shape[0]} rows, {X.shape[1]} features, no missing values = {not df.isnull().any().any()}")
    print("Correlation with target:", df.corr()["Chance of Admit"].drop("Chance of Admit")
          .sort_values(ascending=False).round(3).to_dict())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)              # fit on TRAIN only
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    # ---- 1. Linear regression with gradient descent ----
    with np.errstate(over="ignore", invalid="ignore"):
        w, b, hist = gradient_descent(Xs_tr, y_tr, lr=args.lr, epochs=args.epochs)
    print(f"\n[1] Gradient descent (lr={args.lr}, epochs={args.epochs})")
    rows = []
    if not np.isfinite(hist[-1]):
        first_bad = next(i for i, v in enumerate(hist) if not np.isfinite(v) or v > 1e6)
        print(f"    DIVERGED: MSE exploded by epoch {first_bad}. Learning rate too large "
              f"(stable limit for this data is about 0.206).")
    else:
        print(f"    training MSE {hist[0]:.4f} -> {hist[-1]:.5f}")
        print("    weights:", {c: round(float(v), 4) for c, v in zip(X.columns, w)}, " bias:", round(float(b), 5))
        mse, rmse, r2 = metrics(y_te, predict(Xs_te, w, b))
        print(f"    TEST  MSE={mse:.5f}  RMSE={rmse:.5f}  R2={r2:.5f}")
        rows.append({"Model": "Linear (GD)", "Train MSE": hist[-1], "Test MSE": mse, "Test RMSE": rmse, "Test R2": r2})

    plt.figure(figsize=(6, 4)); plt.plot(np.clip(np.nan_to_num(hist, nan=1e6), None, 1e6)); plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Training MSE (log)"); plt.title("Gradient descent convergence")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUT / "gd_convergence.png", dpi=150)

    # ---- 2. Polynomial regression ----
    print("\n[2] Polynomial regression")
    for d in [1, 2, 3, 4]:
        model = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),
                          ("scaler", StandardScaler()),
                          ("reg", LinearRegression())]).fit(Xs_tr, y_tr)
        n_feat = model.named_steps["poly"].n_output_features_
        tr_mse = mean_squared_error(y_tr, model.predict(Xs_tr))
        mse, rmse, r2 = metrics(y_te, model.predict(Xs_te))
        print(f"    degree {d}: {n_feat:3d} features  train MSE={tr_mse:.5f}  test MSE={mse:.5f}  test R2={r2:.5f}")
        rows.append({"Model": f"Poly d={d}", "Train MSE": tr_mse, "Test MSE": mse, "Test RMSE": rmse, "Test R2": r2})

    # ---- 3. Ridge and Lasso, alpha chosen by 5-fold CV ----
    print("\n[3] Ridge / Lasso (20 alphas from 1e-4 to 1e2, 5-fold CV)")
    alphas = np.logspace(-4, 2, 20)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    coefs = {}
    for name, est in [("Ridge", Ridge()), ("Lasso", Lasso(max_iter=100000))]:
        gs = GridSearchCV(Pipeline([("scaler", StandardScaler()), ("model", est)]),
                          {"model__alpha": alphas}, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        tr_mse = mean_squared_error(y_tr, best.predict(X_tr))
        mse, rmse, r2 = metrics(y_te, best.predict(X_te))
        a = gs.best_params_["model__alpha"]
        print(f"    {name}: alpha={a:.6g}  CV MSE={-gs.best_score_:.6f}  test MSE={mse:.5f}  R2={r2:.5f}")
        coefs[name] = best.named_steps["model"].coef_
        rows.append({"Model": name, "Train MSE": tr_mse, "Test MSE": mse, "Test RMSE": rmse, "Test R2": r2})
    print("    coefficients:\n" + pd.DataFrame(coefs, index=X.columns).round(5).to_string())

    # ---- 4. Underfit / good / overfit ----
    print("\n[4] Fit behaviour")
    under = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1e6))]).fit(X_tr, y_tr)
    for label, m in [("Underfit  Ridge alpha=1e6", under)]:
        print(f"    {label}: train R2={r2_score(y_tr, m.predict(X_tr)):.4f}  test R2={r2_score(y_te, m.predict(X_te)):.4f}")
    print("    Good fit  poly d=2 and Overfit poly d=4: see table above")

    table = pd.DataFrame(rows).sort_values("Test MSE")
    print("\nRANKING BY TEST MSE (lower is better)")
    print(table.round(5).to_string(index=False))
    table.round(5).to_csv(OUT / "regression_results.csv", index=False)
    print(f"\nSaved: {OUT / 'regression_results.csv'} and gd_convergence.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
