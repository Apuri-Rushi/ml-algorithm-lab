"""
Local model server for the project website.

Runs the real Python models (scikit-learn, NumPy, TensorFlow) on request and returns
metrics plus a freshly drawn graph. Uses only the Python standard library for the web
server, and the same functions as the demo/ scripts, so results match the report.

Start it from the repository root or the website folder:
    python website/server.py          (listens on http://127.0.0.1:8000)
Then, in a second terminal:
    cd website && npm run dev
"""
import base64
import io
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from run_classification import load_ckd, make_preprocessor      # demo/run_classification.py
from run_kmeans import DATA as BIKE_CSV, FEATURES as BIKE_FEATURES, kmeans  # demo/run_kmeans.py
from run_regression import DATA as ADMIT_CSV, gradient_descent   # demo/run_regression.py

PORT = 8000
PLOT_LOCK = threading.Lock()          # matplotlib is not thread-safe
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#555", "axes.grid": True,
                     "grid.alpha": .3, "figure.facecolor": "white"})


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def r(v, n=4):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


# --------------------------------------------------------------------------- data (loaded once)
class Data:
    _lock = threading.Lock()
    ckd = admit = bike = iris = None

    @classmethod
    def get_ckd(cls):
        with cls._lock:
            if cls.ckd is None:
                df = load_ckd()
                X = df.drop(columns="class")
                y = df["class"].map({"ckd": 1, "notckd": 0})
                cls.ckd = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            return cls.ckd

    @classmethod
    def get_admit(cls):
        with cls._lock:
            if cls.admit is None:
                df = pd.read_csv(ADMIT_CSV)
                df.columns = df.columns.str.strip()
                df = df.drop(columns=["Serial No."])
                X = df.drop(columns=["Chance of Admit"])
                y = df["Chance of Admit"].to_numpy()
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                cls.admit = (X_tr, X_te, y_tr, y_te, list(X.columns))
            return cls.admit

    @classmethod
    def get_bike(cls):
        with cls._lock:
            if cls.bike is None:
                df = pd.read_csv(BIKE_CSV)
                df["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
                df["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
                X = df[BIKE_FEATURES]
                cls.bike = (df, ((X - X.mean()) / X.std()).to_numpy())
            return cls.bike


# --------------------------------------------------------------------------- Q1 classification
GRIDS = {
    "knn": {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__metric": ["euclidean", "manhattan"]},
    "tree": {"clf__max_depth": [2, 3, 5, 10, None], "clf__min_samples_leaf": [1, 2, 5, 10]},
    "svm": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]},
}
NAMES = {"knn": "KNN", "tree": "Decision Tree", "svm": "SVM (RBF)", "cnn": "1D CNN"}


def build_classifier(model, p):
    if model == "knn":
        clf = KNeighborsClassifier(n_neighbors=int(p.get("k", 9)), metric=p.get("metric", "manhattan"))
    elif model == "tree":
        depth = p.get("depth", 3)
        clf = DecisionTreeClassifier(max_depth=None if depth in (None, "None", "") else int(depth),
                                     min_samples_leaf=int(p.get("leaf", 5)), random_state=42)
    elif model == "svm":
        g = p.get("gamma", "scale")
        clf = SVC(kernel="rbf", C=float(p.get("C", 0.1)), gamma=g if g == "scale" else float(g),
                  probability=True, random_state=42)
    else:
        raise ValueError(f"unknown model {model}")
    return Pipeline([("preprocess", make_preprocessor()), ("clf", clf)])


def run_cnn(X_tr_full, y_tr_full, X_te):
    import os, random
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    random.seed(42); np.random.seed(42); tf.random.set_seed(42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr_full, y_tr_full, test_size=0.2,
                                                random_state=42, stratify=y_tr_full)
    pre = make_preprocessor().fit(X_tr)

    def to3d(A):
        M = pre.transform(A)
        M = M.toarray() if hasattr(M, "toarray") else M
        return np.asarray(M, dtype="float32")[..., None]

    A_tr, A_val, A_te = to3d(X_tr), to3d(X_val), to3d(X_te)
    L = tf.keras.layers
    m = tf.keras.Sequential([
        L.Input(shape=A_tr.shape[1:]),
        L.Conv1D(16, 3, activation="relu", padding="same"), L.MaxPooling1D(2), L.Dropout(0.25),
        L.Conv1D(32, 3, activation="relu", padding="same"), L.MaxPooling1D(2), L.Dropout(0.25),
        L.Flatten(), L.Dense(16, activation="relu"), L.Dropout(0.30), L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    h = m.fit(A_tr, y_tr, validation_data=(A_val, y_val), epochs=100, batch_size=16, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                          restore_best_weights=True)])
    prob = m.predict(A_te, verbose=0).ravel()
    info = {"epochs_run": len(h.history["loss"]),
            "best_epoch": int(np.argmin(h.history["val_loss"])) + 1,
            "parameters": int(m.count_params())}
    return prob, info


def q1(p):
    X_tr, X_te, y_tr, y_te = Data.get_ckd()
    model = p.get("model", "svm")
    if model not in NAMES:
        raise ValueError(f"Unknown model '{model}'. Choose knn, tree, svm or cnn.")
    out = {"model": NAMES[model], "notes": []}

    if model == "cnn":
        prob, info = run_cnn(X_tr, y_tr, X_te)
        out["notes"].append(f"Trained {info['epochs_run']} epochs; best validation loss at epoch "
                            f"{info['best_epoch']}; {info['parameters']:,} parameters. "
                            "Neural-network runs can vary slightly between machines.")
        fitted = None
    else:
        pipe = build_classifier(model, p)
        if p.get("tune"):
            gs = GridSearchCV(pipe, GRIDS[model], cv=StratifiedKFold(5, shuffle=True, random_state=42),
                              scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
            fitted = gs.best_estimator_
            best = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
            out["tuning"] = {"best_params": best, "cv_auc": r(gs.best_score_)}
        else:
            fitted = pipe.fit(X_tr, y_tr)
        prob = fitted.predict_proba(X_te)[:, 1]
        out["train_accuracy"] = r(accuracy_score(y_tr, fitted.predict(X_tr)))

    # Same as the original code: sklearn models use predict() (for the SVM this is the decision
    # boundary, which can differ from thresholding its Platt-calibrated probability at 0.5);
    # the CNN thresholds its sigmoid output at 0.5.
    pred = fitted.predict(X_te) if fitted is not None else (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    out["metrics"] = {"Accuracy": r(accuracy_score(y_te, pred)),
                      "Precision": r(precision_score(y_te, pred, zero_division=0)),
                      "Recall": r(recall_score(y_te, pred)), "F1": r(f1_score(y_te, pred)),
                      "ROC-AUC": r(roc_auc_score(y_te, prob))}
    out["confusion"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

    pi = p.get("patient")
    if pi not in (None, "") and fitted is not None:
        i = max(0, min(int(pi), len(X_te) - 1))
        row = X_te.iloc[[i]]
        pr = float(fitted.predict_proba(row)[0, 1])
        lab = int(fitted.predict(row)[0])
        cols = ["age", "bp", "sg", "al", "hemo", "sc", "htn", "dm"]
        out["patient"] = {"index": i, "true": "CKD" if y_te.iloc[i] == 1 else "not CKD",
                          "p_ckd": round(pr, 3), "predicted": "CKD" if lab == 1 else "not CKD",
                          "features": {c: (None if pd.isna(row[c].iloc[0]) else
                                           (row[c].iloc[0] if isinstance(row[c].iloc[0], str) else float(row[c].iloc[0])))
                                       for c in cols}}

    with PLOT_LOCK:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax[0].plot(fpr, tpr, lw=2, color="#2d3136", label=f"AUC = {roc_auc_score(y_te, prob):.4f}")
        ax[0].plot([0, 1], [0, 1], "--", color="#999", label="Random")
        ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
        ax[0].set_title(f"ROC curve — {NAMES[model]}"); ax[0].legend(loc="lower right")
        cm = np.array([[tn, fp], [fn, tp]])
        ax[1].imshow(cm, cmap="Greys"); ax[1].grid(False)
        for (a, b), v in np.ndenumerate(cm):
            ax[1].text(b, a, str(v), ha="center", va="center", fontsize=16,
                       color="white" if v > cm.max() / 2 else "black")
        ax[1].set_xticks([0, 1], ["Not CKD", "CKD"]); ax[1].set_yticks([0, 1], ["Not CKD", "CKD"])
        ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual"); ax[1].set_title("Confusion matrix (80 test patients)")
        out["image"] = fig_to_b64(fig)
    return out


# --------------------------------------------------------------------------- Q2 regression
def reg_metrics(y, yh):
    mse = mean_squared_error(y, yh)
    return {"MSE": r(mse, 5), "RMSE": r(np.sqrt(mse), 5), "R²": r(r2_score(y, yh), 5)}


def q2(p):
    X_tr, X_te, y_tr, y_te, cols = Data.get_admit()
    mode = p.get("mode", "gd")
    out = {"notes": []}
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    if mode == "gd":
        lr, epochs = float(p.get("lr", 0.05)), int(p.get("epochs", 5000))
        epochs = max(1, min(epochs, 50000))
        with np.errstate(over="ignore", invalid="ignore"):
            w, b, hist = gradient_descent(Xs_tr, y_tr, lr=lr, epochs=epochs)
        hist = np.array(hist)
        out["model"] = f"Linear regression · gradient descent (η = {lr}, {epochs:,} epochs)"
        diverged = not np.isfinite(hist[-1]) or hist[-1] > 1e6
        if diverged:
            bad = int(np.argmax(~np.isfinite(hist) | (hist > 1e6)))
            out["diverged"] = True
            out["notes"].append(f"Diverged: the MSE exploded by epoch {bad}. The learning rate is too large "
                                "(the stable limit for this data is about 0.206).")
        else:
            out["train"] = reg_metrics(y_tr, Xs_tr @ w + b)
            out["test"] = reg_metrics(y_te, Xs_te @ w + b)
            out["weights"] = {c: r(v, 4) for c, v in zip(cols, w)} | {"bias": r(b, 5)}
            out["notes"].append(f"Training MSE fell from {hist[0]:.4f} to {hist[-1]:.5f}.")
        with PLOT_LOCK:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.plot(np.clip(np.nan_to_num(hist, nan=1e6, posinf=1e6), 1e-6, 1e6), color="#2d3136")
            ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Training MSE (log scale)")
            ax.set_title("Gradient descent convergence" + (" — diverged" if diverged else ""))
            out["image"] = fig_to_b64(fig)
        return out

    if mode == "poly":
        d = max(1, min(int(p.get("degree", 2)), 5))
        model = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),
                          ("scaler", StandardScaler()), ("reg", LinearRegression())]).fit(Xs_tr, y_tr)
        out["model"] = f"Polynomial regression · degree {d}"
        out["notes"].append(f"{model.named_steps['poly'].n_output_features_} features for {len(y_tr)} training rows.")
        Xa, Xb = Xs_tr, Xs_te
    else:  # ridge / lasso
        est = Ridge() if mode == "ridge" else Lasso(max_iter=100000)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        a = p.get("alpha", "cv")
        if a in ("cv", "", None):
            gs = GridSearchCV(pipe, {"model__alpha": np.logspace(-4, 2, 20)},
                              cv=KFold(5, shuffle=True, random_state=42),
                              scoring="neg_mean_squared_error", n_jobs=-1).fit(X_tr, y_tr)
            model, alpha = gs.best_estimator_, gs.best_params_["model__alpha"]
            out["notes"].append(f"α chosen by 5-fold CV: {alpha:.6g} (CV MSE {-gs.best_score_:.6f}).")
        else:
            alpha = float(a)
            model = pipe.set_params(model__alpha=alpha).fit(X_tr, y_tr)
        out["model"] = f"{'Ridge (L2)' if mode == 'ridge' else 'Lasso (L1)'} · α = {alpha:.6g}"
        coef = model.named_steps["model"].coef_
        out["weights"] = {c: r(v, 5) for c, v in zip(cols, coef)}
        zeros = [c for c, v in zip(cols, coef) if abs(v) < 1e-12]
        if zeros:
            out["notes"].append("Coefficients set exactly to zero: " + ", ".join(zeros) + ".")
        Xa, Xb = X_tr, X_te

    yh_tr, yh_te = model.predict(Xa), model.predict(Xb)
    out["train"], out["test"] = reg_metrics(y_tr, yh_tr), reg_metrics(y_te, yh_te)
    with PLOT_LOCK:
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        ax.scatter(y_te, yh_te, s=18, color="#2d3136", alpha=.75)
        lo, hi = y_te.min(), y_te.max()
        ax.plot([lo, hi], [lo, hi], "--", color="#999")
        ax.set_xlabel("Actual chance of admit"); ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs predicted — test R² {out['test']['R²']}")
        out["image"] = fig_to_b64(fig)
    return out


# --------------------------------------------------------------------------- Q3 K-Means
def q3(p):
    df, Xs = Data.get_bike()
    k = max(2, min(int(p.get("k", 4)), 8))
    t = time.time()
    labels, cents, wcss = kmeans(Xs, k)
    g = df.assign(cluster=labels).groupby("cluster")
    prof = pd.DataFrame({"hours": g.size(), "temp": g["temp"].mean(), "hum": g["hum"].mean(),
                         "wind": g["windspeed"].mean(), "rentals": g["cnt"].mean()})
    rows = [{"cluster": int(i), "hours": int(rw.hours), "share": round(rw.hours / len(df) * 100, 1),
             "temp": round(rw.temp, 2), "hum": round(rw.hum, 2), "wind": round(rw.wind, 2),
             "rentals": round(rw.rentals, 1)} for i, rw in prof.iterrows()]
    out = {"model": f"K-Means from scratch · K = {k}", "wcss": round(float(wcss), 2), "clusters": rows,
           "notes": [f"10 random restarts, lowest WCSS kept. Finished in {time.time() - t:.1f} s. "
                     "Rentals per hour were computed afterwards; demand was not an input."]}
    with PLOT_LOCK:
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
        cmap = plt.get_cmap("tab10")
        for c in range(k):
            m = labels == c
            ax[0].scatter(Xs[m, 2], Xs[m, 3], s=3, alpha=.4, color=cmap(c), label=f"Cluster {c}")
        ax[0].scatter(cents[:, 2], cents[:, 3], marker="x", s=120, color="black", label="Centroids")
        ax[0].set_xlabel("Temperature (standardised)"); ax[0].set_ylabel("Humidity (standardised)")
        ax[0].set_title("Clusters (2 of 5 dimensions)"); ax[0].legend(markerscale=3, fontsize=8)
        ax[1].bar([str(rw["cluster"]) for rw in rows], [rw["rentals"] for rw in rows],
                  color=[cmap(rw["cluster"]) for rw in rows])
        ax[1].set_xlabel("Cluster"); ax[1].set_ylabel("Mean rentals per hour"); ax[1].set_title("Demand by cluster (post-hoc)")
        out["image"] = fig_to_b64(fig)
    return out


# --------------------------------------------------------------------------- Q4 PCA
def q4(p):
    iris = load_iris()
    X, y = iris.data, iris.target
    n = max(1, min(int(p.get("n", 2)), 4))
    Z = StandardScaler().fit_transform(X)
    S = np.cov(Z, rowvar=False)
    vals, vecs = np.linalg.eigh(S)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    evr = vals / vals.sum()
    W = vecs[:, :n]
    Zp = Z @ W
    recon = Zp @ W.T
    sk = PCA().fit(Z)
    same = bool(np.allclose(sk.explained_variance_ratio_, evr) and np.allclose(np.abs(sk.components_.T), np.abs(vecs)))
    out = {"model": f"PCA by eigen-decomposition · {n} component{'s' if n > 1 else ''}",
           "components": [{"pc": f"PC{i+1}", "eigenvalue": r(vals[i]), "variance": round(evr[i] * 100, 2),
                           "cumulative": round(evr[:i+1].sum() * 100, 2), "kept": i < n} for i in range(4)],
           "retained": round(evr[:n].sum() * 100, 2), "shape": f"150 × {n}",
           "reconstruction_mse": r(np.mean((Z - recon) ** 2), 5), "matches_sklearn": same,
           "loadings": {f: [r(v, 5) for v in vecs[j, :n]] for j, f in enumerate(iris.feature_names)},
           "notes": ["Matches sklearn PCA (up to sign): " + ("yes" if same else "no") + "."]}
    with PLOT_LOCK:
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i, name in enumerate(iris.target_names):
            m = y == i
            if n >= 2:
                ax[0].scatter(Zp[m, 0], Zp[m, 1], s=22, color=colors[i], edgecolor="k", lw=.3, label=name)
            else:
                ax[0].scatter(Zp[m, 0], np.random.default_rng(i).uniform(-.3, .3, m.sum()) + i,
                              s=22, color=colors[i], edgecolor="k", lw=.3, label=name)
        ax[0].set_xlabel(f"PC1 ({evr[0]*100:.1f} %)")
        ax[0].set_ylabel(f"PC2 ({evr[1]*100:.1f} %)" if n >= 2 else "species (jittered)")
        ax[0].set_title("Projection" + (" onto PC1 and PC2" if n >= 2 else " onto PC1 only"))
        ax[0].legend(fontsize=8)
        bars = ax[1].bar([f"PC{i+1}" for i in range(4)], evr * 100,
                         color=["#2d3136" if i < n else "#c8ccd0" for i in range(4)])
        ax[1].plot(range(4), np.cumsum(evr) * 100, "o-", color="#777", label="Cumulative")
        ax[1].set_ylabel("Explained variance (%)"); ax[1].set_title(f"Kept: {evr[:n].sum()*100:.2f} % of variance")
        ax[1].legend(fontsize=8)
        out["image"] = fig_to_b64(fig)
    return out


ROUTES = {"/api/q1": q1, "/api/q2": q2, "/api/q3": q3, "/api/q4": q4}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        fn = ROUTES.get(self.path)
        if fn is None:
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(n) or b"{}")
            t = time.time()
            result = fn(params)
            result["seconds"] = round(time.time() - t, 1)
            self._send(200, result)
        except Exception as e:  # report the error to the page instead of crashing the server
            traceback.print_exc()
            self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % args}\n")


if __name__ == "__main__":
    print("Loading datasets ...")
    Data.get_ckd(); Data.get_admit(); Data.get_bike()
    print(f"Model server ready on http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
