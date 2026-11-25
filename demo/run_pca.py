"""
Experiment 4 - PCA by covariance eigen-decomposition (no PCA estimator) on Iris.

Local version of unsupervised/PCA/pca (2).py (the original mounts Google Drive),
plus a check against sklearn.decomposition.PCA to prove the manual version is right.

Run from the repository root:
    python demo/run_pca.py
    python demo/run_pca.py --show
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "outputs" / "pca"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X, y, names = iris.data, iris.target, iris.target_names
    print(f"Data: {X.shape[0]} flowers x {X.shape[1]} measurements: {iris.feature_names}")

    # 1. standardise
    Z = StandardScaler().fit_transform(X)
    # 2. covariance matrix  S = Z^T Z / (n-1)
    S = np.cov(Z, rowvar=False)
    print("\nCovariance matrix:\n", np.round(S, 5))
    # 3. eigen-decomposition (eigh: S is symmetric), sort descending
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    evr = vals / vals.sum()
    print("\nEigenvalues:            ", np.round(vals, 4))
    print("Explained variance ratio:", np.round(evr * 100, 2), "%")
    print("Cumulative:              ", np.round(np.cumsum(evr) * 100, 2), "%")
    print("\nEigenvectors (columns = PC1..PC4):\n",
          pd.DataFrame(vecs, index=iris.feature_names, columns=["PC1", "PC2", "PC3", "PC4"]).round(5))
    # 4. project
    Z2, Z3 = Z @ vecs[:, :2], Z @ vecs[:, :3]
    print(f"\nProjected shapes: {Z2.shape} (keeps {np.cumsum(evr)[1]*100:.2f}%), "
          f"{Z3.shape} (keeps {np.cumsum(evr)[2]*100:.2f}%)")

    # Sanity check against scikit-learn (signs of components may flip - that's fine)
    sk = PCA().fit(Z)
    same = np.allclose(sk.explained_variance_ratio_, evr) and np.allclose(np.abs(sk.components_.T), np.abs(vecs))
    print("Matches sklearn PCA (up to sign):", same)

    plt.figure(figsize=(6.5, 5))
    for i, n in enumerate(names):
        plt.scatter(Z2[y == i, 0], Z2[y == i, 1], label=n, edgecolor="k", alpha=.8)
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)"); plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    plt.title("Iris projected on first two PCs"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "pca_scatter.png", dpi=150)

    plt.figure(figsize=(6, 4)); plt.plot(range(1, 5), np.cumsum(evr), "o-")
    plt.axhline(.95, ls="--", c="gray"); plt.xticks(range(1, 5))
    plt.xlabel("Components"); plt.ylabel("Cumulative explained variance"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "cumulative_variance.png", dpi=150)
    print(f"Saved: {OUT}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
