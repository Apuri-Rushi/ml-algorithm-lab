"""
Experiment 3 - K-Means clustering from scratch (NumPy only) on Bike Sharing hourly data.

Same algorithm as unsupervised/K-means/03_kmeans_from_scratch.ipynb.

Run from the repository root:
    python demo/run_kmeans.py            # elbow for K = 2..6, then final K = 4
    python demo/run_kmeans.py --k 3      # try a different final K live
    python demo/run_kmeans.py --show
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "raw" / "bike_sharing" / "hour.csv"
OUT = ROOT / "demo" / "outputs" / "kmeans"
FEATURES = ["hour_sin", "hour_cos", "temp", "hum", "windspeed"]


def kmeans(X, k, max_iters=100, n_init=10, random_state=42):
    """Lloyd's algorithm with n_init random restarts; keeps the run with lowest WCSS."""
    rng = np.random.default_rng(random_state)
    best = (np.inf, None, None)
    for _ in range(n_init):
        centroids = X[rng.choice(len(X), size=k, replace=False)].copy()   # random init
        for _ in range(max_iters):
            # assignment step: squared distance of every point to every centroid -> (n, k)
            dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)
            # update step: centroid = mean of its points (re-seed if a cluster empties)
            new = np.array([X[labels == c].mean(axis=0) if np.any(labels == c)
                            else X[rng.integers(len(X))] for c in range(k)])
            converged = np.allclose(centroids, new)
            centroids = new
            if converged:
                break
        wcss = ((X - centroids[labels]) ** 2).sum()
        if wcss < best[0]:
            best = (wcss, labels.copy(), centroids.copy())
    return best[1], best[2], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    # hour on a circle so that 23:00 is next to 00:00
    df["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    X = df[FEATURES]
    Xs = ((X - X.mean()) / X.std()).to_numpy()       # pandas std = sample std (n-1)
    print(f"Data: {len(df)} hourly records, features = {FEATURES}")

    # ---- Elbow ----
    ks, wcss = [2, 3, 4, 5, 6], []
    for k in ks:
        _, _, w = kmeans(Xs, k)
        wcss.append(w)
    print("\nK   WCSS        drop")
    for i, (k, w) in enumerate(zip(ks, wcss)):
        drop = "" if i == 0 else f"{(wcss[i-1] - w) / wcss[i-1] * 100:5.1f}%"
        print(f"{k}   {w:10.2f}  {drop}")
    plt.figure(figsize=(6, 4)); plt.plot(ks, wcss, "o-")
    plt.xlabel("K"); plt.ylabel("WCSS"); plt.title("Elbow method"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "elbow.png", dpi=150)

    # ---- Final model ----
    labels, cents, w = kmeans(Xs, args.k)
    df["cluster"] = labels
    prof = df.groupby("cluster").agg(hours=("cnt", "size"), temp=("temp", "mean"),
                                     hum=("hum", "mean"), wind=("windspeed", "mean"),
                                     rentals_per_hour=("cnt", "mean"))
    prof["share_%"] = (prof["hours"] / len(df) * 100).round(1)
    print(f"\nFinal K = {args.k}, WCSS = {w:.2f}")
    print("Cluster profile (rentals were NOT an input - computed afterwards):")
    print(prof.round(3).to_string())
    prof.round(3).to_csv(OUT / f"cluster_profile_k{args.k}.csv")

    plt.figure(figsize=(6.5, 5))
    plt.scatter(Xs[:, 2], Xs[:, 3], c=labels, s=4, alpha=.5, cmap="viridis")
    plt.scatter(cents[:, 2], cents[:, 3], c="red", marker="x", s=120, label="centroids")
    plt.xlabel("temp (standardised)"); plt.ylabel("humidity (standardised)")
    plt.title(f"K-Means, K = {args.k} (2 of 5 dimensions shown)"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / f"clusters_k{args.k}.png", dpi=150)
    print(f"\nSaved: {OUT}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
