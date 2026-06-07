"""
kmeans_sweep.py
===============
Post-hoc K-sweep on the saved D-features from a v17_inline_kmex.py training run.

Addresses Phase 2 / E5: hyperparameter sensitivity for K (number of prototypes).
Also addresses the question that arose after our first K=16 training run:
"is K=16 the right cluster count for this dataset, or are some prototypes redundant?"

How it works:
    1. Loads embeddings.npy saved by v17_inline_kmex.py (shape: [N, d]).
    2. Runs K-Means with multiple random seeds for each K in {4, 8, 16, 32}.
    3. Reports intra-cluster mean distance, inter-prototype mean distance,
       silhouette coefficient, and cluster-size distribution per K.
    4. Picks the recommended K by the silhouette-based criterion.
    5. Writes structured JSON with all numbers.

Usage:
    python kmeans_sweep.py \\
        --embeddings proposed_inline_seed0/ksweep_data/embeddings.npy \\
        --k_values 4 8 16 32 \\
        --n_seeds 5 \\
        --out_json results/k_sweep.json

Output JSON:
    {
      "embeddings_shape": [N, d],
      "k_values": [4, 8, 16, 32],
      "n_seeds_per_k": 5,
      "per_k": {
        "4": {
          "intra_mean": float, "intra_std": float,
          "inter_mean": float, "inter_std": float,
          "silhouette_mean": float, "silhouette_std": float,
          "ratio_inter_over_intra_mean": float,
          "cluster_sizes_seed0": [n1, n2, n3, n4]
        },
        ...
      },
      "recommended_k": int,
      "recommendation_basis": "best_silhouette" | "best_ratio"
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_diagnostics(
    embeds: np.ndarray, K: int, seed: int,
) -> dict:
    """Run K-Means at given K and compute clustering diagnostics."""
    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(embeds)

    # Intra-cluster: mean pairwise distance within each cluster, averaged across K
    intra_per_k = []
    for k in range(K):
        members = embeds[km.labels_ == k]
        if len(members) > 1:
            intra_per_k.append(pdist(members).mean())
        else:
            intra_per_k.append(0.0)
    intra = float(np.mean(intra_per_k))

    # Inter-prototype: mean pairwise distance between cluster centers
    inter = float(pdist(km.cluster_centers_).mean()) if K > 1 else 0.0

    # Silhouette: standard clustering quality metric, in [-1, 1], higher = better
    try:
        sil = float(silhouette_score(embeds, km.labels_))
    except ValueError:
        # Happens if only one cluster has members; skip
        sil = float('nan')

    # Cluster sizes
    unique, counts = np.unique(km.labels_, return_counts=True)
    sizes = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "intra": intra,
        "inter": inter,
        "ratio_inter_over_intra": inter / intra if intra > 0 else float('inf'),
        "silhouette": sil,
        "inertia": float(km.inertia_),
        "cluster_sizes": sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True,
                        help="Path to embeddings.npy saved by v17_inline_kmex.py")
    parser.add_argument("--k_values", type=int, nargs='+', default=[4, 8, 16, 32])
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of K-Means random seeds per K.")
    parser.add_argument("--out_json", required=True,
                        help="Output path for structured results.")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings} ...")
    embeds = np.load(args.embeddings)
    print(f"  shape: {embeds.shape}")

    per_k = {}
    for K in args.k_values:
        print(f"\nK = {K}:")
        seed_results = []
        for s in range(args.n_seeds):
            d = kmeans_diagnostics(embeds, K=K, seed=s)
            seed_results.append(d)

        intras = np.array([r["intra"] for r in seed_results])
        inters = np.array([r["inter"] for r in seed_results])
        ratios = np.array([r["ratio_inter_over_intra"] for r in seed_results])
        sils   = np.array([r["silhouette"] for r in seed_results])

        per_k[str(K)] = {
            "intra_mean": float(intras.mean()), "intra_std": float(intras.std()),
            "inter_mean": float(inters.mean()), "inter_std": float(inters.std()),
            "ratio_inter_over_intra_mean": float(ratios.mean()),
            "silhouette_mean": float(sils.mean()),
            "silhouette_std": float(sils.std()),
            "cluster_sizes_seed0": seed_results[0]["cluster_sizes"],
        }
        print(f"  intra = {intras.mean():.4f} ± {intras.std():.4f}")
        print(f"  inter = {inters.mean():.4f} ± {inters.std():.4f}")
        print(f"  ratio = {ratios.mean():.4f} (higher = better separated)")
        print(f"  silhouette = {sils.mean():.4f} ± {sils.std():.4f}  "
              f"(higher = better; in [-1, 1])")
        print(f"  cluster sizes (seed=0): {seed_results[0]['cluster_sizes']}")

    # Recommendation: pick K with highest silhouette
    best_K = max(args.k_values, key=lambda k: per_k[str(k)]["silhouette_mean"])
    print(f"\nRecommended K: {best_K} "
          f"(silhouette = {per_k[str(best_K)]['silhouette_mean']:.4f})")

    results = {
        "embeddings_shape": list(embeds.shape),
        "k_values": args.k_values,
        "n_seeds_per_k": args.n_seeds,
        "per_k": per_k,
        "recommended_k": int(best_K),
        "recommendation_basis": "best_silhouette",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {args.out_json}")


if __name__ == "__main__":
    main()
