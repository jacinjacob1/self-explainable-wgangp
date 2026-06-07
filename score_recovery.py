"""
score_recovery.py
=================
Reverse-engineer the `score` column in `train_h5.csv` by trying several plausible
tumor-prominence metrics on the corresponding .h5 files and reporting which one
matches the recorded scores to high numerical precision.

Why: Reviewer 2 flagged "limited dataset slicing from a narrow range" as a concern.
We need to document the exact rule that selected your 1107 slices from the 369-volume
BraTS2020 pool. The CSV preserves the scores but not the formula. This script finds
the formula.

Candidate rules tried:
    1. total_tumor_voxels      = sum of all mask-positive pixels across 3 classes
    2. nonzero_image_voxels    = sum of image-positive voxels (brain region)
    3. brain_area              = number of pixels with image intensity > 0 (any modality)
    4. tumor_modality0_intensity = sum of FLAIR intensity inside tumor mask
    5. brain_modality0_intensity = sum of FLAIR intensity over the brain region
    6. mean_modality0_brain    = mean FLAIR intensity over brain (image > 0)
    7. image_sum_modality0     = sum of FLAIR over ALL pixels (incl. zero background)

For each rule we compute the metric on every .h5 file referenced in the CSV,
then run a correlation + ratio test against the recorded score. A match is declared
if Pearson correlation > 0.9999 AND the per-file ratio (score / metric) is
near-constant (std/mean < 1e-4).

Usage:
    python score_recovery.py --csv h5_train/train_h5.csv --h5_dir h5_train

Outputs:
    Prints a ranked report of all candidates with their correlation and
    constant-ratio statistics. Writes a JSON report to score_recovery_report.json.

The script tolerates the path mismatch in the CSV (recorded paths point to a Mac
user's Downloads folder; this script substitutes h5_dir as the actual location).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Candidate scoring rules
# Each rule takes (image, mask) numpy arrays and returns a single scalar.
#   image: (H, W, C_image)  -- usually C_image=4 (FLAIR, T1, T1Gd, T2)
#   mask:  (H, W, C_mask)   -- usually C_mask=3 (one-hot WT/ET/TC)
# ---------------------------------------------------------------------------

def total_tumor_voxels(image: np.ndarray, mask: np.ndarray) -> float:
    """Sum of all positive mask pixels across the 3 tumor classes."""
    return float(mask.sum())


def nonzero_image_voxels(image: np.ndarray, mask: np.ndarray) -> float:
    """Sum of all positive image voxels (proxy for total tissue signal)."""
    return float((image > 0).sum())


def brain_area(image: np.ndarray, mask: np.ndarray) -> float:
    """Number of pixels where ANY modality has positive intensity."""
    return float(((image > 0).any(axis=-1)).sum())


def tumor_modality0_intensity(image: np.ndarray, mask: np.ndarray) -> float:
    """Sum of FLAIR (channel 0) intensity inside the union of tumor classes."""
    tumor_any = (mask.sum(axis=-1) > 0)
    return float(image[..., 0][tumor_any].sum())


def brain_modality0_intensity(image: np.ndarray, mask: np.ndarray) -> float:
    """Sum of FLAIR (channel 0) intensity over the brain region (image > 0)."""
    brain = (image[..., 0] > 0) if image.ndim == 3 else (image > 0)
    return float(image[..., 0][brain].sum()) if image.ndim == 3 \
        else float(image[brain].sum())


def mean_modality0_brain(image: np.ndarray, mask: np.ndarray) -> float:
    """Mean FLAIR intensity over the brain region."""
    brain = (image[..., 0] > 0) if image.ndim == 3 else (image > 0)
    sel = image[..., 0][brain] if image.ndim == 3 else image[brain]
    return float(sel.mean()) if sel.size > 0 else 0.0


def image_sum_modality0(image: np.ndarray, mask: np.ndarray) -> float:
    """Total sum of FLAIR (channel 0) over the entire image (incl. background)."""
    return float(image[..., 0].sum()) if image.ndim == 3 else float(image.sum())


def total_image_sum(image: np.ndarray, mask: np.ndarray) -> float:
    """Total sum of all image channels."""
    return float(image.sum())


def nonneg_image_sum(image: np.ndarray, mask: np.ndarray) -> float:
    """Sum of positive image values only (handles z-normalized data with negatives)."""
    return float(image[image > 0].sum())


CANDIDATES = {
    "total_tumor_voxels": total_tumor_voxels,
    "nonzero_image_voxels": nonzero_image_voxels,
    "brain_area": brain_area,
    "tumor_modality0_intensity": tumor_modality0_intensity,
    "brain_modality0_intensity": brain_modality0_intensity,
    "mean_modality0_brain": mean_modality0_brain,
    "image_sum_modality0": image_sum_modality0,
    "total_image_sum": total_image_sum,
    "nonneg_image_sum": nonneg_image_sum,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load image and mask arrays from a BraTS .h5 file."""
    with h5py.File(path, "r") as f:
        image = np.array(f["image"])
        mask = np.array(f["mask"])
    return image, mask


def evaluate_candidates(csv_path: Path, h5_dir: Path) -> dict:
    """Try every candidate rule and report which one best matches the CSV scores."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # The CSV paths point at a Mac user's Downloads folder; we substitute h5_dir.
    df["local_path"] = df["filename"].apply(lambda fn: h5_dir / fn)

    missing = ~df["local_path"].apply(lambda p: p.exists())
    if missing.any():
        print(f"WARNING: {missing.sum()} files missing under {h5_dir}; "
              f"will be skipped.")
        df = df[~missing].reset_index(drop=True)

    print(f"Computing {len(CANDIDATES)} candidate metrics on {len(df)} files...")
    # Pre-compute metrics for all files (single pass through h5 reads)
    results = {name: [] for name in CANDIDATES}
    recorded_scores = []

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"  {i}/{len(df)}")
        image, mask = load_h5(row["local_path"])
        recorded_scores.append(row["score"])
        for name, fn in CANDIDATES.items():
            results[name].append(fn(image, mask))

    recorded = np.array(recorded_scores)
    report = []
    for name, vals in results.items():
        vals = np.array(vals)
        # Skip degenerate metrics (constant or zero)
        if vals.std() < 1e-12 or vals.sum() == 0:
            report.append({
                "rule": name,
                "pearson": float("nan"),
                "ratio_mean": float("nan"),
                "ratio_std": float("nan"),
                "ratio_cv": float("nan"),
                "match": False,
                "note": "degenerate (constant or all zero)",
            })
            continue
        # Pearson correlation
        pearson = float(np.corrcoef(vals, recorded)[0, 1])
        # Constant-ratio test: if score = c * metric, then std(score/metric) ≈ 0
        nonzero = vals > 0
        if not nonzero.any():
            ratio_mean = ratio_std = ratio_cv = float("nan")
            match = False
            note = "all values zero"
        else:
            ratios = recorded[nonzero] / vals[nonzero]
            ratio_mean = float(ratios.mean())
            ratio_std = float(ratios.std())
            ratio_cv = ratio_std / abs(ratio_mean) if ratio_mean != 0 else float("inf")
            match = (pearson > 0.9999) and (ratio_cv < 1e-4)
            note = "MATCH" if match else ""
        report.append({
            "rule": name,
            "pearson": pearson,
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "ratio_cv": ratio_cv,
            "match": match,
            "note": note,
        })

    report.sort(key=lambda r: (-r["pearson"] if not np.isnan(r["pearson"]) else 1.0,
                               r["ratio_cv"] if not np.isnan(r["ratio_cv"]) else float("inf")))
    return {"recorded_count": len(df), "candidates": report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="h5_train/train_h5.csv")
    parser.add_argument("--h5_dir", type=str, default="h5_train")
    parser.add_argument("--out", type=str, default="score_recovery_report.json")
    args = parser.parse_args()

    report = evaluate_candidates(Path(args.csv), Path(args.h5_dir))

    print()
    print("=" * 80)
    print(f"{'rule':32s}  {'pearson':>9s}  {'ratio_mean':>12s}  {'cv':>10s}  match")
    print("-" * 80)
    for r in report["candidates"]:
        pearson_str = f"{r['pearson']:.6f}" if not np.isnan(r["pearson"]) else "    nan"
        mean_str = f"{r['ratio_mean']:.4e}" if not np.isnan(r["ratio_mean"]) else "        nan"
        cv_str = f"{r['ratio_cv']:.2e}" if not np.isnan(r["ratio_cv"]) else "      nan"
        flag = "MATCH" if r["match"] else r["note"]
        print(f"{r['rule']:32s}  {pearson_str:>9s}  {mean_str:>12s}  {cv_str:>10s}  {flag}")
    print("=" * 80)
    print()

    matches = [r for r in report["candidates"] if r["match"]]
    if matches:
        m = matches[0]
        print(f"BEST MATCH: '{m['rule']}'")
        print(f"  Pearson r = {m['pearson']:.6f}")
        print(f"  score ≈ {m['ratio_mean']:.4f} × {m['rule']}")
        print(f"  (CV of ratio = {m['ratio_cv']:.2e}; effectively constant)")
        print()
        print("This is your slice-selection rule. Document it in the revised "
              "Experimental Setup section.")
    else:
        # No exact match; report the strongest correlation as a starting point.
        best = max(report["candidates"],
                   key=lambda r: -1 if np.isnan(r["pearson"]) else r["pearson"])
        print(f"NO EXACT MATCH FOUND. Best correlation: '{best['rule']}' "
              f"(r = {best['pearson']:.4f}).")
        print("The actual scoring rule may be a composite of the candidates "
              "above. Inspect the JSON report for details and consider extending "
              "the candidate list.")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
