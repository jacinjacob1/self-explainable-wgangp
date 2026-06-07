"""
precompute_npy_masks.py
========================
Populate the missing .npy mask files in the overlay_train cache directory.

The H5OverlayDataset in v17_inline_kmex.py caches each sample as TWO files:
  - overlay_train/<volume_X_slice_Y>.png    (the RGB overlay image)
  - overlay_train/<volume_X_slice_Y>.npy    (the 3-channel one-hot mask)

In our deployed cache, only the .png files were written; the .npy files are
missing. This script reads each H5 file in the data directory, extracts the
3-channel one-hot mask exactly as H5OverlayDataset.__getitem__ does, and
saves the .npy file alongside the existing .png.

After running this script once, the H5OverlayDataset takes its cache branch
and reads both .png + .npy without re-touching the H5 files.

Usage:
    python precompute_npy_masks.py \\
        --data_dir h5_train \\
        --overlay_cache overlay_train

Expected runtime: ~30 seconds for 1107 files on any reasonable disk.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def build_one_hot_mask(msk: np.ndarray) -> np.ndarray:
    """Convert raw H5 mask to 3-channel one-hot float32 with shape (H, W, 3).

    This logic mirrors H5OverlayDataset.__getitem__ exactly so the resulting
    .npy files are bitwise-identical to what the dataset would have cached
    on the first read of each H5 file.

    Args:
        msk: Raw mask array from H5 file. Can be (H, W, 3) one-hot already,
             or (H, W) integer labels.

    Returns:
        Array of shape (H, W, 3), dtype float32, values in {0.0, 1.0}.
    """
    if msk.ndim == 3 and msk.shape[-1] == 3:
        # Already one-hot — just cast to float32
        return msk.astype("float32")

    # Integer labels (H, W) → expand to 3-channel one-hot
    # Convention matches the dataset's `make_overlay_rgb` indexing.
    label = msk
    return np.stack([
        (label == 1).astype("float32"),
        (label == 2).astype("float32"),
        (label == 3).astype("float32"),
    ], axis=-1)


def main() -> None:
    import h5py

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root directory of H5 files (e.g., h5_train).")
    parser.add_argument("--overlay_cache", required=True,
                        help="Cache directory containing .png files; will "
                             "have matching .npy files written into it.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing .npy files. Default skips them.")
    parser.add_argument("--check_only", action="store_true",
                        help="Don't write anything; just report what's missing.")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    cache_root = Path(args.overlay_cache)

    if not data_root.is_dir():
        sys.exit(f"--data_dir not found: {data_root}")
    if not cache_root.is_dir():
        sys.exit(f"--overlay_cache not found: {cache_root}")

    h5_files = sorted(data_root.rglob("*.h5"))
    print(f"Found {len(h5_files)} H5 files in {data_root}")

    if len(h5_files) == 0:
        sys.exit("No H5 files; nothing to do.")

    written = 0
    skipped = 0
    missing_png = []
    errors = []

    for i, h5_path in enumerate(h5_files):
        stem = h5_path.stem  # e.g., volume_1_slice_71
        cache_png = cache_root / f"{stem}.png"
        cache_npy = cache_root / f"{stem}.npy"

        if not cache_png.exists():
            missing_png.append(stem)
            continue

        if cache_npy.exists() and not args.force:
            skipped += 1
            continue

        if args.check_only:
            written += 1  # would-write count
            continue

        # Read the H5 file's mask, expand to one-hot, save .npy
        try:
            with h5py.File(str(h5_path), "r") as f:
                msk = np.array(f["mask"])
            one_hot = build_one_hot_mask(msk)
            np.save(cache_npy, one_hot)
            written += 1
        except Exception as e:
            errors.append((stem, str(e)))
            continue

        if (i + 1) % 100 == 0:
            verb = "would-write" if args.check_only else "wrote"
            print(f"  Processed {i + 1}/{len(h5_files)} ({verb} {written}, skipped {skipped})")

    print()
    print("=" * 60)
    if args.check_only:
        print(f"CHECK-ONLY MODE — nothing written")
        print(f"  Would write:   {written}")
    else:
        print(f"Wrote {written} .npy files")
    print(f"Skipped (.npy already existed): {skipped}")
    if missing_png:
        print(f"H5 files with no matching .png in cache: {len(missing_png)}")
        for stem in missing_png[:5]:
            print(f"  - {stem}.png missing")
        if len(missing_png) > 5:
            print(f"  ... and {len(missing_png) - 5} more")
    if errors:
        print(f"Errors: {len(errors)}")
        for stem, msg in errors[:5]:
            print(f"  - {stem}: {msg}")
    print("=" * 60)


if __name__ == "__main__":
    main()
