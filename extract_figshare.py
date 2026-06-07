"""
extract_figshare.py
===================
Convert the Figshare brain tumor dataset (3064 .mat files) into a folder of
256x256 grayscale PNG images suitable for FID computation against our
BraTS-trained generator outputs.

Background:
    Figshare dataset = 3064 T1ce MRI images from 233 patients, organized
    as MATLAB v7.3 .mat files in 4 subfolders. Each file stores:
        cjdata.image      → 512x512 int16 (raw MRI intensity)
        cjdata.tumorMask  → 512x512 uint8 binary
        cjdata.label      → 1/2/3 (meningioma/glioma/pituitary)
        cjdata.PID        → patient ID
        cjdata.tumorBorder → border coordinates (not needed for our purpose)

What this script does:
    1. Walks the 4 subfolders, locates all .mat files
    2. For each: loads image, applies per-image min-max normalization to [0, 255]
    3. Resizes to 256x256 with bilinear interpolation
    4. Optionally also saves the resized binary tumor mask (for future use)
    5. Writes results to two output directories

What this script does NOT do:
    - Apply our model's overlay coloring (we keep Figshare as grayscale for FID)
    - Filter by tumor class (we use all 3064 images)
    - Use the cvind.mat fold assignments (irrelevant for distribution-overlap FID)

Usage (on the Mac where the data lives):
    python extract_figshare.py \\
        --src ~/Downloads/figshare_brain_tumor \\
        --out_images ~/Downloads/figshare_extracted/images \\
        --out_masks  ~/Downloads/figshare_extracted/masks \\
        --img_size 256

Expected runtime: ~3-5 minutes for all 3064 files on a typical laptop.

The output `images/` folder is then uploaded/synced to the GPU server for
crossdataset_fid.py to consume.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def load_mat_file(path: str) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Read one Figshare .mat file and return (image, mask, label, pid).

    All fields use the cjdata.* layout documented in README_2024.txt.

    Args:
        path: Path to a single .mat file.

    Returns:
        image: 2D float array, raw MRI intensities (typical range int16).
        mask:  2D uint8 array, binary tumor mask.
        label: int in {1, 2, 3}.
        pid:   patient ID as decoded ASCII string.
    """
    import h5py
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        image = np.array(cj["image"][...], dtype=np.float32)
        mask = np.array(cj["tumorMask"][...], dtype=np.uint8)
        label = int(np.array(cj["label"]).flatten()[0])
        pid_arr = np.array(cj["PID"]).flatten()
        try:
            pid = "".join(chr(int(c)) for c in pid_arr)
        except (ValueError, OverflowError):
            pid = str(pid_arr.tolist())
    return image, mask, label, pid


def normalize_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Per-image min-max normalize to [0, 255] uint8.

    Figshare images come in raw int16 with varying dynamic ranges per file
    (e.g., 0-3366 in file 1.mat). Per-image min-max is the standard choice
    for visualizing MRI scans and matches how the BraTS overlay pipeline
    handles intensity normalization.

    Args:
        image: 2D float array.

    Returns:
        2D uint8 array in [0, 255].
    """
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    norm = (image - lo) / (hi - lo)
    return (norm * 255).clip(0, 255).astype(np.uint8)


def resize_to(arr: np.ndarray, size: int, mode: str) -> np.ndarray:
    """Resize 2D array to (size, size) using PIL.

    Args:
        arr: 2D uint8 (image) or 2D uint8 (mask).
        size: Target side length in pixels.
        mode: "BILINEAR" for images, "NEAREST" for masks.

    Returns:
        2D uint8 array of shape (size, size).
    """
    pil = Image.fromarray(arr, mode="L")
    if mode == "BILINEAR":
        resampled = pil.resize((size, size), Image.BILINEAR)
    elif mode == "NEAREST":
        resampled = pil.resize((size, size), Image.NEAREST)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.array(resampled, dtype=np.uint8)


def discover_mat_files(src_dir: str) -> list[str]:
    """Walk all subdirectories of src_dir and return paths to all .mat files
    EXCLUDING the top-level cvind.mat.

    Args:
        src_dir: Root directory containing the 4 brainTumorDataPublic_* folders.

    Returns:
        Sorted list of absolute paths to .mat files.
    """
    src = Path(src_dir)
    candidates = []
    for p in src.rglob("*.mat"):
        if p.name.lower() == "cvind.mat":
            continue
        candidates.append(str(p.absolute()))
    candidates.sort()
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Root directory containing the Figshare folders.")
    parser.add_argument("--out_images", required=True,
                        help="Output directory for PNG images.")
    parser.add_argument("--out_masks", default=None,
                        help="Output directory for PNG masks (optional).")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Target square resolution.")
    parser.add_argument("--save_metadata", action="store_true",
                        help="Also write a CSV listing label/PID per filename.")
    args = parser.parse_args()

    files = discover_mat_files(args.src)
    print(f"Discovered {len(files)} .mat files under {args.src}")
    if len(files) == 0:
        sys.exit("No .mat files found. Check --src path.")

    os.makedirs(args.out_images, exist_ok=True)
    if args.out_masks:
        os.makedirs(args.out_masks, exist_ok=True)

    metadata_rows = []
    errors = []

    for i, path in enumerate(files):
        # Use the original filename's stem so files are traceable
        stem = Path(path).stem  # e.g., "1" from "1.mat"
        out_name = f"{stem}.png"

        try:
            image, mask, label, pid = load_mat_file(path)
        except Exception as e:
            errors.append((path, str(e)))
            continue

        img_uint8 = normalize_image_to_uint8(image)
        img_resized = resize_to(img_uint8, args.img_size, "BILINEAR")
        Image.fromarray(img_resized, mode="L").save(
            os.path.join(args.out_images, out_name))

        if args.out_masks:
            mask_resized = resize_to((mask * 255).astype(np.uint8),
                                     args.img_size, "NEAREST")
            Image.fromarray(mask_resized, mode="L").save(
                os.path.join(args.out_masks, out_name))

        metadata_rows.append({
            "filename": out_name,
            "src_path": path,
            "label": label,
            "label_name": {1: "meningioma", 2: "glioma", 3: "pituitary"}.get(label, "unknown"),
            "patient_id": pid,
        })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(files)}")

    print(f"\nDone. Wrote {len(metadata_rows)} images to {args.out_images}/")

    if args.save_metadata:
        import csv
        meta_path = os.path.join(os.path.dirname(args.out_images),
                                 "figshare_metadata.csv")
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"Wrote metadata to {meta_path}")

    if errors:
        print(f"\n[warn] {len(errors)} files failed to read:")
        for path, msg in errors[:5]:
            print(f"  {path}: {msg}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


if __name__ == "__main__":
    main()
