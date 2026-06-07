"""
crossdataset_fid.py
====================
Cross-dataset FID evaluation for the IEEE Access revision:
addresses Reviewer 2 #5 ("Lack of cross-dataset evaluation").

Computes three FID comparisons, all in 256x256 grayscale:

    (1) Real BraTS ↔ Real Figshare         → the "ceiling" / modality gap
    (2) Generated ↔ Real BraTS (grayscale) → in-domain generation quality
    (3) Generated ↔ Real Figshare          → the actual cross-dataset test

Interpretation:
    - (3) ≈ (1): our generated images are about as distant from Figshare as
              real BraTS is from Figshare → model captures BraTS distribution
              without producing Figshare-specific anomalies (cleanest outcome)
    - (3) >> (1): our generated images are further from Figshare than real
              BraTS is → model produces BraTS-specific artifacts not in real MRI
    - (2) alone is a sanity check that grayscale FID gives a similar order to
              the RGB-overlay FID already reported in Table I

Grayscale rationale:
    Our generated samples have colored tumor overlays; Figshare is grayscale
    T1ce. Computing FID directly would conflate distributional differences
    with colormap differences. We convert generated and BraTS samples to
    grayscale to isolate the anatomy/texture comparison.

Usage:
    python crossdataset_fid.py \\
        --ckpt_g proposed_inline_seed0_v2/G_final.pt \\
        --data_dir h5_train \\
        --overlay_cache h5_train/overlay_train \\
        --figshare_dir /path/to/figshare_extracted/images \\
        --n_samples 3000 \\
        --seed 0 \\
        --out_json results/crossdataset_fid.json

Author: [your name]
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import utils as vutils

# Import architectures and dataset from the training codebase
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from v17_inline_kmex import build_generator, H5OverlayDataset  # noqa: E402


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Generator loading (matches the architecture used in v17_inline_kmex.py)
# =============================================================================

def load_generator_from_checkpoint(
    ckpt_path: str,
    nz: int, nc: int, mask_nc: int,
    img_size: int, ngf: int,
    num_prototypes: int,
    use_overlays: bool,
    device: torch.device,
) -> torch.nn.Module:
    """Reconstruct the proposed-model Generator and load weights."""
    import math
    steps = int(math.log2(img_size)) - int(math.log2(4))

    class _ArgsShim:
        pass
    shim = _ArgsShim()
    # Set BOTH 'gf' (v17_inline_kmex.py's actual attribute name) and 'ngf'
    # (the more conventional name) so the shim works regardless of which
    # attribute build_generator references.
    for k, v in [("nc", nc), ("mask_nc", mask_nc), ("nz", nz),
                 ("gf", ngf), ("ngf", ngf), ("ndf", 64),
                 ("num_prototypes", num_prototypes),
                 ("use_overlays", use_overlays)]:
        setattr(shim, k, v)

    G = build_generator(steps, shim).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    G.load_state_dict(sd, strict=True)
    return G


# =============================================================================
# Generate samples and save as grayscale PNGs
# =============================================================================

def generate_grayscale_samples(
    G: torch.nn.Module,
    mask_dataset: torch.utils.data.Dataset,
    n_samples: int,
    nz: int,
    batch_size: int,
    device: torch.device,
    out_dir: str,
    seed: int = 0,
) -> int:
    """Generate n_samples from G, convert to 256x256 grayscale, save as PNG.

    The conversion to grayscale is performed by simple channel-mean —
    sufficient for FID since FID's Inception network re-projects to 3 channels
    internally via per-channel-mean replication for grayscale inputs.

    Returns:
        Number of images written.
    """
    set_seed(seed)
    G.eval()
    os.makedirs(out_dir, exist_ok=True)
    indices = np.random.choice(len(mask_dataset), size=n_samples, replace=True)

    written = 0
    with torch.no_grad():
        i = 0
        while i < n_samples:
            batch_idx = indices[i : i + batch_size]
            masks = torch.stack([mask_dataset[int(j)][1] for j in batch_idx]).to(device)
            # CRITICAL: the latent must be 4D [B, nz, 1, 1] to match the
            # generator backbone (a ConvTranspose stack expecting a spatial
            # 1x1 input) and to broadcast correctly with the mask embedding
            # zc.unsqueeze(-1).unsqueeze(-1) of shape [B, nz, 1, 1] inside
            # the generator's forward(). A 2D [B, nz] latent silently
            # broadcasts to a malformed 4D tensor and produces corrupt output.
            z = torch.randn(len(batch_idx), nz, 1, 1, device=device)
            fakes = G(z, masks).cpu()  # [B, 3, H, W] in [-1, 1]

            # Convert to grayscale uint8 [0, 255]
            fakes_01 = (fakes.clamp(-1, 1) + 1) / 2
            fakes_gray = fakes_01.mean(dim=1, keepdim=False)  # [B, H, W]
            fakes_uint8 = (fakes_gray * 255).clamp(0, 255).byte().numpy()

            for arr in fakes_uint8:
                Image.fromarray(arr, mode="L").save(
                    os.path.join(out_dir, f"gen_{written:06d}.png"))
                written += 1

            i += len(batch_idx)
            if written % 500 < batch_size:
                print(f"  Generated {written}/{n_samples}")
    return written


# =============================================================================
# Export real BraTS images as grayscale PNGs
# =============================================================================

def export_brats_grayscale(
    dataset: torch.utils.data.Dataset,
    out_dir: str,
    n_samples: int = -1,
) -> int:
    """Save BraTS dataset images as 256x256 grayscale PNGs.

    The BraTS samples already arrive at 256x256 from H5OverlayDataset. For the
    FID comparison we want the raw FLAIR-base appearance without overlay color,
    so we strip the colored overlay by extracting the grayscale-equivalent
    channel-mean. (Original BraTS FLAIR slices have all three channels equal
    where there is no overlay; the channel-mean is faithful to the base
    anatomy in non-tumor regions and a desaturated color in tumor regions.)
    """
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    target = len(dataset) if n_samples <= 0 else min(n_samples, len(dataset))

    for idx in range(target):
        x, _m = dataset[idx]  # x: [3, H, W] in [-1, 1]
        x_01 = (x.clamp(-1, 1) + 1) / 2
        x_gray = x_01.mean(dim=0)  # [H, W]
        arr = (x_gray * 255).clamp(0, 255).byte().numpy()
        Image.fromarray(arr, mode="L").save(
            os.path.join(out_dir, f"brats_{written:06d}.png"))
        written += 1

    return written


# =============================================================================
# FID computation between two directories of PNGs
# =============================================================================

def compute_fid_between(dir_a: str, dir_b: str, device_str: str,
                          batch_size: int = 16) -> float:
    """Compute FID between two folders of PNG images without any DataLoader.

    Why this implementation exists:
        The torch-fidelity library uses an internal DataLoader with worker
        processes by default. In containerized environments where /dev/shm
        is restricted (e.g., 64 MB), the worker processes crash with a
        bus error when transferring tensors via shared memory. Setting
        num_workers=0 in torch-fidelity's API is unreliable across versions.
        This function uses direct synchronous feature extraction instead.

    Implementation:
        - Loads pretrained Inception-v3 from torchvision (cached after
          first call to torch hub).
        - Replaces the final classifier layer with Identity to extract
          2048-dim pooled features.
        - Reads PNGs in batches, applies standard ImageNet preprocessing
          (resize to 299x299, normalize), extracts features.
        - Computes FID = ||μ_a − μ_b||² + Tr(Σ_a + Σ_b − 2(Σ_a Σ_b)^0.5)
          using the same scipy.linalg.sqrtm matrix square root used in
          compute_metrics.py.

    Note on absolute-value comparability:
        FIDs computed here use the standard torchvision Inception-v3
        preprocessing. These values are internally consistent across the
        three comparisons (BraTS↔Figshare, Gen↔BraTS, Gen↔Figshare) but
        are not directly comparable to FIDs reported in Table I from
        compute_metrics.py, which uses different Inception preprocessing.
        Use the ratio (cross/ceiling) as the interpretive quantity.
    """
    import os
    import numpy as np
    import torch
    from PIL import Image
    from scipy import linalg as scipy_linalg
    from torchvision import transforms
    from torchvision.models import inception_v3, Inception_V3_Weights

    device = torch.device(device_str)

    # ---- Load Inception-v3 with classifier head replaced by Identity ----
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True).to(device).eval()
    model.fc = torch.nn.Identity()

    # Standard ImageNet preprocessing for Inception-v3
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(directory: str) -> np.ndarray:
        """Extract 2048-dim Inception features from all PNGs in directory."""
        png_files = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory) if f.lower().endswith(".png")
        ])
        feats = []
        with torch.no_grad():
            for i in range(0, len(png_files), batch_size):
                batch_paths = png_files[i : i + batch_size]
                # PIL load + RGB-convert + preprocess; do this on CPU
                batch = torch.stack([
                    preprocess(Image.open(p).convert("RGB"))
                    for p in batch_paths
                ]).to(device)
                # Inception in eval mode returns tensor (not tuple) when
                # aux_logits=True since we removed the aux training output.
                out = model(batch)
                if isinstance(out, tuple):
                    out = out[0]
                feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)

    # ---- Extract features from both folders ----
    feat_a = extract_features(dir_a)
    feat_b = extract_features(dir_b)

    # ---- Compute FID via the standard mean/covariance + matrix-sqrt formula ----
    mu_a = feat_a.mean(axis=0)
    mu_b = feat_b.mean(axis=0)
    sigma_a = np.cov(feat_a, rowvar=False)
    sigma_b = np.cov(feat_b, rowvar=False)

    diff = mu_a - mu_b
    covmean, _ = scipy_linalg.sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_a + sigma_b - 2.0 * covmean))

    # Free Inception from GPU before returning
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return fid


def count_pngs(directory: str) -> int:
    return len([p for p in os.listdir(directory) if p.endswith(".png")])


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_g", required=True, help="Trained Generator checkpoint.")
    parser.add_argument("--data_dir", required=True, help="BraTS H5 training directory.")
    parser.add_argument("--overlay_cache", required=True,
                        help="BraTS overlay cache directory.")
    parser.add_argument("--figshare_dir", required=True,
                        help="Directory of Figshare PNGs from extract_figshare.py.")
    parser.add_argument("--out_json", required=True,
                        help="Output JSON with the three FID values + metadata.")

    parser.add_argument("--n_samples", type=int, default=3000,
                        help="Number of synthetic samples to generate.")
    parser.add_argument("--n_brats", type=int, default=-1,
                        help="Number of BraTS reals to include. -1 = all.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--mask_nc", type=int, default=3)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--num_prototypes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep_tempdirs", action="store_true",
                        help="Keep generated PNG directories for inspection.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    if not os.path.isdir(args.figshare_dir):
        sys.exit(f"--figshare_dir not found: {args.figshare_dir}")
    n_figshare = count_pngs(args.figshare_dir)
    if n_figshare == 0:
        sys.exit(f"No PNGs in {args.figshare_dir}. Did you run extract_figshare.py?")
    print(f"Figshare PNGs present: {n_figshare}")

    timing = {}

    # ---- Load G ----
    print(f"\nLoading G from {args.ckpt_g} ...")
    t0 = time.time()
    G = load_generator_from_checkpoint(
        ckpt_path=args.ckpt_g, nz=args.nz, nc=args.nc, mask_nc=args.mask_nc,
        img_size=args.img_size, ngf=args.ngf,
        num_prototypes=args.num_prototypes,
        use_overlays=True, device=device,
    )
    timing["load_g"] = time.time() - t0

    # ---- BraTS dataset (for masks + for real images) ----
    print("Setting up BraTS dataset ...")
    brats = H5OverlayDataset(
        root=args.data_dir, img_size=args.img_size,
        overlay_cache=args.overlay_cache,
    )
    print(f"  BraTS samples: {len(brats)}")

    # ---- Generate synthetic samples ----
    print(f"\n[1/4] Generating {args.n_samples} synthetic samples ...")
    t0 = time.time()
    gen_dir = tempfile.mkdtemp(prefix="gen_gray_", dir=".")
    n_gen = generate_grayscale_samples(
        G=G, mask_dataset=brats, n_samples=args.n_samples,
        nz=args.nz, batch_size=args.batch_size, device=device,
        out_dir=gen_dir, seed=args.seed,
    )
    timing["generate"] = time.time() - t0
    print(f"  -> wrote {n_gen} PNGs to {gen_dir}/")

    # ---- Export BraTS reals as grayscale PNGs ----
    print(f"\n[2/4] Exporting BraTS reals as grayscale ...")
    t0 = time.time()
    brats_dir = tempfile.mkdtemp(prefix="brats_gray_", dir=".")
    n_brats = export_brats_grayscale(brats, brats_dir, n_samples=args.n_brats)
    timing["export_brats"] = time.time() - t0
    print(f"  -> wrote {n_brats} PNGs to {brats_dir}/")

    # ---- Three FID comparisons ----
    print(f"\n[3/4] Computing FID (1/3): Real BraTS ↔ Real Figshare (CEILING) ...")
    t0 = time.time()
    fid_brats_figshare = compute_fid_between(brats_dir, args.figshare_dir, args.device)
    timing["fid_brats_figshare"] = time.time() - t0
    print(f"  -> FID(BraTS, Figshare) = {fid_brats_figshare:.4f}")

    print(f"\n[3/4] Computing FID (2/3): Generated ↔ Real BraTS (IN-DOMAIN) ...")
    t0 = time.time()
    fid_gen_brats = compute_fid_between(gen_dir, brats_dir, args.device)
    timing["fid_gen_brats"] = time.time() - t0
    print(f"  -> FID(Generated, BraTS) [grayscale] = {fid_gen_brats:.4f}")

    print(f"\n[3/4] Computing FID (3/3): Generated ↔ Real Figshare (CROSS-DATASET) ...")
    t0 = time.time()
    fid_gen_figshare = compute_fid_between(gen_dir, args.figshare_dir, args.device)
    timing["fid_gen_figshare"] = time.time() - t0
    print(f"  -> FID(Generated, Figshare) = {fid_gen_figshare:.4f}")

    # ---- Interpretation ----
    # Compute the ratio between cross-dataset gen-Figshare and BraTS-Figshare.
    # A ratio close to 1 means our generated images are about as far from
    # Figshare as real BraTS is — the cleanest possible outcome.
    ceiling_ratio = fid_gen_figshare / max(fid_brats_figshare, 1e-6)

    if ceiling_ratio < 1.1:
        interpretation = (
            "Strong cross-dataset performance. Generated images are "
            "approximately as distant from Figshare as real BraTS is, "
            "indicating the model captures BraTS distribution without "
            "producing Figshare-specific anomalies."
        )
    elif ceiling_ratio < 1.5:
        interpretation = (
            "Moderate cross-dataset performance. Generated images are "
            "somewhat further from Figshare than real BraTS is, suggesting "
            "the model produces some BraTS-specific characteristics not "
            "present in real T1ce MRI from a different acquisition source."
        )
    else:
        interpretation = (
            "Weak cross-dataset performance. Generated images are "
            "substantially further from Figshare than real BraTS is, "
            "indicating BraTS-specific generation artifacts. The "
            "framework's images do not generalize to a different MRI "
            "acquisition source."
        )

    # ---- Write results ----
    results = {
        "checkpoint": args.ckpt_g,
        "config": {k: v for k, v in vars(args).items() if k != "ckpt_g"},
        "counts": {
            "figshare": n_figshare,
            "brats": n_brats,
            "generated": n_gen,
        },
        "metrics_grayscale_fid": {
            "fid_real_brats_vs_real_figshare_CEILING": fid_brats_figshare,
            "fid_generated_vs_real_brats_INDOMAIN": fid_gen_brats,
            "fid_generated_vs_real_figshare_CROSSDATASET": fid_gen_figshare,
            "ratio_cross_over_ceiling": ceiling_ratio,
        },
        "interpretation": interpretation,
        "timing_sec": timing,
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[4/4] Wrote {args.out_json}")

    # ---- Cleanup ----
    if not args.keep_tempdirs:
        import shutil
        shutil.rmtree(gen_dir, ignore_errors=True)
        shutil.rmtree(brats_dir, ignore_errors=True)
    else:
        print(f"\nKept temp dirs:")
        print(f"  Generated:  {gen_dir}")
        print(f"  BraTS gray: {brats_dir}")

    # ---- Pretty print summary ----
    print("\n" + "=" * 70)
    print("CROSS-DATASET FID SUMMARY (all grayscale, 256x256)")
    print("=" * 70)
    print(f"  FID(Real BraTS, Real Figshare)  =  {fid_brats_figshare:.2f}    [CEILING]")
    print(f"  FID(Generated, Real BraTS)      =  {fid_gen_brats:.2f}    [IN-DOMAIN]")
    print(f"  FID(Generated, Real Figshare)   =  {fid_gen_figshare:.2f}    [CROSS-DATASET]")
    print(f"  Ratio (cross/ceiling)           =  {ceiling_ratio:.3f}")
    print()
    print(f"  Interpretation: {interpretation}")
    print("=" * 70)


if __name__ == "__main__":
    main()
