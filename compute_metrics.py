"""
compute_metrics.py
==================
Comprehensive metric evaluation for the IEEE Access revision of:
    "Self-Explainable WGAN-GP for Brain Tumor MRI Synthesis Using KMEx Prototype Learning"

Addresses Reviewer 1 #4: "Stronger GAN metrics required (FID, LPIPS, Precision/Recall, Diversity)"

For a given trained Generator checkpoint, this script:
    1. Generates N synthetic samples conditioned on a mask source.
    2. Computes the full metric suite against the real training distribution.
    3. Writes a structured JSON results file.

Metrics computed:
    - FID:        Fréchet Inception Distance (lower = better)
    - KID:        Kernel Inception Distance, unbiased (lower = better)
    - LPIPS:      mean pairwise LPIPS among generated samples (higher = more diverse)
    - Precision:  Kynkäänniemi et al. 2019 (higher = better fidelity)
    - Recall:     Kynkäänniemi et al. 2019 (higher = better coverage)
    - SSIM:       Mean SSIM between generated and nearest real (legacy compatibility)
    - MS-SSIM:    Mean pairwise MS-SSIM among generated samples (MMIS, legacy)

Usage:
    # Evaluate the proposed model checkpoint:
    python compute_metrics.py \\
        --ckpt_g proposed_inline_seed0/G_final.pt \\
        --data_dir h5_train \\
        --overlay_cache h5_train/overlay_train \\
        --model proposed \\
        --n_samples 5000 \\
        --mask_source held_out_brats \\
        --held_out_dir h5_held_out \\
        --out_json results/proposed_seed0_metrics.json

    # Evaluate a baseline checkpoint:
    python compute_metrics.py \\
        --ckpt_g baselines_cond_seed0/wgangp_plain/G_final.pt \\
        --data_dir h5_train \\
        --overlay_cache h5_train/overlay_train \\
        --model baseline_wgangp \\
        --n_samples 5000 \\
        --mask_source training \\
        --out_json results/baseline_wgangp_seed0_metrics.json

Output JSON schema:
    {
      "checkpoint": "...",
      "model": "proposed|baseline_*",
      "config": {n_samples, mask_source, seed, ...},
      "metrics": {
        "fid": float,
        "kid_mean": float, "kid_std": float,
        "lpips_diversity": float,
        "precision": float, "recall": float,
        "ssim": float, "ms_ssim_mmis": float,
        "n_real": int, "n_fake": int
      },
      "timing_sec": {generation: float, fid: float, ...}
    }

Author: [your name]
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import utils as vutils


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """Seed all relevant RNGs."""
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Model factory — must match training-time architecture exactly
# =============================================================================

def build_generator(model_type: str, nz: int, nc: int, mask_nc: int,
                    img_size: int, ngf: int = 64) -> torch.nn.Module:
    """Construct the Generator architecture matching the training-time model.

    For baselines, this builds the CondGenerator from baseline_conditional.py.
    For the proposed model, this builds the Generator from v17.py.

    Args:
        model_type: One of "proposed", "baseline_dcgan", "baseline_wgangp",
                    "baseline_sagan".
        nz: Latent dimensionality.
        nc: Image channel count.
        mask_nc: Mask channel count.
        img_size: Output resolution.
        ngf: Generator filter count.

    Returns:
        Instantiated nn.Module (weights NOT loaded; load separately).
    """
    use_attention = (model_type == "baseline_sagan")

    if model_type.startswith("baseline_"):
        # Import lazily so that this script is portable even if one of the two
        # codebases is missing from PYTHONPATH.
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from baseline_conditional import CondGenerator
        return CondGenerator(
            nz=nz, nc=nc, mask_nc=mask_nc, ngf=ngf,
            img_size=img_size, use_attention=use_attention,
        )

    elif model_type == "proposed":
        # The proposed model's Generator from v17.py / v17_inline_kmex.py
        # exposes a build_generator(steps, args) function (not a class).
        # We construct a minimal args namespace matching what the function
        # actually reads: nz, gf, nc, mask_nc.  Note 'gf' not 'ngf'.
        import sys, math
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from v17_inline_kmex import build_generator as _proposed_build_g
        except ImportError:
            from v17 import build_generator as _proposed_build_g

        class _ArgsShim:
            pass
        shim = _ArgsShim()
        shim.nz = nz
        shim.gf = ngf            # v17 uses 'gf' as the filter count
        shim.nc = nc
        shim.mask_nc = mask_nc

        steps = int(math.log2(img_size)) - int(math.log2(4))
        return _proposed_build_g(steps, shim)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_generator(ckpt_path: str, model_type: str, nz: int, nc: int,
                   mask_nc: int, img_size: int, device: torch.device) -> torch.nn.Module:
    """Load a Generator checkpoint and move it to device in eval mode."""
    G = build_generator(model_type, nz, nc, mask_nc, img_size)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Some checkpoints are bare state dicts; others are wrapped in a dict.
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    G.load_state_dict(sd, strict=True)
    G.to(device).eval()
    return G


# =============================================================================
# Mask sourcing — what conditions our generated samples?
# =============================================================================

def get_mask_source(
    mask_source: str,
    data_dir: str,
    overlay_cache: str,
    img_size: int,
    held_out_dir: Optional[str] = None,
) -> torch.utils.data.Dataset:
    """Return a dataset that yields (image, mask) pairs; we use only the masks.

    Args:
        mask_source: "training" (reuse training masks) or "held_out_brats"
                     (use masks from BraTS volumes NOT in training set).
        data_dir: Training H5 directory.
        overlay_cache: Overlay cache directory.
        img_size: Image resolution.
        held_out_dir: Required if mask_source="held_out_brats".

    Returns:
        Dataset whose __getitem__ returns (image_tensor, mask_tensor).
    """
    # Import H5OverlayDataset from either codebase (same class definition)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from baseline_conditional import H5OverlayDataset
    except ImportError:
        from v17_inline_kmex import H5OverlayDataset

    if mask_source == "training":
        return H5OverlayDataset(root=data_dir, img_size=img_size,
                                overlay_cache=overlay_cache)
    elif mask_source == "held_out_brats":
        if held_out_dir is None or not os.path.isdir(held_out_dir):
            raise FileNotFoundError(
                f"mask_source=held_out_brats requires --held_out_dir; "
                f"got {held_out_dir}")
        return H5OverlayDataset(root=held_out_dir, img_size=img_size,
                                overlay_cache=None)
    else:
        raise ValueError(f"Unknown mask_source: {mask_source}")


# =============================================================================
# Sample generation
# =============================================================================

def generate_samples(
    G: torch.nn.Module,
    mask_dataset: torch.utils.data.Dataset,
    n_samples: int,
    nz: int,
    batch_size: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Generate n_samples fake images by sampling masks (with replacement) and
    drawing fresh latents for each.

    Returns:
        Tensor of shape (n_samples, nc, H, W) in [-1, 1] on CPU.
    """
    set_seed(seed)
    G.eval()
    fakes = []
    n_done = 0
    indices = np.random.choice(len(mask_dataset), size=n_samples, replace=True)

    with torch.no_grad():
        i = 0
        while i < n_samples:
            batch_idx = indices[i : i + batch_size]
            masks = torch.stack([mask_dataset[int(j)][1] for j in batch_idx]).to(device)
            # Match training-time z shape: [B, nz, 1, 1], not [B, nz].
            # Without this, broadcasting in G's forward produces [B, nz, B, nz]
            # latents and the output spatial dims depend on batch size,
            # making the last partial batch incompatible with the others.
            z = torch.randn(len(batch_idx), nz, 1, 1, device=device)
            out = G(z, masks).cpu()
            fakes.append(out)
            i += len(batch_idx)
            n_done += len(batch_idx)
            if n_done % 500 < batch_size:
                print(f"  Generated {n_done}/{n_samples}")
    return torch.cat(fakes, dim=0)[:n_samples]


# =============================================================================
# Real-sample loader
# =============================================================================

def load_real_samples(
    dataset: torch.utils.data.Dataset,
    n_samples: int,
    batch_size: int = 16,
) -> torch.Tensor:
    """Load up to n_samples real images from a dataset in [-1, 1]."""
    n = min(n_samples, len(dataset))
    reals = []
    for i in range(0, n, batch_size):
        batch = torch.stack([dataset[j][0] for j in range(i, min(i + batch_size, n))])
        reals.append(batch)
    return torch.cat(reals, dim=0)


# =============================================================================
# Metric computation — these wrap external libraries
# =============================================================================

def save_image_tensor_to_dir(tensor: torch.Tensor, out_dir: str) -> None:
    """Save each image in tensor [-1, 1] as a PNG in out_dir.

    torch-fidelity reads from disk, so we materialize the tensors.
    """
    os.makedirs(out_dir, exist_ok=True)
    imgs = (tensor.clamp(-1, 1) + 1) / 2  # [0, 1]
    for i, img in enumerate(imgs):
        vutils.save_image(img, os.path.join(out_dir, f"{i:06d}.png"), normalize=False)


def compute_fid_kid(
    reals: torch.Tensor, fakes: torch.Tensor, device: torch.device,
) -> dict:
    """Compute FID and unbiased KID directly from InceptionV3 features.

    Avoids torch-fidelity's internal DataLoader (which spawns worker
    processes that require shared memory that the Docker container
    doesn't have). Uses our existing _inception_features() extractor.

    Returns:
        Dict with keys "fid", "kid_mean", "kid_std".
    """
    from scipy import linalg as scipy_linalg

    print("  Extracting Inception features (real)...")
    feats_real = _inception_features(reals, device)
    print(f"    shape={feats_real.shape}")
    print("  Extracting Inception features (fake)...")
    feats_fake = _inception_features(fakes, device)
    print(f"    shape={feats_fake.shape}")

    # ----- FID -----
    # FID = ||mu_r - mu_f||^2 + Tr(sigma_r + sigma_f - 2 * sqrt(sigma_r @ sigma_f))
    mu_r, mu_f = feats_real.mean(axis=0), feats_fake.mean(axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_f = np.cov(feats_fake, rowvar=False)
    diff = mu_r - mu_f

    # Stable matrix sqrt — scipy returns complex due to small numerical error
    covmean, _ = scipy_linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_val = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))

    # ----- KID (unbiased) -----
    # Polynomial-kernel MMD^2 over n_subsets random subsets.
    # Subset size = min(1000, smaller-set-size) per the standard.
    n_subsets, subset_size = 10, min(1000, len(feats_real), len(feats_fake))
    rng = np.random.RandomState(42)
    mmds = []
    for _ in range(n_subsets):
        idx_r = rng.choice(len(feats_real), size=subset_size, replace=False)
        idx_f = rng.choice(len(feats_fake), size=subset_size, replace=False)
        x, y = feats_real[idx_r], feats_fake[idx_f]
        d = x.shape[1]
        K_xx = (x @ x.T / d + 1) ** 3
        K_yy = (y @ y.T / d + 1) ** 3
        K_xy = (x @ y.T / d + 1) ** 3
        # Unbiased MMD^2 — zero out the diagonal of self-kernels
        np.fill_diagonal(K_xx, 0)
        np.fill_diagonal(K_yy, 0)
        m = subset_size
        mmd = (K_xx.sum() / (m * (m - 1))
               + K_yy.sum() / (m * (m - 1))
               - 2 * K_xy.mean())
        mmds.append(mmd)

    return {
        "fid": fid_val,
        "kid_mean": float(np.mean(mmds)),
        "kid_std": float(np.std(mmds)),
    }


def _count_pngs(directory: str) -> int:
    return len(list(Path(directory).glob("*.png")))


def compute_lpips_diversity(
    fakes: torch.Tensor,
    device: torch.device,
    n_pairs: int = 1000,
) -> float:
    """Mean pairwise LPIPS distance among generated samples.

    Higher = more diverse generations. Random pairs are sampled to keep the
    cost finite (n_pairs=1000 is a standard subsample).
    """
    import lpips
    loss = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    n = fakes.size(0)
    rng = np.random.RandomState(0)
    pairs = [(rng.randint(0, n), rng.randint(0, n)) for _ in range(n_pairs)]
    pairs = [(a, b) for (a, b) in pairs if a != b]
    distances = []
    with torch.no_grad():
        for (a, b) in pairs:
            xa = fakes[a:a+1].to(device).clamp(-1, 1)
            xb = fakes[b:b+1].to(device).clamp(-1, 1)
            d = loss(xa, xb).item()
            distances.append(d)
    return float(np.mean(distances))


def compute_precision_recall(
    reals: torch.Tensor,
    fakes: torch.Tensor,
    device: torch.device,
    k_nearest: int = 5,
) -> dict:
    """Precision/Recall via Kynkäänniemi et al. 2019 (prdc package).

    Uses InceptionV3 features. k=5 is the recommended default.

    Returns:
        Dict with keys "precision", "recall".
    """
    from prdc import compute_prdc
    feats_real = _inception_features(reals, device)
    feats_fake = _inception_features(fakes, device)
    metrics = compute_prdc(
        real_features=feats_real,
        fake_features=feats_fake,
        nearest_k=k_nearest,
    )
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
    }


def _inception_features(images: torch.Tensor, device: torch.device,
                        batch_size: int = 50) -> np.ndarray:
    """Extract InceptionV3 pool3 features for a tensor batch in [-1, 1].

    Returns:
        np.ndarray of shape (N, 2048).
    """
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True).to(device).eval()
    # Replace final FC with identity so we get pool3 features
    model.fc = torch.nn.Identity()
    preprocess = weights.transforms()
    feats = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size]
            # Map from [-1, 1] -> [0, 1] -> PIL-style 0-255 -> preprocess
            batch_01 = (batch.clamp(-1, 1) + 1) / 2
            batch_resized = F.interpolate(batch_01, size=(299, 299),
                                          mode='bilinear', align_corners=False)
            batch_norm = (batch_resized - torch.tensor([0.485, 0.456, 0.406],
                                                       device=batch_resized.device).view(1, 3, 1, 1)) / \
                         torch.tensor([0.229, 0.224, 0.225],
                                      device=batch_resized.device).view(1, 3, 1, 1)
            batch_norm = batch_norm.to(device)
            f = model(batch_norm)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_ssim_msssim(
    reals: torch.Tensor,
    fakes: torch.Tensor,
    device: torch.device,
) -> dict:
    """Legacy SSIM and MS-SSIM (MMIS) for backward compatibility with the
    original manuscript's Table I.

    SSIM is computed pairwise between (fake_i, real_i) for i in [0, min(N)).
    MS-SSIM is computed as mean pairwise on random fake-fake pairs (the MMIS
    diversity metric the original paper used).
    """
    from skimage.metrics import structural_similarity as sk_ssim
    n = min(reals.size(0), fakes.size(0))
    ssim_values = []
    for i in range(n):
        r = (reals[i].clamp(-1, 1) + 1) / 2
        f = (fakes[i].clamp(-1, 1) + 1) / 2
        # SSIM on grayscale (mean of channels) at full resolution
        r_gray = r.mean(0).numpy()
        f_gray = f.mean(0).numpy()
        ssim_values.append(sk_ssim(r_gray, f_gray, data_range=1.0))

    # MMIS: mean pairwise MS-SSIM among fakes
    # MS-SSIM with default 5 scales requires images >= 160x160 px.
    H, W = fakes.shape[-2:]
    if min(H, W) < 160:
        print(f"  [warn] MS-SSIM requires image >= 160px; got {H}x{W}. "
              f"Skipping MMIS (returning 0.0).")
        msssim_values = [0.0]
    else:
        from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
        msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        rng = np.random.RandomState(0)
        pairs = [(rng.randint(0, n), rng.randint(0, n)) for _ in range(500)]
        pairs = [(a, b) for (a, b) in pairs if a != b]
        msssim_values = []
        for (a, b) in pairs:
            fa = fakes[a:a+1].to(device)
            fb = fakes[b:b+1].to(device)
            msssim_values.append(msssim_metric(fa, fb).item())

    return {
        "ssim": float(np.mean(ssim_values)),
        "ms_ssim_mmis": float(np.mean(msssim_values)),
    }


# =============================================================================
# Main entry
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_g", required=True, help="Generator checkpoint path.")
    parser.add_argument("--model", required=True,
                        choices=["proposed", "baseline_dcgan", "baseline_wgangp",
                                 "baseline_sagan"],
                        help="Which model architecture to load.")
    parser.add_argument("--data_dir", required=True,
                        help="Training H5 dir (real samples reference).")
    parser.add_argument("--overlay_cache", required=True,
                        help="Overlay cache for training data.")
    parser.add_argument("--held_out_dir", default=None,
                        help="Held-out H5 dir; required if --mask_source=held_out_brats.")
    parser.add_argument("--mask_source", default="training",
                        choices=["training", "held_out_brats"],
                        help="Where the generated-sample conditioning masks come from.")

    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Number of synthetic samples to generate.")
    parser.add_argument("--n_real", type=int, default=0,
                        help="Number of real samples to use. 0 = all available.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--mask_nc", type=int, default=3)
    parser.add_argument("--nz", type=int, default=128)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", required=True,
                        help="Output path for the structured results JSON.")
    parser.add_argument("--keep_tempdirs", action="store_true",
                        help="Keep the generated-image tempdirs (for inspection).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    timing = {}
    config = {**vars(args)}

    # 1. Load model
    print(f"Loading {args.model} from {args.ckpt_g} ...")
    t0 = time.time()
    G = load_generator(
        ckpt_path=args.ckpt_g, model_type=args.model,
        nz=args.nz, nc=args.nc, mask_nc=args.mask_nc,
        img_size=args.img_size, device=device,
    )
    timing["load"] = time.time() - t0

    # 2. Mask source
    print(f"Setting up mask source ({args.mask_source}) ...")
    mask_ds = get_mask_source(
        mask_source=args.mask_source,
        data_dir=args.data_dir,
        overlay_cache=args.overlay_cache,
        img_size=args.img_size,
        held_out_dir=args.held_out_dir,
    )

    # 3. Generate fakes
    print(f"Generating {args.n_samples} samples ...")
    t0 = time.time()
    fakes = generate_samples(
        G=G, mask_dataset=mask_ds,
        n_samples=args.n_samples, nz=args.nz,
        batch_size=args.batch_size, device=device, seed=args.seed,
    )
    timing["generation"] = time.time() - t0
    print(f"  done. shape={tuple(fakes.shape)}, time={timing['generation']:.1f}s")

    # 4. Load reals from training dataset
    print(f"Loading real samples ...")
    real_ds = get_mask_source(
        mask_source="training", data_dir=args.data_dir,
        overlay_cache=args.overlay_cache, img_size=args.img_size,
    )
    n_real = args.n_real or len(real_ds)
    reals = load_real_samples(real_ds, n_samples=n_real)
    print(f"  loaded {reals.size(0)} real samples")

    # 5. FID/KID (direct, no PNG materialization, no torch-fidelity DataLoader)
    print("Computing FID/KID (direct from features)...")
    t0 = time.time()
    fid_kid = compute_fid_kid(reals, fakes, device=device)
    timing["fid_kid"] = time.time() - t0
    print(f"  FID = {fid_kid['fid']:.4f}, KID = {fid_kid['kid_mean']:.4e}")

    # 6. LPIPS diversity
    print("Computing LPIPS pairwise diversity (1000 pairs) ...")
    t0 = time.time()
    lpips_div = compute_lpips_diversity(fakes, device=device, n_pairs=1000)
    timing["lpips"] = time.time() - t0
    print(f"  LPIPS diversity = {lpips_div:.4f}")

    # 7. Precision/Recall
    print("Computing Precision/Recall (Kynkäänniemi et al., k=5) ...")
    t0 = time.time()
    pr = compute_precision_recall(reals, fakes, device=device, k_nearest=5)
    timing["pr"] = time.time() - t0
    print(f"  Precision = {pr['precision']:.4f}, Recall = {pr['recall']:.4f}")

    # 8. SSIM and MS-SSIM/MMIS (legacy)
    print("Computing SSIM and MMIS (legacy compatibility) ...")
    t0 = time.time()
    ssim_mmis = compute_ssim_msssim(reals, fakes, device=device)
    timing["ssim_mmis"] = time.time() - t0
    print(f"  SSIM = {ssim_mmis['ssim']:.4f}, MMIS = {ssim_mmis['ms_ssim_mmis']:.4f}")

    # 9. (PNG materialization removed — was only needed for torch-fidelity)

    # 10. Write results
    results = {
        "checkpoint": args.ckpt_g,
        "model": args.model,
        "config": {k: v for k, v in config.items() if k != "ckpt_g"},
        "metrics": {
            **fid_kid,
            "lpips_diversity": lpips_div,
            **pr,
            **ssim_mmis,
            "n_real": int(reals.size(0)),
            "n_fake": int(fakes.size(0)),
        },
        "timing_sec": timing,
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out_json}")

    # Pretty-print summary
    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    for k, v in results["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:24s}  {v:.4f}")
        else:
            print(f"  {k:24s}  {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
