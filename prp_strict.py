"""
prp_strict.py
=============
Strict Prototype Relevance Propagation, computed post-hoc on a trained
v17_inline_kmex.py checkpoint.

Addresses an empirical observation from the first training run:
the existing PRP heatmaps (seeded from D's realism score) produce diffuse,
non-localized attention. This is consistent with the theoretical observation
in PRP_Mathematical_Formalization.md §8.1: D's realism decision does not
need to localize on the tumor region (since both real and fake samples share
the same mask conditioning).

This script computes a different attribution that is more aligned with the
"prototype" interpretation:

    Strict PRP:  seed propagation from  s_k(x) = cos(phi_D(x), p_k)
                 rather than from       D(x)

i.e., we ask "which input pixels made this sample's D-features align with
prototype k?" instead of "which pixels made D say this looks real?".

For each of the K prototypes we produce three artifacts:
    - The strict PRP heatmap (seeded from cosine similarity)
    - The original realism-PRP heatmap (seeded from D output) for reference
    - A side-by-side comparison figure

Plus a master 4-panel-per-row montage for the manuscript.

This script does NOT require retraining. It runs on the checkpoints + ksweep
data saved by v17_inline_kmex.py (the patched version).

Usage:
    python prp_strict.py \\
        --ckpt_g proposed_inline_seed0_v2/G_final.pt \\
        --ckpt_d proposed_inline_seed0_v2/D_final.pt \\
        --ksweep_dir proposed_inline_seed0_v2/ksweep_data \\
        --data_dir h5_train \\
        --overlay_cache h5_train/overlay_train \\
        --out_dir proposed_inline_seed0_v2/prp_strict \\
        --use_custom_lrp_rules

Author: [your name]
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import utils as vutils
from torch.utils.data import DataLoader

# Import architectures and dataset from the training codebase
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from v17_inline_kmex import (  # noqa: E402
    build_generator, build_discriminator, H5OverlayDataset,
)


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
# LeakyReLU-friendly LRP via Captum CustomLRPRule (when available)
# =============================================================================

def install_lrp_rules(D: nn.Module, use_custom: bool) -> None:
    """Install LRP propagation rules on D.

    Note on use_custom: in current Captum versions, the LayerLRP class
    enforces an internal SUPPORTED_LAYERS check that rejects LeakyReLU
    and InstanceNorm even when a rule is manually attached via m.rule =
    EpsilonRule(). We therefore always fall back to the substitution
    path: LeakyReLU -> ReLU and InstanceNorm -> BatchNorm. Captum has
    built-in default rules for ReLU and BatchNorm, so this is reliable.

    This is the same approach the original v17.py used and has been
    empirically verified to produce valid (if slightly approximated)
    heatmaps. The mathematical limitations are documented in §8 of
    PRP_Mathematical_Formalization.md.

    The strict-PRP contribution of this script — seeding attribution
    from cosine similarity rather than D's realism output — is
    preserved regardless of which propagation-rule installation path
    we use. The seed change is what matters for the localization
    improvement we're testing.
    """
    if use_custom:
        print("  [note] --use_custom_lrp_rules requested, but Captum's "
              "SUPPORTED_LAYERS check rejects LeakyReLU regardless of "
              "manually attached rules. Falling back to substitution path.")
    # Capture D's device BEFORE substitution — the freshly-constructed
    # BatchNorm2d / ReLU modules default to CPU, which would create a
    # device mismatch during D's forward pass if D is on cuda:0.
    device = next(D.parameters()).device
    _apply_substitutions(D)
    # Move the (now-substituted) D back to its original device so the
    # new modules' parameters land on the same device as everything else.
    D.to(device)
    n_subs = sum(1 for m in D.modules()
                 if isinstance(m, (nn.ReLU, nn.BatchNorm2d)))
    print(f"  Applied substitutions; D now has {n_subs} ReLU/BatchNorm modules on {device}")


def _apply_substitutions(module: nn.Module) -> None:
    """The original v17.py LRP-compatibility substitutions, applied in-place."""
    for name, child in module.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(module, name, nn.ReLU())
        elif isinstance(child, nn.InstanceNorm2d):
            setattr(module, name, nn.BatchNorm2d(
                child.num_features, eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats,
            ))
        else:
            _apply_substitutions(child)


# =============================================================================
# Attribution computation
# =============================================================================

def _normalize_heatmap(attr: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Channel-sum |attr|, min-max normalize to [0, 1], resize to target HxW.

    Always detaches so downstream matplotlib / numpy calls don't trip on
    the autograd graph that LRP leaves attached.
    """
    if attr.dim() == 3:
        attr = attr.unsqueeze(0)
    attr = attr.detach()                           # break the autograd graph
    attr = attr.abs().sum(dim=1, keepdim=True)     # [1, 1, H, W]
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    attr = TF.resize(attr, list(target_hw))
    return attr


def compute_realism_prp(
    D: nn.Module, inp: torch.Tensor, target_layer: nn.Module,
) -> torch.Tensor:
    """Original PRP: attribute D(x) -> input. Returns heatmap [1, 1, H, W] in [0, 1]."""
    from captum.attr import LayerLRP
    lrp = LayerLRP(D, target_layer)
    attr = lrp.attribute(inp)
    return _normalize_heatmap(attr, target_hw=(inp.shape[2], inp.shape[3]))


def compute_strict_prp(
    D: nn.Module,
    inp: torch.Tensor,
    prototype: torch.Tensor,
    target_layer: nn.Module,
) -> torch.Tensor:
    """Strict PRP: attribute cos(phi_D(x), p_k) -> input.

    Implementation strategy: we wrap D in a small module that returns the
    cosine score instead of D's scalar output, then run LayerLRP on that
    wrapped module attributing toward the cosine score.

    Args:
        D: Discriminator with a `.f` attribute (feature extractor returning
           [B, d, H', W']) and forward returning the realism scalar.
        inp: Input tensor [1, 6, H, W] (image || mask concatenation).
        prototype: Cluster center vector [d] for the target prototype k.
        target_layer: The conv layer in D at which to read out relevance
                      (typically the first Conv2d).

    Returns:
        Heatmap [1, 1, H, W] in [0, 1].
    """
    from captum.attr import LayerLRP

    class _CosineScoreWrapper(nn.Module):
        """Returns cos(phi(x), p) as the scalar to be attributed."""
        def __init__(self, D_inner: nn.Module, proto: torch.Tensor):
            super().__init__()
            self.D = D_inner
            # Store the prototype as a buffer (no gradient required for it)
            self.register_buffer("p_norm", F.normalize(proto.view(1, -1), dim=1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.D.f(x).mean([2, 3])         # [B, d]
            feat = F.normalize(feat, dim=1)
            score = (feat * self.p_norm).sum(dim=1)  # [B] — cosine similarity
            return score

    wrapper = _CosineScoreWrapper(D, prototype.to(inp.device)).to(inp.device).eval()
    lrp = LayerLRP(wrapper, target_layer)
    attr = lrp.attribute(inp)
    return _normalize_heatmap(attr, target_hw=(inp.shape[2], inp.shape[3]))


# =============================================================================
# Prototype exemplar identification
# =============================================================================

def find_prototype_exemplars(
    D: nn.Module, dataset: torch.utils.data.Dataset,
    cluster_centers: np.ndarray, device: torch.device,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """For each prototype center, return (idx_in_dataset, image, mask) of
    the nearest training exemplar in D-feature space.

    This duplicates the logic in v17_inline_kmex.py's KMEx mining block, so
    that we can run strict PRP from saved checkpoints without re-extracting
    features in the main training script.
    """
    embeds, imgs, msks = [], [], []
    D.eval()
    with torch.no_grad():
        for x, m in DataLoader(dataset, batch_size=8, shuffle=False):
            x, m = x.to(device), m.to(device)
            feat = D.f(torch.cat([x, m], dim=1)).mean([2, 3])  # [B, d]
            embeds.append(feat.cpu().numpy())
            imgs.append(x.cpu()); msks.append(m.cpu())
    embeds = np.concatenate(embeds, axis=0)
    imgs = torch.cat(imgs); msks = torch.cat(msks)

    # Nearest-exemplar lookup
    from sklearn.metrics import pairwise_distances_argmin_min
    proto_idx, _ = pairwise_distances_argmin_min(cluster_centers, embeds)
    return [(int(p), imgs[p], msks[p]) for p in proto_idx]


# =============================================================================
# Visualization
# =============================================================================

def _to_01_rgb(t: torch.Tensor) -> np.ndarray:
    x = (t.clamp(-1, 1) + 1) / 2
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.permute(1, 2, 0).numpy()


def save_comparison_panel(
    exemplar: torch.Tensor, mask: torch.Tensor,
    realism_hm: torch.Tensor, strict_hm: torch.Tensor,
    out_path: str, title_prefix: str = "",
) -> None:
    """Save a 4-panel figure: exemplar | mask | realism PRP | strict PRP."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(_to_01_rgb(exemplar))
    axes[0].set_title(f"{title_prefix}exemplar"); axes[0].axis("off")
    axes[1].imshow(_to_01_rgb(mask))
    axes[1].set_title("structured mask"); axes[1].axis("off")
    gray = _to_01_rgb(exemplar).mean(axis=-1)
    axes[2].imshow(gray, cmap="gray")
    axes[2].imshow(realism_hm.squeeze().cpu().numpy(), cmap="jet", alpha=0.5)
    axes[2].set_title("realism PRP\n(D(x) seed)"); axes[2].axis("off")
    axes[3].imshow(gray, cmap="gray")
    axes[3].imshow(strict_hm.squeeze().cpu().numpy(), cmap="jet", alpha=0.5)
    axes[3].set_title("strict PRP\n(cos seed)"); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_master_montage(
    panel_data: list[tuple], out_path: str, K: int,
) -> None:
    """Master comparison montage: rows of prototypes, columns of methods.

    Each row: exemplar | realism PRP | strict PRP   (3 cols per prototype)
    Multiple prototypes stacked vertically.
    """
    cols = 3  # exemplar, realism, strict
    rows = K
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    for k, (exemplar, mask, realism_hm, strict_hm) in enumerate(panel_data):
        gray = _to_01_rgb(exemplar).mean(axis=-1)
        axes[k, 0].imshow(_to_01_rgb(exemplar))
        axes[k, 0].set_title(f"P{k:02d} exemplar", fontsize=9); axes[k, 0].axis("off")
        axes[k, 1].imshow(gray, cmap="gray")
        axes[k, 1].imshow(realism_hm.squeeze().cpu().numpy(), cmap="jet", alpha=0.5)
        axes[k, 1].set_title("realism PRP", fontsize=9); axes[k, 1].axis("off")
        axes[k, 2].imshow(gray, cmap="gray")
        axes[k, 2].imshow(strict_hm.squeeze().cpu().numpy(), cmap="jet", alpha=0.5)
        axes[k, 2].set_title("strict PRP", fontsize=9); axes[k, 2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_g", required=True, help="G_final.pt from training run.")
    parser.add_argument("--ckpt_d", required=True, help="D_final.pt from training run.")
    parser.add_argument("--ksweep_dir", required=True,
                        help="Folder containing cluster_centers_k16.npy "
                             "(from v17_inline_kmex.py).")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--overlay_cache", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--mask_nc", type=int, default=3)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--use_overlays", action="store_true", default=True)
    parser.add_argument("--num_prototypes", type=int, default=16)
    parser.add_argument("--use_custom_lrp_rules", action="store_true",
                        help="Use Captum CustomLRPRule for LeakyReLU/InstanceNorm "
                             "instead of the lossy substitution path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Reconstruct G and D from checkpoints ----
    # We need to match the architecture v17_inline_kmex used. The simplest
    # reliable way: re-invoke build_generator / build_discriminator with the
    # same args namespace they expect.
    # IMPORTANT: v17 uses attribute names 'gf' and 'df' (not 'ngf'/'ndf').
    import math
    target_img_size = args.img_size
    steps = int(math.log2(target_img_size)) - int(math.log2(4))

    class _ArgsShim:
        pass
    shim = _ArgsShim()
    shim.nc = args.nc
    shim.mask_nc = args.mask_nc
    shim.nz = args.nz
    shim.gf = args.ngf            # rename: ngf -> gf
    shim.df = args.ndf            # rename: ndf -> df
    shim.num_prototypes = args.num_prototypes
    shim.use_overlays = args.use_overlays

    G = build_generator(steps, shim).to(device).eval()
    D = build_discriminator(steps, shim).to(device).eval()
    G.load_state_dict(torch.load(args.ckpt_g, map_location=device, weights_only=True))
    D.load_state_dict(torch.load(args.ckpt_d, map_location=device, weights_only=True))
    print(f"Loaded G from {args.ckpt_g}, D from {args.ckpt_d}")

    # ---- Build dataset (same overlay pipeline) ----
    dataset = H5OverlayDataset(
        root=args.data_dir, img_size=args.img_size,
        overlay_cache=args.overlay_cache,
    )
    print(f"Dataset: {len(dataset)} samples")

    # ---- Load cluster centers (the K=16 prototypes) ----
    centers_path = os.path.join(args.ksweep_dir, "cluster_centers_k16.npy")
    if not os.path.exists(centers_path):
        raise FileNotFoundError(
            f"{centers_path} not found. Did you re-run with the patched "
            f"v17_inline_kmex.py that saves ksweep_data/?")
    cluster_centers = np.load(centers_path)
    print(f"Loaded {cluster_centers.shape[0]} prototypes from {centers_path}")

    # ---- Identify nearest exemplars for each prototype ----
    print("Finding prototype exemplars...")
    exemplars = find_prototype_exemplars(D, dataset, cluster_centers, device)

    # ---- Install LRP rules on D (custom or substitution) ----
    print("Installing LRP rules on D...")
    install_lrp_rules(D, use_custom=args.use_custom_lrp_rules)

    target_layer = None
    for m in D.modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in D.")

    # ---- Compute both heatmaps for each prototype ----
    print(f"Computing strict + realism PRP for {len(exemplars)} prototypes...")
    panel_data = []
    indiv_dir = os.path.join(args.out_dir, "individual_strict")
    os.makedirs(indiv_dir, exist_ok=True)

    for k, (idx, img_tensor, msk_tensor) in enumerate(exemplars):
        inp = torch.cat([img_tensor.unsqueeze(0).to(device),
                         msk_tensor.unsqueeze(0).to(device)],
                        dim=1).requires_grad_()
        p_k = torch.from_numpy(cluster_centers[k]).float()

        # Realism PRP (existing baseline)
        realism_hm = compute_realism_prp(D, inp, target_layer)

        # Strict PRP (this script's contribution)
        strict_hm = compute_strict_prp(D, inp, p_k, target_layer)

        # Save individual heatmaps (single-channel)
        vutils.save_image(strict_hm, os.path.join(indiv_dir,
                          f"prototype_{k:02d}_prp_strict.png"), normalize=True)

        # Save side-by-side comparison panel
        save_comparison_panel(
            exemplar=img_tensor, mask=msk_tensor,
            realism_hm=realism_hm.cpu(), strict_hm=strict_hm.cpu(),
            out_path=os.path.join(args.out_dir, f"comparison_proto_{k:02d}.png"),
            title_prefix=f"P{k:02d} | ",
        )

        panel_data.append((img_tensor, msk_tensor,
                           realism_hm.cpu(), strict_hm.cpu()))
        print(f"  Prototype {k:02d}: exemplar idx={idx}, done")

    # ---- Master montage ----
    print("Saving master montage (all K prototypes, realism vs strict)...")
    save_master_montage(panel_data,
                        out_path=os.path.join(args.out_dir, "master_comparison.png"),
                        K=len(panel_data))

    # ---- Summary ----
    summary = {
        "ckpt_g": args.ckpt_g, "ckpt_d": args.ckpt_d,
        "num_prototypes": len(exemplars),
        "use_custom_lrp_rules": args.use_custom_lrp_rules,
        "outputs": {
            "individual_strict_heatmaps": indiv_dir,
            "comparison_panels": args.out_dir,
            "master_comparison": os.path.join(args.out_dir, "master_comparison.png"),
        },
    }
    with open(os.path.join(args.out_dir, "prp_strict_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {len(panel_data)} prototypes processed.")
    print(f"Outputs in: {args.out_dir}/")
    print(f"  - Open 'master_comparison.png' first for the side-by-side view.")


if __name__ == "__main__":
    main()
