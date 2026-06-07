"""
baseline_conditional.py
=======================
Conditional baselines for the IEEE Access revision of:
  "Self-Explainable WGAN-GP for Brain Tumor MRI Synthesis Using KMEx Prototype Learning"

This file is a strict modification of the original baseline.py: it adds tumor-mask
conditioning to DCGAN, WGAN-GP (plain), and SAGAN-lite so that all three baselines
receive the same input information as the proposed model in v17.py.

Conditioning mechanism (matches v17.py exactly):
  * MaskEncoder: 3 conv blocks + global avg pool + linear -> [B, nz] mask embedding
  * Generator(z, m): z' = z + proj(MaskEncoder(m)); deconv backbone produces image
  * Discriminator: input is concat([image, mask]) along channel dim, so 3+3 = 6 channels
  * Optimizer: TTUR (g_lr=1e-4, d_lr=2e-5), betas=(0.0, 0.99), critic_steps=2 for WGAN variants
  * DCGAN keeps its canonical training regime (BCE loss, lr=2e-4, betas=(0.5, 0.999))
  * Inputs: 3-channel RGB overlays (image with tumor-region color blend), 256x256

The proposed model (v17.py) and all three baselines are now trained with:
  * IDENTICAL dataset (H5OverlayDataset over BraTS2023 .h5 files)
  * IDENTICAL conditioning mechanism (MaskEncoder + concat in D)
  * IDENTICAL image resolution, channel count, and batch size
  * IDENTICAL number of training epochs (default 3000 to match the proposed model)

The ONLY differences between baselines and the proposed model are now:
  * Loss formulation (BCE vs. WGAN-GP)
  * Use of self-attention (SAGAN-lite only)
  * Absence of prototype loss + KMEx mining + PRP (these are the proposed contributions)

Usage:
  python baseline_conditional.py \\
      --data_dir /path/to/BraTS2023/h5/files \\
      --out_root ./baselines_cond_out \\
      --epochs 3000 \\
      --batch_size 16 \\
      --seed 0

  # Optionally pre-build the overlay cache (recommended on first run):
  python baseline_conditional.py --data_dir /path/to/h5 --overlay_cache ./overlay_train \\
      --precompute_only

  # Run only one baseline (for selective re-training):
  python baseline_conditional.py --data_dir ... --models wgangp

Author: [your name]
License: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as vutils


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """Set seeds for all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic conv kernels (slight speed cost, large reproducibility gain)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Overlay utilities (RGB overlays from image + mask)
# Copied verbatim from v17.py so the dataset pipeline is bit-identical.
# =============================================================================

PALETTE = {
    0: (0, 0, 0),          # background
    1: (190, 190, 190),    # edema / whole tumor (light gray)
    2: (220, 50, 50),      # enhancing tumor (red)
    3: (255, 170, 60),     # core / necrosis (orange)
}


def _to_uint8_im(x: np.ndarray) -> np.ndarray:
    """Robust percentile-based normalization to uint8 [0, 255]."""
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(x.min())
        hi = float(x.max()) if x.max() > x.min() else 1.0
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def make_overlay_rgb(gray2d: np.ndarray, mask2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend tumor-class colors onto a grayscale MRI slice to produce an RGB overlay.

    Args:
        gray2d: 2D grayscale MRI slice (any float range).
        mask2d: 2D integer label map with values in {0, 1, 2, 3}.
        alpha:  Blending strength for the color overlay.

    Returns:
        H x W x 3 uint8 RGB image.
    """
    base = _to_uint8_im(gray2d)
    H, W = base.shape
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    color_mask = np.zeros((H, W, 3), dtype=np.float32)
    for cls, color in PALETTE.items():
        if cls == 0:
            continue
        m = (mask2d == cls)
        if m.any():
            color_mask[m] = np.array(color, dtype=np.float32)

    blended = (1.0 - alpha) * rgb + alpha * color_mask
    bg = (mask2d == 0) & (base < 4)
    blended[bg] = 0
    return blended.clip(0, 255).astype(np.uint8)


# =============================================================================
# Dataset (H5 -> RGB overlay + 3-channel mask)
# Mirrors H5OverlayDataset from v17.py to guarantee identical inputs.
# =============================================================================

class H5OverlayDataset(Dataset):
    """Loads BraTS2023 .h5 slice files and yields (image, mask) tensors in [-1, 1].

    Each .h5 file is expected to expose:
        f['image']: array shaped (H, W, >=3); channel 0 used as grayscale base.
        f['mask']:  array shaped (H, W, 3); one-hot per tumor class (WT/ET/TC).

    Returns:
        x: FloatTensor [3, img_size, img_size] in [-1, 1]  (RGB overlay).
        m: FloatTensor [3, img_size, img_size] in [-1, 1]  (mask channels).
    """

    def __init__(
        self,
        root: str,
        img_size: int = 256,
        overlay_alpha: float = 0.45,
        overlay_cache: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.paths = sorted(str(p) for p in self.root.rglob("*.h5"))
        if not self.paths:
            raise FileNotFoundError(f"No .h5 files found under: {root}")
        self.img_size = img_size
        self.alpha = overlay_alpha
        self.cache = Path(overlay_cache) if overlay_cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_paths(self, h5_path: str) -> Tuple[Path, Path]:
        stem = Path(h5_path).stem
        return self.cache / f"{stem}.png", self.cache / f"{stem}.npy"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h5_path = self.paths[idx]

        # Cache hit path
        if self.cache:
            img_p, msk_p = self._cache_paths(h5_path)
            if img_p.exists() and msk_p.exists():
                img = Image.open(img_p).convert("RGB")
                m_arr = np.load(msk_p)
            else:
                img, m_arr = self._build(h5_path)
                img.save(img_p)
                np.save(msk_p, m_arr)
        else:
            img, m_arr = self._build(h5_path)

        # To tensor, resize, normalize
        img = img.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        x = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 127.5 - 1.0
        m = torch.from_numpy(m_arr).permute(2, 0, 1)
        m = F.interpolate(m.unsqueeze(0), size=(self.img_size, self.img_size),
                          mode="nearest").squeeze(0)
        m = m * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return x, m

    @staticmethod
    def _build(h5_path: str) -> Tuple[Image.Image, np.ndarray]:
        """Read one .h5 file and build the RGB overlay PIL image + 3-channel mask array."""
        with h5py.File(h5_path, "r") as f:
            vol = np.array(f["image"])
            msk = np.array(f["mask"])

        gray = vol[..., 0] if (vol.ndim == 3 and vol.shape[-1] >= 1) else vol.squeeze()
        if msk.ndim == 3 and msk.shape[-1] == 3:
            label = (msk.argmax(axis=-1) + 1) * (msk.max(axis=-1) > 0)
            m_arr = msk.astype("float32")
        else:
            label = msk
            m_arr = np.stack(
                [(label == 1).astype("float32"),
                 (label == 2).astype("float32"),
                 (label == 3).astype("float32")],
                axis=-1,
            )
        overlay = make_overlay_rgb(gray, label, alpha=0.45)
        return Image.fromarray(overlay), m_arr


# =============================================================================
# Shared building blocks
# =============================================================================

class MaskEncoder(nn.Module):
    """Tiny CNN that encodes a 3-channel mask into a [B, nz] embedding.

    Architecture identical to v17.py:
        Conv(3->32, k=3, s=2) -> ReLU
        Conv(32->64, k=3, s=2) -> ReLU
        Conv(64->128, k=3, s=2) -> ReLU
        AdaptiveAvgPool2d(1)
        Linear(128 -> nz)
    """

    def __init__(self, in_ch: int = 3, nz: int = 128) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, nz)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        h = self.enc(m).view(m.size(0), -1)
        return self.fc(h)


class SelfAttention(nn.Module):
    """SAGAN-style self-attention layer (Zhang et al., 2019)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 8, 1, bias=False)
        self.phi   = nn.Conv2d(in_channels, in_channels // 8, 1, bias=False)
        self.g     = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
        self.o     = nn.Conv2d(in_channels // 2, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, C // 8, -1)
        phi   = self.phi(x).view(B, C // 8, -1)
        attn  = torch.softmax(theta.transpose(1, 2).bmm(phi), -1)
        g     = self.g(x).view(B, C // 2, -1)
        o     = g.bmm(attn.transpose(1, 2)).view(B, C // 2, H, W)
        o     = self.o(o)
        return x + self.gamma * o


def weights_init_normal(m: nn.Module) -> None:
    """DCGAN-style normal weight initialization."""
    name = m.__class__.__name__.lower()
    if "conv" in name or "linear" in name:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "batchnorm" in name or "instancenorm" in name:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0)


# =============================================================================
# Conditional Generators
# =============================================================================

def _make_g_backbone(nz: int, nc: int, ngf: int, img_size: int,
                     use_attention: bool = False) -> nn.Sequential:
    """Build a DCGAN/WGAN-style transposed-conv backbone (z 1x1 -> image)."""
    layers = []
    s = 4
    c = ngf * 8
    layers += [
        nn.ConvTranspose2d(nz, c, 4, 1, 0, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(True),
    ]
    attention_inserted = False
    while s < img_size // 2:
        next_c = max(c // 2, ngf)
        layers += [
            nn.ConvTranspose2d(c, next_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(next_c),
            nn.ReLU(True),
        ]
        c = next_c
        s *= 2
        if use_attention and (s == 64) and not attention_inserted:
            layers.append(SelfAttention(c))
            attention_inserted = True
    layers += [
        nn.ConvTranspose2d(c, nc, 4, 2, 1, bias=False),
        nn.Tanh(),
    ]
    return nn.Sequential(*layers)


class CondGenerator(nn.Module):
    """Shared conditional Generator wrapper for DCGAN / WGAN-GP / SAGAN-lite.

    Forward: G(z, m) where z is [B, nz] noise and m is [B, mask_nc, H, W].
    """

    def __init__(
        self,
        nz: int = 128,
        nc: int = 3,
        mask_nc: int = 3,
        ngf: int = 64,
        img_size: int = 256,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.nz = nz
        self.backbone = _make_g_backbone(nz, nc, ngf, img_size, use_attention)
        self.menc = MaskEncoder(in_ch=mask_nc, nz=nz)
        self.proj = nn.Linear(nz, nz, bias=False)

    def forward(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Mirror v17.py: add projected mask embedding to z, then deconv backbone
        zc = self.proj(self.menc(m))           # [B, nz]
        # Reshape z to [B, nz, 1, 1] if it came in flat
        if z.dim() == 2:
            z = z.view(z.size(0), self.nz, 1, 1)
        z = z + zc.unsqueeze(-1).unsqueeze(-1)
        return self.backbone(z)


# =============================================================================
# Conditional Discriminators
# =============================================================================

class CondDiscriminatorDCGAN(nn.Module):
    """Conditional DCGAN discriminator. Input is concat([image, mask])."""

    def __init__(
        self,
        nc: int = 3,
        mask_nc: int = 3,
        ndf: int = 64,
        img_size: int = 256,
    ) -> None:
        super().__init__()
        c_in = nc + mask_nc
        layers = [
            nn.Conv2d(c_in, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        c = ndf
        s = img_size // 2
        while s > 4:
            layers += [
                nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(c * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c *= 2
            s //= 2
        layers += [nn.Conv2d(c, 1, 4, 1, 0, bias=False)]
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.main(torch.cat([x, m], dim=1)).view(-1)


class CondDiscriminatorWGAN(nn.Module):
    """Conditional WGAN-GP discriminator (with optional self-attention)."""

    def __init__(
        self,
        nc: int = 3,
        mask_nc: int = 3,
        ndf: int = 64,
        img_size: int = 256,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        c_in = nc + mask_nc
        layers = [
            nn.Conv2d(c_in, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        c = ndf
        s = img_size // 2
        attention_inserted = False
        while s > 4:
            layers += [
                nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(c * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c *= 2
            s //= 2
            if use_attention and (s == 32) and not attention_inserted:
                layers.append(SelfAttention(c))
                attention_inserted = True
        layers += [nn.Conv2d(c, 1, 4, 1, 0, bias=False)]
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.main(torch.cat([x, m], dim=1)).view(-1)


# =============================================================================
# Gradient penalty for WGAN-GP variants
# =============================================================================

def gradient_penalty_cond(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Conditional gradient penalty (Gulrajani et al., 2017), with mask held fixed."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_inter = D(inter, mask)
    grad = autograd.grad(
        outputs=d_inter.sum(),
        inputs=inter,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(real.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


# =============================================================================
# Sample / checkpoint helpers
# =============================================================================

def save_sample_grid(samples: torch.Tensor, path: str, nrow: int = 8) -> None:
    """Save a grid of generator outputs to disk, clamped to [0, 1]."""
    imgs = (samples.detach().cpu() + 1) / 2
    imgs = imgs.clamp(0, 1)
    vutils.save_image(imgs, path, nrow=nrow, normalize=False)


def save_checkpoint(G: nn.Module, D: nn.Module, out_dir: str, tag: str) -> None:
    """Save G and D state dicts under `out_dir/{G,D}_{tag}.pt`."""
    torch.save(G.state_dict(), os.path.join(out_dir, f"G_{tag}.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, f"D_{tag}.pt"))


# =============================================================================
# Training loops
# =============================================================================

def _get_fixed_eval_batch(
    loader: DataLoader, sample_n: int, mask_nc: int, img_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Grab a fixed conditioning batch from the loader for periodic sample saves."""
    try:
        masks = next(iter(loader))[1].to(device)
    except StopIteration:
        masks = torch.zeros(0, mask_nc, img_size, img_size, device=device)
    if masks.size(0) >= sample_n:
        return masks[:sample_n]
    base = masks if masks.size(0) > 0 else \
        torch.zeros(1, mask_nc, img_size, img_size, device=device)
    reps = (sample_n + base.size(0) - 1) // base.size(0)
    return base.repeat(reps, 1, 1, 1)[:sample_n]


def train_wgangp_cond(
    dataloader: DataLoader,
    out_dir: str,
    device: torch.device,
    img_size: int,
    nc: int,
    mask_nc: int,
    epochs: int = 3000,
    nz: int = 128,
    n_critic: int = 2,
    gp_lambda: float = 10.0,
    g_lr: float = 1e-4,
    d_lr: float = 2e-5,
    betas: Tuple[float, float] = (0.0, 0.99),
    use_attention: bool = False,
    seed: int = 0,
    sample_every: int = 100,
    ckpt_every: int = 500,
    resume_g: Optional[str] = None,
    resume_d: Optional[str] = None,
    start_epoch: int = 0,
) -> dict:
    """Train a conditional WGAN-GP (plain) or SAGAN-lite (use_attention=True).

    Args (resume): If resume_g and resume_d paths are given, load those state dicts
    before training. Loop iterates over range(start_epoch + 1, epochs + 1).
    The training log file is appended (not truncated) when start_epoch > 0.
    """
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    G = CondGenerator(nz=nz, nc=nc, mask_nc=mask_nc, ngf=64,
                      img_size=img_size, use_attention=use_attention).to(device)
    D = CondDiscriminatorWGAN(nc=nc, mask_nc=mask_nc, ndf=64,
                              img_size=img_size, use_attention=use_attention).to(device)
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    if resume_g and resume_d:
        print(f"Resuming WGAN-GP-cond from {resume_g} and {resume_d}")
        G.load_state_dict(torch.load(resume_g, map_location=device))
        D.load_state_dict(torch.load(resume_d, map_location=device))

    opt_G = torch.optim.Adam(G.parameters(), lr=g_lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=betas)

    fixed_z = torch.randn(64, nz, device=device)
    fixed_m = _get_fixed_eval_batch(dataloader, 64, mask_nc, img_size, device)

    log_path = os.path.join(out_dir, "train_log.jsonl")
    # Append mode if resuming; truncate if starting fresh
    open(log_path, "a" if start_epoch > 0 else "w").close()
    t0 = time.time()

    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            G.train(); D.train()
            for real, mask in dataloader:
                real, mask = real.to(device), mask.to(device)
                bsz = real.size(0)

                # ---- Critic steps ----
                for _ in range(n_critic):
                    z = torch.randn(bsz, nz, device=device)
                    with torch.no_grad():
                        fake = G(z, mask)
                    d_real = D(real, mask)
                    d_fake = D(fake, mask)
                    gp = gradient_penalty_cond(D, real, fake, mask, device)
                    loss_D = (d_fake - d_real).mean() + gp_lambda * gp
                    opt_D.zero_grad(set_to_none=True)
                    loss_D.backward()
                    opt_D.step()

                # ---- Generator step ----
                z = torch.randn(bsz, nz, device=device)
                fake = G(z, mask)
                loss_G = -D(fake, mask).mean()
                opt_G.zero_grad(set_to_none=True)
                loss_G.backward()
                opt_G.step()

            # ---- Logging ----
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "loss_D": float(loss_D.item()),
                    "loss_G": float(loss_G.item()),
                    "gp": float(gp.item()),
                    "time_min": (time.time() - t0) / 60.0,
                }) + "\n")

            # ---- Periodic sample save ----
            if (epoch % sample_every == 0) or epoch == 1 or epoch == epochs:
                G.eval()
                with torch.no_grad():
                    samples = G(fixed_z, fixed_m)
                save_sample_grid(samples, os.path.join(out_dir, f"samples_epoch_{epoch:04d}.png"))

            # ---- Periodic checkpoint ----
            if (epoch % ckpt_every == 0) or epoch == epochs:
                save_checkpoint(G, D, out_dir, tag=f"epoch_{epoch:04d}")

    except (KeyboardInterrupt, Exception) as e:
        # Emergency-save the current weights so a crash doesn't lose progress.
        # 'epoch' is the loop variable; if it's defined we record where we died.
        last_epoch = locals().get("epoch", start_epoch)
        emergency_tag = f"emergency_epoch_{last_epoch:04d}"
        print(f"[Emergency] Caught {type(e).__name__}: {e}. "
              f"Saving emergency checkpoint as {emergency_tag} and re-raising.")
        save_checkpoint(G, D, out_dir, tag=emergency_tag)
        raise

    # Final samples + checkpoint
    G.eval()
    with torch.no_grad():
        z = torch.randn(64, nz, device=device)
        samples = G(z, fixed_m)
    save_sample_grid(samples, os.path.join(out_dir, "samples_final.png"))
    save_checkpoint(G, D, out_dir, tag="final")

    return {"G": G, "D": D, "log_path": log_path}


def train_dcgan_cond(
    dataloader: DataLoader,
    out_dir: str,
    device: torch.device,
    img_size: int,
    nc: int,
    mask_nc: int,
    epochs: int = 3000,
    nz: int = 128,
    lr: float = 2e-4,
    betas: Tuple[float, float] = (0.5, 0.999),
    seed: int = 0,
    sample_every: int = 100,
    ckpt_every: int = 500,
    resume_g: Optional[str] = None,
    resume_d: Optional[str] = None,
    start_epoch: int = 0,
) -> dict:
    """Train a conditional DCGAN with standard BCE-with-logits adversarial loss.

    Resume support: pass resume_g, resume_d, start_epoch to continue from a checkpoint.
    """
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    G = CondGenerator(nz=nz, nc=nc, mask_nc=mask_nc, ngf=64,
                      img_size=img_size, use_attention=False).to(device)
    D = CondDiscriminatorDCGAN(nc=nc, mask_nc=mask_nc, ndf=64,
                               img_size=img_size).to(device)
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    if resume_g and resume_d:
        print(f"Resuming DCGAN-cond from {resume_g} and {resume_d}")
        G.load_state_dict(torch.load(resume_g, map_location=device))
        D.load_state_dict(torch.load(resume_d, map_location=device))

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, nz, device=device)
    fixed_m = _get_fixed_eval_batch(dataloader, 64, mask_nc, img_size, device)

    log_path = os.path.join(out_dir, "train_log.jsonl")
    open(log_path, "a" if start_epoch > 0 else "w").close()
    t0 = time.time()

    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            G.train(); D.train()
            for real, mask in dataloader:
                real, mask = real.to(device), mask.to(device)
                bsz = real.size(0)
                z = torch.randn(bsz, nz, device=device)
                fake = G(z, mask)

                # ---- D step ----
                logits_real = D(real, mask)
                logits_fake = D(fake.detach(), mask)
                loss_D = bce(logits_real, torch.ones_like(logits_real)) + \
                         bce(logits_fake, torch.zeros_like(logits_fake))
                opt_D.zero_grad(set_to_none=True)
                loss_D.backward()
                opt_D.step()

                # ---- G step ----
                logits_fake = D(fake, mask)
                loss_G = bce(logits_fake, torch.ones_like(logits_fake))
                opt_G.zero_grad(set_to_none=True)
                loss_G.backward()
                opt_G.step()

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "loss_D": float(loss_D.item()),
                    "loss_G": float(loss_G.item()),
                    "time_min": (time.time() - t0) / 60.0,
                }) + "\n")

            if (epoch % sample_every == 0) or epoch == 1 or epoch == epochs:
                G.eval()
                with torch.no_grad():
                    samples = G(fixed_z, fixed_m)
                save_sample_grid(samples, os.path.join(out_dir, f"samples_epoch_{epoch:04d}.png"))

            if (epoch % ckpt_every == 0) or epoch == epochs:
                save_checkpoint(G, D, out_dir, tag=f"epoch_{epoch:04d}")

    except (KeyboardInterrupt, Exception) as e:
        last_epoch = locals().get("epoch", start_epoch)
        emergency_tag = f"emergency_epoch_{last_epoch:04d}"
        print(f"[Emergency] Caught {type(e).__name__}: {e}. "
              f"Saving emergency checkpoint as {emergency_tag} and re-raising.")
        save_checkpoint(G, D, out_dir, tag=emergency_tag)
        raise

    G.eval()
    with torch.no_grad():
        z = torch.randn(64, nz, device=device)
        samples = G(z, fixed_m)
    save_sample_grid(samples, os.path.join(out_dir, "samples_final.png"))
    save_checkpoint(G, D, out_dir, tag="final")

    return {"G": G, "D": D, "log_path": log_path}


# =============================================================================
# Precompute-only mode: build overlay cache without training
# =============================================================================

def precompute_overlays(data_dir: str, cache_dir: str, img_size: int) -> None:
    """Iterate the dataset once to materialize every overlay image + mask in the cache."""
    print(f"Precomputing overlay cache at {cache_dir} from {data_dir} ...")
    ds = H5OverlayDataset(root=data_dir, img_size=img_size, overlay_cache=cache_dir)
    for i in range(len(ds)):
        _ = ds[i]
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} cached")
    print("Overlay cache complete.")


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conditional baselines (DCGAN, WGAN-GP plain, SAGAN-lite) "
                    "matched to v17.py's conditioning for fair comparison."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder containing BraTS2023 .h5 files.")
    parser.add_argument("--out_root", type=str, default="./baselines_cond_out",
                        help="Root output folder for all three baseline runs.")
    parser.add_argument("--overlay_cache", type=str, default=None,
                        help="Folder for cached overlay PNGs + .npy masks. "
                             "Strongly recommended for multi-run efficiency.")
    parser.add_argument("--precompute_only", action="store_true",
                        help="Build the overlay cache from .h5 files and exit "
                             "(no training).")
    parser.add_argument("--models", type=str, default="all",
                        choices=["all", "dcgan", "wgangp", "sagan"],
                        help="Which baseline(s) to train.")

    # Hyperparameters
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--nc", type=int, default=3, help="Image channels.")
    parser.add_argument("--mask_nc", type=int, default=3, help="Mask channels.")
    parser.add_argument("--nz", type=int, default=128, help="Latent dimensionality.")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_critic", type=int, default=2,
                        help="Critic steps per generator step (WGAN variants).")
    parser.add_argument("--gp_lambda", type=float, default=10.0)
    parser.add_argument("--g_lr", type=float, default=1e-4,
                        help="WGAN variants generator learning rate.")
    parser.add_argument("--d_lr", type=float, default=2e-5,
                        help="WGAN variants discriminator learning rate (TTUR).")
    parser.add_argument("--wgan_beta1", type=float, default=0.0)
    parser.add_argument("--wgan_beta2", type=float, default=0.99)
    parser.add_argument("--dcgan_lr", type=float, default=2e-4)
    parser.add_argument("--dcgan_beta1", type=float, default=0.5)
    parser.add_argument("--dcgan_beta2", type=float, default=0.999)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers. Default 0 (safest; multiprocessing "
                             "workers can crash pin_memory threads on long runs). "
                             "Set >0 only if you've verified stability.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable CUDA pin_memory. Off by default for stability.")
    parser.add_argument("--resume_g", type=str, default=None,
                        help="Path to Generator state_dict to resume from.")
    parser.add_argument("--resume_d", type=str, default=None,
                        help="Path to Discriminator state_dict to resume from.")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch to resume from. Loop runs (start_epoch+1) .. epochs.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # ---- Precompute mode short-circuit ----
    if args.precompute_only:
        if args.overlay_cache is None:
            raise ValueError("--precompute_only requires --overlay_cache to be set.")
        precompute_overlays(args.data_dir, args.overlay_cache, args.img_size)
        return

    # ---- Set up output directory and dataset ----
    os.makedirs(args.out_root, exist_ok=True)
    dataset = H5OverlayDataset(
        root=args.data_dir,
        img_size=args.img_size,
        overlay_cache=args.overlay_cache,
    )

    pin_memory = args.pin_memory and args.device.startswith("cuda") and torch.cuda.is_available()
    # In containerized environments /dev/shm is often tiny (64 MB), which causes
    # DataLoader workers to crash with "out of shared memory" when using the
    # default 'fork' multiprocessing context with shared-memory tensor transfer.
    # We switch to the 'file_system' sharing strategy which uses regular files
    # under /tmp instead of POSIX shared memory. The cost is a marginal slowdown
    # in tensor transfer; the benefit is that workers actually work.
    if args.num_workers > 0:
        import torch.multiprocessing as mp_t
        try:
            mp_t.set_sharing_strategy("file_system")
        except Exception:
            pass
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    device = torch.device(args.device)

    # ---- Write manifest ----
    manifest = {
        "data_dir": args.data_dir,
        "out_root": args.out_root,
        "img_size": args.img_size,
        "nc": args.nc,
        "mask_nc": args.mask_nc,
        "nz": args.nz,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_critic": args.n_critic,
        "gp_lambda": args.gp_lambda,
        "g_lr": args.g_lr,
        "d_lr": args.d_lr,
        "wgan_betas": [args.wgan_beta1, args.wgan_beta2],
        "dcgan_lr": args.dcgan_lr,
        "dcgan_betas": [args.dcgan_beta1, args.dcgan_beta2],
        "seed": args.seed,
        "device": args.device,
        "models": args.models,
    }
    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {args.out_root}/manifest.json")
    print(f"Dataset size: {len(dataset)} samples")

    # ---- Run the requested baselines ----
    runs = []
    if args.models in ("all", "dcgan"):
        runs.append(("dcgan", "DCGAN-cond"))
    if args.models in ("all", "wgangp"):
        runs.append(("wgangp_plain", "WGAN-GP-cond"))
    if args.models in ("all", "sagan"):
        runs.append(("sagan_lite", "SAGAN-lite-cond"))

    for i, (folder, name) in enumerate(runs, start=1):
        out_dir = os.path.join(args.out_root, folder)
        print(f"\n[{i}/{len(runs)}] Training {name} -> {out_dir}")

        # Resume arguments only apply to the FIRST run requested in this invocation.
        # If you need to resume a later model (e.g. only wgangp), invoke with --models
        # wgangp specifically rather than --models all.
        is_first = (i == 1)
        rg = args.resume_g if is_first else None
        rd = args.resume_d if is_first else None
        se = args.start_epoch if is_first else 0

        if folder == "dcgan":
            train_dcgan_cond(
                dataloader=dataloader,
                out_dir=out_dir,
                device=device,
                img_size=args.img_size,
                nc=args.nc,
                mask_nc=args.mask_nc,
                epochs=args.epochs,
                nz=args.nz,
                lr=args.dcgan_lr,
                betas=(args.dcgan_beta1, args.dcgan_beta2),
                seed=args.seed,
                sample_every=args.sample_every,
                ckpt_every=args.ckpt_every,
                resume_g=rg,
                resume_d=rd,
                start_epoch=se,
            )
        else:
            train_wgangp_cond(
                dataloader=dataloader,
                out_dir=out_dir,
                device=device,
                img_size=args.img_size,
                nc=args.nc,
                mask_nc=args.mask_nc,
                epochs=args.epochs,
                nz=args.nz,
                n_critic=args.n_critic,
                gp_lambda=args.gp_lambda,
                g_lr=args.g_lr,
                d_lr=args.d_lr,
                betas=(args.wgan_beta1, args.wgan_beta2),
                use_attention=(folder == "sagan_lite"),
                seed=args.seed,
                sample_every=args.sample_every,
                ckpt_every=args.ckpt_every,
                resume_g=rg,
                resume_d=rd,
                start_epoch=se,
            )

    print(f"\nAll runs complete. Outputs under: {args.out_root}")


if __name__ == "__main__":
    main()
