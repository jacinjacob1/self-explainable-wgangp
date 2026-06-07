# =========================
# Self‑Explainable WGAN‑GP + KMEx for Brain‑MRI Overlays (256×256, 2‑channel (FLAIR+T1Gd))
# -------------------------------------------------------------
# Full pipeline:
#   1. WGAN‑GP training on flat MRI overlay images
#   2. KMEx prototype mining (post‑hoc self‑explanation)
#   3. PRP heat‑map generation for transparency
#   4. Diversity metrics logging
# -------------------------------------------------------------
# Quick start (macOS/Linux):
#   $ conda create -n xgan python=3.10 -y && conda activate xgan
#   $ pip install torch torchvision torchaudio scikit-learn captum tqdm scipy matplotlib
#   $ python wgan_kmex_brain_mri.py --epochs 250 --data_dir ~/Downloads/train
# -------------------------------------------------------------

import os, math, random, argparse, json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torch import autograd
from torchvision import transforms, utils as vutils
import h5py
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances_argmin_min
from captum.attr import LayerLRP
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
import torchvision.transforms.functional as TF

def compute_mmis_score(generator, nz, device, loader, mask_nc, H, num_samples=64):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, nz, 1, 1, device=device)
        # Prepare masks to match num_samples
        try:
            m = next(iter(loader))[1].to(device)
        except StopIteration:
            m = torch.zeros(0, mask_nc, H, H, device=device)
        if m.size(0) >= num_samples:
            m = m[:num_samples]
        else:
            base = m if m.size(0) > 0 else torch.zeros(1, mask_nc, H, H, device=device)
            reps = (num_samples + base.size(0) - 1) // base.size(0)
            m = base.repeat(reps, 1, 1, 1)[:num_samples]

        fake_imgs = generator(z, m)
        fake_imgs = (fake_imgs + 1) / 2  # Scale to [0, 1] for MS-SSIM

        scores = []
        for i in range(0, num_samples - 1, 2):
            ssim = ms_ssim(fake_imgs[i].unsqueeze(0), fake_imgs[i+1].unsqueeze(0), data_range=1.0, size_average=True)
            scores.append(ssim.item())
        return sum(scores) / len(scores) if scores else 0.0
    
# ----------------------------- CLI -----------------------------

def get_args():
    p = argparse.ArgumentParser("Self‑Explainable WGAN‑GP (KMEx)")
    p.add_argument("--data_dir", required=True, help="Folder with PNG/JPG MRI overlay slices")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--nc", type=int, default=2)
    p.add_argument("--nz", type=int, default=128)
    p.add_argument("--gf", type=int, default=64)
    p.add_argument("--df", type=int, default=64)
    p.add_argument("--g_lr", type=float, default=1e-4)
    p.add_argument("--d_lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--gp_lambda", type=float, default=10.0)
    p.add_argument("--num_prototypes", type=int, default=16)
    p.add_argument("--kmex_refit_every", type=int, default=100,
                   help="Refit anchor matrix via K-Means every N epochs. "
                        "0 disables inline refitting (fully post-hoc / original v17 behavior). "
                        "Default 100 enables EM-style inline KMEx.")
    p.add_argument("--kmex_ema_alpha", type=float, default=0.9,
                   help="EMA weight for OLD anchors during refit. "
                        "0.9 = smooth update (recommended); "
                        "1.0 = freeze anchors; "
                        "0.0 = hard replace.")
    p.add_argument("--save_every", type=int, default=500,
                   help="Save G and D checkpoints every N epochs. "
                        "Always also saves at the final epoch. "
                        "0 disables periodic saves (final only).")
    p.add_argument("--resume_g", type=str, default=None,
                   help="Path to G state_dict to resume from. "
                        "Pair with --resume_d and --start_epoch.")
    p.add_argument("--resume_d", type=str, default=None,
                   help="Path to D state_dict to resume from. "
                        "D.prototypes anchor matrix is included in D's state dict, "
                        "so the anchors recover their epoch-N values exactly.")
    p.add_argument("--start_epoch", type=int, default=0,
                   help="Epoch to resume from. Training loop runs "
                        "(start_epoch + 1) .. epochs. Default 0 = train from scratch.")
    p.add_argument("--out_dir", default="outv16", help="Output directory for generated images and prototypes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_nc", type=int, default=3, help="Number of mask channels (conditioning)")
    p.add_argument("--lambda_proto", type=float, default=0.05, help="Prototype pull strength for G")
    p.add_argument("--proto_margin", type=float, default=1.0, help="Prototype separation margin")
    p.add_argument("--critic_steps", type=int, default=2, help="Number of D steps per G step")
    p.add_argument("--use_overlays", action="store_true", help="Create color overlays (image+mask) and train on them (RGB, nc=3).")
    p.add_argument("--overlay_cache", type=str, default=None, help="If set, save/serve overlays from this folder.")
    p.add_argument("--overlay_alpha", type=float, default=0.45, help="Alpha for blending mask colors onto the grayscale base.")
    p.add_argument("--precompute_only", action="store_true",
                   help="Generate overlay PNGs from all .h5 files and exit without training")
    return p.parse_args()


# ----------------------- Utilities -----------------------------
# ----------------------- Utilities -----------------------------
def save_two_channel_preview(tensor_bchw, path_prefix):
    """
    Save a preview of a 2‑channel tensor batch in [-1,1].
    Writes: {path_prefix}_flair.png and {path_prefix}_t1gd.png using channel 0 and 1.
    """
    import torchvision.utils as vutils
    import torch
    # tensor_bchw: [B,2,H,W]
    x = (tensor_bchw + 1) / 2  # back to [0,1]
    vutils.save_image(x[:,0:1], f"{path_prefix}_flair.png", normalize=False)
    vutils.save_image(x[:,1:2], f"{path_prefix}_t1gd.png", normalize=False)

# ================= Overlay utilities (RGB overlays from image+mask) =================
PALETTE = {
    0: (0, 0, 0),          # background
    1: (190, 190, 190),    # edema/WT (light gray)
    2: (220, 50, 50),      # enhancing tumor (red)
    3: (255, 170, 60),     # core/necrosis (orange)
}

def _to_uint8_im(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max()) if x.max() > x.min() else (0.0, 1.0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def make_overlay_rgb(gray2d: np.ndarray, mask2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
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

class H5OverlayDataset(Dataset):
    def __init__(self, root: str, img_size: int = 256, overlay_alpha: float = 0.45, overlay_cache: str | None = None):
        self.root = Path(root)
        self.paths = sorted([str(p) for p in self.root.rglob('*.h5')])
        self.img_size = img_size
        self.alpha = overlay_alpha
        self.cache = Path(overlay_cache) if overlay_cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_path(self, h5_path: str) -> Path:
        return self.cache / (Path(h5_path).stem + '.png') if self.cache else Path()

    def __getitem__(self, idx: int):
        import numpy as np
        h5_path = self.paths[idx]
        cache_img_path = self._cache_path(h5_path) if self.cache else None
        cache_mask_path = cache_img_path.with_suffix('.npy') if (self.cache and cache_img_path) else None
        if self.cache and cache_img_path.exists():
            img = Image.open(cache_img_path).convert('RGB')
            # Try to load cached mask .npy file
            if cache_mask_path.exists():
                msk = np.load(cache_mask_path)
                m = torch.from_numpy(msk.astype('float32')).permute(2,0,1)
            else:
                # Mask file missing in cache; raise error or fallback
                raise RuntimeError(f"Cached mask file not found: {cache_mask_path}")
        else:
            with h5py.File(h5_path, 'r') as f:
                vol = np.array(f['image'])
                msk = np.array(f['mask'])
            gray = vol[..., 0] if (vol.ndim == 3 and vol.shape[-1] >= 1) else vol.squeeze()
            if msk.ndim == 3 and msk.shape[-1] == 3:
                label = (msk.argmax(axis=-1) + 1) * (msk.max(axis=-1) > 0)
            else:
                label = msk
            overlay = make_overlay_rgb(gray, label, alpha=self.alpha)
            img = Image.fromarray(overlay)
            if msk.ndim == 3 and msk.shape[-1] == 3:
                m_arr = msk.astype('float32')
            else:
                m_arr = np.stack([(label==1).astype('float32'), (label==2).astype('float32'), (label==3).astype('float32')], axis=-1)
            m = torch.from_numpy(m_arr).permute(2,0,1)
            if self.cache:
                img.save(cache_img_path)
                # Save mask as .npy in cache
                np.save(cache_mask_path, m_arr)

        img = img.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        x = torch.from_numpy(np.array(img).transpose(2,0,1)).float() / 127.5 - 1.0
        m = torch.nn.functional.interpolate(m.unsqueeze(0), size=(self.img_size, self.img_size), mode='nearest').squeeze(0)
        m = m * 2.0 - 1.0
        return x, m
# ----------------------- Dataset -------------------------------


# Dataset for .h5 files, extracting T1ce modality (index 1)

# Dataset for .h5 files, extracting FLAIR (0) + T1Gd (2) as a 2‑channel tensor
class H5FlairT1GdDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".h5")]
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import h5py, numpy as np, torch, torch.nn.functional as F
        with h5py.File(self.paths[idx], "r") as f:
            vol = f["image"][:]  # (H,W,4) float
            msk = f["mask"][:]   # (H,W,3) uint8 {0,1}
            img = vol[:, :, [0, 2]].astype("float32")  # (H,W,2) FLAIR+T1Gd
            msk = msk.astype("float32")                # (H,W,3)
        # Per‑channel min‑max to [0,1] for image
        eps = 1e-8
        cmins = img.reshape(-1, 2).min(axis=0)
        cmaxs = img.reshape(-1, 2).max(axis=0)
        img = (img - cmins) / (cmaxs - cmins + eps)
        # To tensor
        x = torch.from_numpy(img).permute(2, 0, 1)  # [2,H,W]
        m = torch.from_numpy(msk).permute(2, 0, 1)  # [3,H,W], in [0,1]
        # Resize both
        x = F.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
        m = F.interpolate(m.unsqueeze(0), size=(self.img_size, self.img_size), mode="nearest").squeeze(0)
        # Map image to [-1,1]; map mask to [-1,1] for D concatenation stability
        x = x * 2 - 1
        m = m * 2 - 1
        return x, m
# ----------------------- Model Blocks -------------------------

# MaskEncoder and conditional Generator wrapper
class MaskEncoder(nn.Module):
    def __init__(self, in_ch=3, nz=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, nz)
    def forward(self, m):
        h = self.enc(m).view(m.size(0), -1)
        zc = self.fc(h)  # [B, nz]
        return zc
# ----------------------- Model Blocks -------------------------

def build_generator(steps, args):
    # backbone
    layers: List[nn.Module] = []
    c_in, c_out = args.nz, args.gf * (2 ** (steps - 1))
    layers += [nn.ConvTranspose2d(c_in, c_out, 4, 1, 0, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]
    for _ in range(steps):
        layers += [nn.ConvTranspose2d(c_out, c_out // 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out // 2), nn.ReLU(True)]
        c_out //= 2
    layers += [nn.ConvTranspose2d(c_out, args.nc, 3, 1, 1, bias=False), nn.Tanh()]
    backbone = nn.Sequential(*layers)
    class G(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.menc = MaskEncoder(in_ch=args.mask_nc, nz=args.nz)
            self.proj = nn.Linear(args.nz, args.nz, bias=False)
        def forward(self, z, m):
            # condition z with mask embedding
            zc = self.proj(self.menc(m))  # [B, nz]
            z = z + zc.unsqueeze(-1).unsqueeze(-1)
            return self.backbone(z)
    return G()

def build_discriminator(steps, args):
    layers: List[nn.Module] = []
    c_in, c_out = args.nc + args.mask_nc, args.df
    layers += [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False), nn.LeakyReLU(0.2)]
    for _ in range(steps - 1):
        layers += [nn.Conv2d(c_out, c_out * 2, 4, 2, 1, bias=False), nn.InstanceNorm2d(c_out * 2, affine=True), nn.LeakyReLU(0.2)]
        c_out *= 2
    features = nn.Sequential(*layers)
    pool = nn.AdaptiveAvgPool2d(1)
    head = nn.Conv2d(c_out, 1, 1, 1, 0, bias=False)
    class D(nn.Module):
        def __init__(self):
            super().__init__()
            self.f, self.pool, self.head = features, pool, head
            self.feat_dim = c_out
            # Learnable prototype bank in feature space
            self.prototypes = nn.Parameter(torch.randn(args.num_prototypes, self.feat_dim))
        def features_vec(self, x):
            f = self.f(x).mean([2,3])  # [B, feat_dim]
            return F.normalize(f, dim=1)
        def forward(self, x):
            return self.head(self.pool(self.f(x))).view(-1)
    return D()

# ----------------------- Utils -------------------------------

def gradient_penalty(D, real, fake, device, lamb):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_inter = D(inter)
    grad = autograd.grad(d_inter.sum(), inter, create_graph=True)[0]
    return ((grad.view(real.size(0), -1).norm(2, dim=1) - 1) ** 2).mean() * lamb

# ----------------------- Inline KMEx (EM-style anchor refit) ------------------

def refit_anchors_inline(
    D: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 8,
    ema_alpha: float = 0.9,
    kmeans_seed: int = 0,
) -> dict:
    """Refit the anchor matrix D.prototypes via K-Means on D-features.

    This implements the inline-KMEx step of the proposed self-explainable framework.
    Every refit_every epochs (controlled by the caller), we:
      1. Pass the full training set through D's feature extractor.
      2. Run K-Means with K = D.prototypes.shape[0] on those features.
      3. EMA-blend the new cluster centers into D.prototypes:
            A_new = alpha * A_old + (1 - alpha) * kmeans.cluster_centers_
    The L2 normalization at the use site (G's prototype-pull computation) means
    we can keep raw vectors in storage; only direction matters for cosine
    similarity.

    Args:
        D: Discriminator with .f (conv features) and .prototypes (K, d) parameter.
        dataset: The training dataset (yields (image, mask) pairs).
        device: Torch device.
        batch_size: KMeans pass batch size. Smaller = lower memory; 8 is safe.
        ema_alpha: EMA weight for the OLD anchors. 0.9 is a smooth update;
                   1.0 would freeze refits; 0.0 would replace completely.
        kmeans_seed: KMeans random_state for reproducibility.

    Returns:
        Dict with diagnostic statistics for logging (cluster compactness,
        anchor drift magnitude).
    """
    from sklearn.cluster import KMeans
    was_training = D.training
    D.eval()
    embeds = []
    with torch.no_grad():
        for x, m in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x, m = x.to(device), m.to(device)
            feat = D.f(torch.cat([x, m], dim=1)).mean([2, 3])  # [B, d]
            embeds.append(feat.cpu())
    embeds = torch.cat(embeds).numpy()

    K = D.prototypes.shape[0]
    km = KMeans(n_clusters=K, random_state=kmeans_seed, n_init=10).fit(embeds)
    new_centers = torch.from_numpy(km.cluster_centers_).float().to(D.prototypes.device)

    # Diagnostics
    old = D.prototypes.data.clone()
    drift = float(torch.norm(new_centers - old, p=2, dim=1).mean().item())
    inertia = float(km.inertia_)

    # EMA blend (in-place; D.prototypes remains an nn.Parameter)
    with torch.no_grad():
        D.prototypes.data.copy_(ema_alpha * old + (1.0 - ema_alpha) * new_centers)

    if was_training:
        D.train()
    return {"anchor_drift_l2": drift, "kmeans_inertia": inertia}


# ----------------------- Main -------------------------------

def main():
    args = get_args()
    # If training on RGB overlays, force nc=3 for generator/discriminator I/O
    if args.use_overlays and args.nc != 3:
        print("[Info] Using overlays: setting --nc to 3 (RGB).")
        args.nc = 3
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    mmis_scores = []

    # Ensure generator output matches dataset size (power-of-two)
    orig_size = args.img_size
    pow2_size = int(2 ** round(math.log2(max(4, orig_size))))
    if pow2_size != orig_size:
        print(f"[Note] Adjusting img_size from {orig_size} to nearest power-of-two: {pow2_size}")
    target_img_size = pow2_size
    SAMPLE_N = min(64, args.batch_size)

    def mask_black_background(img_tensor):
        threshold = 0.05
        mask = img_tensor > threshold
        img_tensor = img_tensor * mask
        return img_tensor

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*args.nc, [0.5]*args.nc)
    ])

    # If using overlays and no overlay_cache is set, default to overlay_train inside data_dir
    if args.use_overlays and args.overlay_cache is None:
        args.overlay_cache = str(Path(args.data_dir) / "overlay_train")

    if args.use_overlays:
        dataset = H5OverlayDataset(
            root=args.data_dir,
            img_size=target_img_size,
            overlay_alpha=args.overlay_alpha,
            overlay_cache=args.overlay_cache,
        )
    else:
        dataset = H5FlairT1GdDataset(args.data_dir, img_size=target_img_size)

    # Early exit for precompute_only overlays
    if args.precompute_only and args.use_overlays:
        print(f"[Info] Precomputing overlays into {args.overlay_cache}")
        for _ in DataLoader(dataset, batch_size=1, shuffle=False):
            pass
        print("[Info] Done precomputing overlays.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    steps = int(math.log2(target_img_size)) - int(math.log2(4))
    G, D = build_generator(steps, args).to(device), build_discriminator(steps, args).to(device)
    optim_G = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    optim_D = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    # ----- Resume from checkpoint if requested -----
    # Loading G and D from a prior run's epoch-N checkpoints allows continuing
    # training without losing the work from the first N epochs. Note: the Adam
    # optimizer state (moment estimates) is NOT restored, so there is a brief
    # ~50-100 step warm-up while Adam's running statistics rebuild. This is
    # negligible over thousands of epochs but worth noting.
    if args.resume_g and args.resume_d:
        print(f"[Resume] Loading G from {args.resume_g}")
        G.load_state_dict(torch.load(args.resume_g, map_location=device, weights_only=True))
        print(f"[Resume] Loading D from {args.resume_d}")
        D.load_state_dict(torch.load(args.resume_d, map_location=device, weights_only=True))
        print(f"[Resume] Training will run epochs {args.start_epoch + 1} .. {args.epochs}")

    fixed_z = torch.randn(SAMPLE_N, args.nz, 1, 1, device=device)
    print(f"Epochs: {args.epochs}\nStarting training…")

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        d_loss_total = g_loss_total = 0; batches = 0
        for i, (real, cond_m) in enumerate(loader):
            real, cond_m = real.to(device), cond_m.to(device)
            bsz = real.size(0)

            # Train D (args.critic_steps per G update)
            for _ in range(args.critic_steps):
                optim_D.zero_grad()
                z = torch.randn(bsz, args.nz, 1, 1, device=device)
                fake = G(z, cond_m).detach()
                d_real = D(torch.cat([real, cond_m], dim=1)).mean()
                d_fake = D(torch.cat([fake, cond_m], dim=1)).mean()
                gp = gradient_penalty(D, torch.cat([real, cond_m], dim=1), torch.cat([fake, cond_m], dim=1), device, args.gp_lambda)
                d_loss = d_fake - d_real + gp
                d_loss.backward(); optim_D.step()

            # Train G
            optim_G.zero_grad()
            z = torch.randn(bsz, args.nz, 1, 1, device=device)
            fake = G(z, cond_m)
            g_adv = -D(torch.cat([fake, cond_m], dim=1)).mean()
            # Prototype pull with cosine distance on L2-normalized embeddings
            feat_f = D.features_vec(torch.cat([fake, cond_m], dim=1))  # [B, d], L2-normalized
            P = F.normalize(D.prototypes, dim=1).detach()  # [K, d], stop grad w.r.t G
            # cosine distance = 1 - cosine_similarity
            cos_sim = torch.matmul(feat_f, P.t())  # [B, K]
            cos_dist = 1.0 - cos_sim
            min_dists, _ = torch.min(cos_dist, dim=1)
            proto_pull = min_dists.mean()
            g_loss = g_adv + args.lambda_proto * proto_pull
            g_loss.backward(); optim_G.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            batches += 1

        avg_d_loss = d_loss_total / batches
        avg_g_loss = g_loss_total / batches
        print(f"Epoch [{epoch}/{args.epochs}]  D_loss: {avg_d_loss:.4f}  G_loss: {avg_g_loss:.4f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            # Use a batch of conditioning masks for visualization
            try:
                sample_masks = next(iter(loader))[1].to(device)
            except StopIteration:
                sample_masks = torch.zeros(0, args.mask_nc, target_img_size, target_img_size, device=device)
            if sample_masks.size(0) >= SAMPLE_N:
                sample_masks = sample_masks[:SAMPLE_N]
            else:
                base = sample_masks if sample_masks.size(0) > 0 else torch.zeros(1, args.mask_nc, target_img_size, target_img_size, device=device)
                reps = (SAMPLE_N + base.size(0) - 1) // base.size(0)
                sample_masks = base.repeat(reps, 1, 1, 1)[:SAMPLE_N]
            imgs = G(fixed_z, sample_masks).cpu()
            imgs = (imgs + 1) / 2  # Scale to [0, 1]
            imgs = imgs.clamp(0, 1)
            vutils.save_image(imgs, f"{args.out_dir}/epoch_{epoch:04d}.png", nrow=8, normalize=False)
        if epoch % 10 == 0 or epoch == args.epochs:
            mmis_score = compute_mmis_score(G, args.nz, device, loader, args.mask_nc, target_img_size, num_samples=SAMPLE_N)
            print(f"Epoch {epoch} | MMIS Score: {mmis_score:.4f}")
            mmis_scores.append(mmis_score)

            # Plot MMIS scores
            plt.figure()
            plt.plot(range(1, len(mmis_scores)+1), mmis_scores, label="MMIS Score")
            plt.xlabel("Epoch")
            plt.ylabel("MMIS (MS-SSIM)")
            plt.title("MMIS Score over Epochs")
            plt.grid(True)
            plt.legend()
            plt.savefig("mmis_score_plot.png")
            plt.close()

        # ----- Inline KMEx anchor refit (EM-style alternation) -----
        # Every `kmex_refit_every` epochs (and not at epoch 0), re-fit the anchor
        # matrix D.prototypes via K-Means on D's features, EMA-blended with the
        # current anchors. This makes the prototype set genuinely inline rather
        # than purely post-hoc, supporting the self-explainable framing.
        if (args.kmex_refit_every > 0
                and epoch % args.kmex_refit_every == 0
                and epoch > 0):
            diag = refit_anchors_inline(
                D=D,
                dataset=dataset,
                device=device,
                batch_size=8,
                ema_alpha=args.kmex_ema_alpha,
                kmeans_seed=args.seed,
            )
            print(f"[Inline-KMEx] Epoch {epoch}: anchor drift L2 = "
                  f"{diag['anchor_drift_l2']:.4f}, KMeans inertia = "
                  f"{diag['kmeans_inertia']:.2f}")

        # ----- Periodic checkpoint save -----
        # Save G and D every `save_every` epochs (and always at the final epoch).
        # These checkpoints support compute_metrics.py, post-hoc analysis, and
        # the K-sweep we'll run on the saved D-features. Without these the
        # only output is sample PNGs, which cannot reproduce the metrics.
        if ((args.save_every > 0 and epoch % args.save_every == 0)
                or epoch == args.epochs):
            ckpt_dir = args.out_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            tag = f"epoch_{epoch:04d}" if epoch != args.epochs else "final"
            torch.save(G.state_dict(), os.path.join(ckpt_dir, f"G_{tag}.pt"))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, f"D_{tag}.pt"))
            print(f"[Checkpoint] Saved G_{tag}.pt and D_{tag}.pt")

    # ----- KMEx -----
    print("Mining KMEx prototypes…")
    embeds, imgs, msks = [], [], []
    with torch.no_grad():
        for x, m in DataLoader(dataset, batch_size=8):
            # Compute embeddings using both image and mask concatenated (for D input)
            embeds.append(D.f(torch.cat([x.to(device), m.to(device)], dim=1)).mean([2,3]).cpu())
            imgs.append(x)
            msks.append(m)
    embeds = torch.cat(embeds).numpy()
    imgs = torch.cat(imgs)
    msks = torch.cat(msks)
    kmeans = KMeans(args.num_prototypes, random_state=args.seed).fit(embeds)
    proto_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeds)

    # ----- Save raw embeddings + cluster artifacts for post-hoc K-sweep -----
    # The K-sweep (over K in {4, 8, 16, 32}) can now be run on saved D-features
    # without retraining. This is what supports the hyperparameter sensitivity
    # study (E5) in the manuscript revision plan.
    sweep_dir = os.path.join(args.out_dir, "ksweep_data")
    os.makedirs(sweep_dir, exist_ok=True)
    np.save(os.path.join(sweep_dir, "embeddings.npy"), embeds)
    np.save(os.path.join(sweep_dir, "cluster_centers_k16.npy"), kmeans.cluster_centers_)
    np.save(os.path.join(sweep_dir, "cluster_labels_k16.npy"), kmeans.labels_)
    # Cluster-size histogram: how many of the 1107 samples fall into each cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}
    with open(os.path.join(sweep_dir, "cluster_sizes_k16.json"), "w") as f:
        json.dump(cluster_sizes, f, indent=2)
    print(f"[K-sweep data] Saved embeddings (shape {embeds.shape}), "
          f"cluster_centers, cluster_labels, cluster_sizes to {sweep_dir}/")
    print(f"[K-sweep data] Cluster size distribution at K=16: {cluster_sizes}")

    protos = imgs[proto_idx]
    proto_msks = msks[proto_idx]
    vutils.save_image(protos, f"{args.out_dir}/prototypes.png", nrow=4, normalize=True, value_range=(-1, 1))
    
    split_dir = os.path.join(args.out_dir, 'individual_prototypes')
    os.makedirs(split_dir, exist_ok=True)

    # protos: Tensor[num_prototypes, C, H, W]
    for i, img_tensor in enumerate(protos):
        # Save each prototype image separately
        vutils.save_image(
            img_tensor,
            os.path.join(split_dir, f'prototype_{i:02d}.png'),
            normalize=True, value_range=(-1, 1)
        )
    
    def replace_leaky_relu_with_relu(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.LeakyReLU):
                setattr(module, name, torch.nn.ReLU())
            else:
                replace_leaky_relu_with_relu(child)
    
    print("Associating last epoch generated images with prototypes...")
    G.eval()
    with torch.no_grad():
        # Regenerate the final epoch images using fixed_z
        # Use a batch of conditioning masks for visualization
        try:
            sample_masks = next(iter(loader))[1].to(device)
        except StopIteration:
            sample_masks = torch.zeros(0, args.mask_nc, target_img_size, target_img_size, device=device)
        if sample_masks.size(0) >= SAMPLE_N:
            sample_masks = sample_masks[:SAMPLE_N]
        else:
            base = sample_masks if sample_masks.size(0) > 0 else torch.zeros(1, args.mask_nc, target_img_size, target_img_size, device=device)
            reps = (SAMPLE_N + base.size(0) - 1) // base.size(0)
            sample_masks = base.repeat(reps, 1, 1, 1)[:SAMPLE_N]
        final_fakes = G(fixed_z, sample_masks).cpu()  # Tensor[SAMPLE_N, C, H, W]
        # Compute feature embeddings using the discriminator
        fake_feats = D.f(torch.cat([final_fakes.to(device), sample_masks[:final_fakes.size(0)].to(device)], dim=1)).mean([2, 3]).cpu().numpy()
    # Assign each fake image to the nearest prototype cluster center
    assoc_idx, _ = pairwise_distances_argmin_min(fake_feats, kmeans.cluster_centers_)

    assoc_dir = os.path.join(args.out_dir, 'associations')
    os.makedirs(assoc_dir, exist_ok=True)
    for i, (img_tensor, proto_i) in enumerate(zip(final_fakes, assoc_idx)):
        # Save each fake with its associated prototype index
        vutils.save_image(
            img_tensor,
            os.path.join(assoc_dir, f'fake_{i:02d}_proto_{proto_i:02d}.png'),
            normalize=True, value_range=(-1, 1)
        )

    from torchvision.utils import make_grid

    paired_dir = os.path.join(args.out_dir, 'prototype_pairs')
    os.makedirs(paired_dir, exist_ok=True)

    for i, (img_tensor, proto_i) in enumerate(zip(final_fakes, assoc_idx)):
        proto_img = protos[proto_i]

        # Stack prototype and generated image side-by-side
        pair_grid = make_grid([proto_img, img_tensor], nrow=2, normalize=True, value_range=(-1, 1))

        vutils.save_image(
            pair_grid,
            os.path.join(paired_dir, f'pair_{i:02d}_with_proto_{proto_i:02d}.png')
        )

    # ----- PRP Heatmaps -----
    print("Generating PRP heatmaps…")
    import torch.nn as nn
    from captum.attr._core.layer.layer_lrp import LayerLRP

    def replace_leaky_relu_with_relu(module):
        for name, child in module.named_children():
            if isinstance(child, nn.LeakyReLU):
                setattr(module, name, nn.ReLU())
            else:
                replace_leaky_relu_with_relu(child)

    def replace_instancenorm_with_batchnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.InstanceNorm2d):
                setattr(module, name, nn.BatchNorm2d(
                    child.num_features, eps=child.eps, momentum=child.momentum,
                    affine=child.affine, track_running_stats=child.track_running_stats
                ))
            else:
                replace_instancenorm_with_batchnorm(child)

    D.eval()
    replace_leaky_relu_with_relu(D)
    replace_instancenorm_with_batchnorm(D)

    target_layer = None
    for m in D.modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in Discriminator.")

    lrp = LayerLRP(D, target_layer)
    G.to(device)
    D.to(device)

    # Generate PRP heatmaps using both image and mask (for 6-channel D input)
    # ALSO accumulates data for the publication-quality three-panel figures.
    panel_data = []  # List of (exemplar, mask, heatmap) for each prototype
    for i, (img_tensor, msk_tensor) in enumerate(zip(protos, proto_msks)):
        inp = torch.cat([img_tensor.unsqueeze(0).to(device),
                         msk_tensor.unsqueeze(0).to(device)], dim=1).requires_grad_()
        attr = lrp.attribute(inp)
        # If attr is not 4D, reshape it safely
        if attr.dim() == 3:
            attr = attr.unsqueeze(0)
        # Reduce across channels to get grayscale heatmap
        attr = attr.abs().sum(dim=1, keepdim=True)  # [1, 1, H, W]
        # Normalize to 0-1
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        # Resize to match prototype image size (optional)
        attr = TF.resize(attr, [inp.shape[2], inp.shape[3]])
        # Save the heatmap (single-channel)
        vutils.save_image(attr, f"{args.out_dir}/prototype_{i:02d}_prp.png", normalize=True)

        # Accumulate data for the combined panel figures
        panel_data.append((img_tensor.cpu(), msk_tensor.cpu(), attr.squeeze(0).cpu()))

    # ----- Publication-quality three-panel figures per prototype -----
    # Each prototype gets one figure with three panels:
    #   (a) exemplar RGB overlay image
    #   (b) structured mask (3-channel one-hot) visualized as RGB
    #   (c) PRP heatmap overlaid on the grayscale component of the exemplar
    # Plus a master 4x4 grid showing all 16 prototypes' three-panel layouts.
    print("Generating publication-quality three-panel PRP figures...")
    panel_dir = os.path.join(args.out_dir, "prp_panels")
    os.makedirs(panel_dir, exist_ok=True)

    def _to_01_rgb(t: torch.Tensor) -> np.ndarray:
        """Convert a [-1, 1] tensor of shape [C, H, W] to a [0, 1] HxWx3 array."""
        x = (t.clamp(-1, 1) + 1) / 2
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x.permute(1, 2, 0).numpy()

    for i, (img_t, msk_t, hm_t) in enumerate(panel_data):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        # (a) Exemplar
        axes[0].imshow(_to_01_rgb(img_t))
        axes[0].set_title(f"Prototype {i:02d} exemplar")
        axes[0].axis("off")
        # (b) Mask
        axes[1].imshow(_to_01_rgb(msk_t))
        axes[1].set_title("Mask (3-channel)")
        axes[1].axis("off")
        # (c) Heatmap on grayscale base
        gray = _to_01_rgb(img_t).mean(axis=-1)
        axes[2].imshow(gray, cmap="gray")
        axes[2].imshow(hm_t.squeeze().numpy(), cmap="jet", alpha=0.5)
        axes[2].set_title("PRP heatmap")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(panel_dir, f"panel_prototype_{i:02d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Master 4x4 montage: 16 prototypes, each shown as exemplar + heatmap overlay
    # (one row per prototype would be too tall; show as 2 rows of 8 panels showing
    # the heatmap overlay only — most compact and most informative for the paper).
    n_protos = len(panel_data)
    rows = 4
    cols = (n_protos + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    for i, (img_t, msk_t, hm_t) in enumerate(panel_data):
        r, c = i // cols, i % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        gray = _to_01_rgb(img_t).mean(axis=-1)
        ax.imshow(gray, cmap="gray")
        ax.imshow(hm_t.squeeze().numpy(), cmap="jet", alpha=0.5)
        ax.set_title(f"P{i:02d}", fontsize=10)
        ax.axis("off")
    # Hide any unused axes
    for j in range(n_protos, rows * cols):
        r, c = j // cols, j % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "prp_montage_all.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {n_protos} three-panel figures to {panel_dir}/")
    print(f"  Saved master montage to {args.out_dir}/prp_montage_all.png")


    # ----- Diversity Metrics -----
    print("Calculating diversity metrics…")
    intra = [pdist(embeds[kmeans.labels_ == i]).mean() if sum(kmeans.labels_ == i) > 1 else 0 for i in range(args.num_prototypes)]
    inter = pdist(kmeans.cluster_centers_).mean()
    metrics = {"intra_cluster": float(np.mean(intra)), "inter_prototype": float(inter)}
    (Path(args.out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)

if __name__ == '__main__':
    main()