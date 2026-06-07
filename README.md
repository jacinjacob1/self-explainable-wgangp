# Self-Explainable WGAN-GP for Brain Tumor MRI Synthesis Using KMEx Prototype Learning

Official code for the IEEE Access paper **"Self-Explainable WGAN-GP for Brain
Tumor MRI Synthesis Using KMEx Prototype Learning"** (Manuscript ID
**Access-2026-09588**).

This repository contains the proposed model, the baseline GANs, the full
evaluation suite, and the analysis/figure scripts needed to reproduce the
results reported in the paper.

---

## Overview

The framework augments a conditional WGAN-GP with two mechanisms:

1. **Anchor-based regularization** — aligns the discriminator's feature space
   toward mined cluster centers, providing the fidelity and training-stability
   gains.
2. **Inline-KMEx prototype mining** — an EM-style K-Means alternation on the
   discriminator's feature space that associates each synthesized image with a
   prototype for case-based interpretation.

A **Prototype Relevance Projection (PRP)** procedure then derives pixel-level
relevance heatmaps via layer-wise relevance propagation.

A component-wise ablation shows the anchor regularization (not the prototype
mining) is responsible for the fidelity gains; the prototype mining provides
the explanation scaffold.

---

## Repository contents

| File | Purpose |
|------|---------|
| `v17_inline_kmex.py` | Proposed model: conditional WGAN-GP + anchor regularization + inline-KMEx |
| `baseline_conditional.py` | Baseline GANs (DCGAN-cond, WGAN-GP-cond, SAGAN-lite-cond) |
| `compute_metrics.py` | Evaluation: FID, KID, LPIPS, Precision/Recall, SSIM, MMIS |
| `crossdataset_fid.py` | Cross-dataset FID (BraTS ↔ Figshare) |
| `kmeans_sweep.py` | K-sensitivity (silhouette / separation over K) |
| `prp_strict.py` | PRP heatmap generation |
| `score_recovery.py` | Signal-driven slice-selection rule |
| `extract_figshare.py` | Convert the Figshare `.mat` dataset to PNG + masks |
| `precompute_npy_masks.py` | Build the overlay-cache `.npy` mask files |
| `plot_anchor_drift.py` | Anchor-drift convergence figure |
| `plot_loss_curves.py` | WGAN-GP training-dynamics figure |
| `requirements.txt` | Pinned dependencies |
| `proposed_inline_seed0_v2.log` | Training log of the reported run (for verifying convergence) |

---

## Installation

```bash
git clone https://github.com/jacinjacob1/self-explainable-wgangp.git
cd <REPO>
python -m venv venv && source venv/bin/activate    # optional
pip install -r requirements.txt
```

Python 3.10+ and a CUDA-capable GPU are recommended. The reported run used a
single NVIDIA H100. Apple Silicon (MPS) and CPU are also supported via the
`--device` flag on the evaluation scripts.

---

## Data

The datasets are **not** redistributed here (their licenses require download
from the original sources).

### BraTS2020 (training)
Download from the [BraTS 2020 challenge](https://www.med.upenn.edu/cbica/brats2020/data.html).
Each volume is expected as an HDF5 file with `image` (H×W×4: T1, T1Gd, T2,
FLAIR) and `mask` (H×W×3 one-hot) arrays. Place the `.h5` files in `h5_train/`.

The training subset is selected by a signal-driven rule (top-3 axial slices per
volume by FLAIR-intensity score), yielding **1,107 slices from 369 volumes**.
See `score_recovery.py`.

### Figshare brain tumor dataset (cross-dataset evaluation)
Download from [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
(Cheng et al., 2015). Convert to PNG with:

```bash
python extract_figshare.py \
    --src path/to/figshare_mat_files \
    --out_images figshare_extracted/images \
    --out_masks  figshare_extracted/masks \
    --img_size 256 --save_metadata
```

---

## Reproduction

### 0. Build the overlay-mask cache (one time)

```bash
python precompute_npy_masks.py --data_dir h5_train --overlay_cache overlay_train
```

### 1. Train the proposed model

```bash
python v17_inline_kmex.py \
    --data_dir h5_train \
    --overlay_cache overlay_train \
    --epochs 3000 \
    --batch_size 16 \
    --critic_steps 2 \
    --g_lr 1e-4 \
    --d_lr 2e-5 \
    --gp_lambda 10 \
    --nz 128 \
    --gf 64 \
    --num_prototypes 16 \
    --kmex_refit_every 100 \
    --kmex_ema_alpha 0.9 \
    --lambda_proto 0.05 \
    --seed 0 \
    --use_overlays \
    --out_dir proposed_inline_seed0_v2
```

### 2. Train the matched baseline

```bash
python baseline_conditional.py \
    --data_dir h5_train \
    --overlay_cache overlay_train \
    --models wgangp \
    --epochs 3000 \
    --batch_size 16 \
    --n_critic 2 \
    --g_lr 1e-4 \
    --d_lr 2e-5 \
    --gp_lambda 10 \
    --nz 128 \
    --seed 0 \
    --out_root baselines
```

### 3. Compute metrics

```bash
python compute_metrics.py \
    --model proposed \
    --ckpt_g proposed_inline_seed0_v2/G_final.pt \
    --data_dir h5_train \
    --overlay_cache overlay_train \
    --n_samples 3000 \
    --seed 0 \
    --out_json results/proposed_metrics.json
```

### 4. K-sensitivity, PRP, cross-dataset

```bash
# K-sweep (needs the discriminator feature embeddings exported by prp_strict)
python kmeans_sweep.py --embeddings results/d_embeddings.npy \
    --k_values 4 8 16 32 --n_seeds 5 --out_json results/k_sweep.json

# PRP heatmaps
python prp_strict.py --ckpt_g proposed_inline_seed0_v2/G_final.pt \
    --ckpt_d proposed_inline_seed0_v2/D_final.pt \
    --data_dir h5_train --overlay_cache overlay_train \
    --num_prototypes 16 --out_dir results/prp --seed 0

# Cross-dataset FID
python crossdataset_fid.py \
    --ckpt_g proposed_inline_seed0_v2/G_final.pt \
    --data_dir h5_train --overlay_cache overlay_train \
    --figshare_dir figshare_extracted/images \
    --n_samples 3000 --seed 0 --out_json results/crossdataset_fid.json
```

### 5. Figures

```bash
python plot_anchor_drift.py --log proposed_inline_seed0_v2.log \
    --out figures/anchor_drift_curve
python plot_loss_curves.py --proposed_log proposed_inline_seed0_v2.log \
    --out figures/loss_curves
```

---

## Key results (seed 0)

| Metric | Baseline (WGAN-GP-cond) | Proposed |
|--------|------------------------|----------|
| FID ↓ | 205.97 | **154.47** (−25%) |
| KID ×10² ↓ | 20.60 | **15.46** |
| LPIPS (diversity) ↑ | 0.177 | **0.205** |
| Recall ↑ | 0.0000 | 0.0045 |
| SSIM ↑ | 0.726 | 0.720 |

**Ablation:** A1 (WGAN-GP) 205.97 → A2 (KMEx mining only) 197.62 → A4 (full)
154.47 — the anchor regularization, not the prototype mining, drives the gain.

**Anchor convergence:** drift decreases monotonically 48.95 → 4.89 (≈10×) over
the 3000-epoch run (verifiable from `proposed_inline_seed0_v2.log`).

**Cross-dataset:** ceiling (BraTS↔Figshare) 188.29; in-domain 159.39;
cross-dataset 254.08; ratio 1.35.

---

## Reproducibility notes

- All reported numbers use **seed 0**. Absolute FID values are high in part
  because the Inception-v3 feature extractor is ImageNet-trained (not
  domain-tuned) and the reference set is small (1,107 samples); the paper
  emphasizes the **relative** improvement over the matched baseline under
  identical evaluation conditions.
- Precision/Recall are estimated on a small reference set and should be read
  with that caveat.
- The training log is included so the anchor-drift and loss-curve figures can
  be regenerated and verified directly.

---

## Citation

```bibtex
@article{accesskmex2026,
  title   = {Self-Explainable WGAN-GP for Brain Tumor MRI Synthesis Using KMEx Prototype Learning},
  author  = {<authors>},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript ID Access-2026-09588}
}
```

## License

Released under the MIT License (see `LICENSE`).

## Acknowledgements

BraTS2020 (Menze et al.; Bakas et al.) and the Figshare brain tumor dataset
(Cheng et al., 2015). We thank the participating radiologists for the reader
study.
