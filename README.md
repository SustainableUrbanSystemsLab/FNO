#  Fourier Neural Operators for Urban Wind Wakes

This repository contains a highly specialized, massively parallelized machine learning ecosystem dedicated to predicting High-Fidelity urban wind aerodynamics. 

It is designed to solve **The Fourier Wake Blurring Problem (Gibbs Phenomenon)**—a critical mathematical flaw where standard FNOs naturally blur and fail to capture the sharp boundaries of aerodynamic wakes behind large physical structures. To solve this, this repository contains **four distinct architectures**, a unified memory-mapped cluster pipeline, and a comprehensive mathematical diagnostic suite.

---

## 🏗️ 1. Architectures Implemented

All model architectures reside in `core/models/` and have their own dedicated multi-GPU training execution scripts in `pipelines/train/`.

| Model Name | Script | Description |
| :--- | :--- | :--- |
| **Standard FNO** | `distributed.py` | A massive high-capacity ($M=48, W=128$) standard Fourier operator pushed to extreme limits to force retention of high-frequency step functions. |
| **Hybrid FNO** | `hybrid.py` | An architectural merger that wraps the global FNO within extremely localized Convolutional (CNN) skip-connections to manually force edge reconstruction (similar to a U-Net). |
| **Strict PINN-FNO** | `pinn.py` | An operator bound by rigid multi-objective calculus. During training, it explicitly calculates continuity and momentum equations to mathematically ban hallucinated winds. |
| **Geo-FNO** | `geo.py` | A geometric specialist that uses an SDF-conditional Latent Coordinate Encoder to physically warp the computational grid around buildings, giving the FFTs a "clean" topological space unhindered by solid boundaries. |

---

## ⚡ 2. Deployment on PACE ICE Cluster

The entire repository is dynamically linked via `config.toml` (and override configs like `configs/config_test.toml`). It natively utilizes SLURM for multi-GPU training.

We use **Rolling Atomic Checkpoints**. The framework will safely and atomically overwrite temporary `.pth.tmp` files every epoch before solidifying the name—this strictly protects cluster users from hitting gigabyte-scale Disk Quota `zipfile` corruption.

**Trigger a Cluster Training Job:**
```bash
bash slurm/deploy_ice.sh --config configs/config_test.toml --script pipelines/train/geo.py --gpu h200 --ngpus 1 --fresh
```

---

## 📊 3. The Master Diagnostic Suite

Instead of blindly measuring MSE, this repository has a highly specialized **Wake Diagnostic Engine** tailored strictly for aerodynamic physics.

**To run the Comprehensive Comparison Script:**
*(Warning: Run this exclusively in an interactive `salloc` compute session to prevent Login-Node OOM memory assassination, as it seamlessly loads ~4GB of models directly into RAM).*
```bash
uv run python tools/compare_all_models.py --data test_csv/ML_FormFlux_1_135.csv --output model_comparison.png
```

### Metrics Mathematically Handled:
- **R-squared ($R^2$)**: The native coefficient of determination.
- **Structural Similarity (SSIM)**: Explicitly tests the structural image boundaries matching the Conditional Transformer.
- **Gradient Correlation (`grad_corr`)**: Extracts Sobel $(x, y)$ coordinate derivatives and correlates the gradient jumps exactly at building boundaries.
- **Mean Absolute Percentage Error (MAPE)**: Bound aggressively to regions $v > 0.1 \text{ m/s}$ to prevent mathematically infinite errors near solid objects.
- **Wake MAE**: Evaluates the model exclusively on the aerodynamic dark zones.
- **Peak MAE**: Evaluates the model exclusively on high-velocity fluid jets.

---

## 📝 Document Logs

Please rely on the official logs for research progression.
- `model_architectures_explained.md` - Full mathematical breakdown of Jacobian limits, spectral wave physics, and gradient conflicts.
- `experiments_log.md` - Complete training history tracking the effect of Loss configurations (Wake weight, Peak weight, Spectral weight) on prediction behavior.
- `test_commands_ice.md` - The repository cheat-sheet for exact `python` commands.
