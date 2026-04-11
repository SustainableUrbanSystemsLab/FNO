#!/usr/bin/env python3
"""
Export trained PyTorch models to ONNX format.
=============================================
Converts existing .pt/.pth checkpoints to ONNX - no retraining needed.

Supports:
  - Pix2PixHD Generator (single forward pass -> clean ONNX export)
  - WindTransformer Denoiser (exports the denoiser; DDIM loop runs in server code)
  - HybridFNO (spectral conv replaced with real matmul for DML-compatible export)

Usage:
  # Pix2PixHD
  python export_onnx.py --arch pix2pixhd \
      --checkpoint results_pix2pixhd/checkpoints/ckpt_best.pt \
      --stats results_pix2pixhd/checkpoints/stats.pt \
      --output pix2pixhd_generator.onnx

  # WindTransformer (denoiser only)
  python export_onnx.py --arch windtransformer \
      --checkpoint results/checkpoints/wind_model_best.pth \
      --stats results/checkpoints/stats.pt \
      --output wind_denoiser.onnx

  # Verify exported ONNX against PyTorch
  python export_onnx.py --arch pix2pixhd \
      --checkpoint results_pix2pixhd/checkpoints/ckpt_best.pt \
      --stats results_pix2pixhd/checkpoints/stats.pt \
      --output pix2pixhd_generator.onnx \
      --verify
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

# ================================================================
# Constants (must match training scripts)
# ================================================================
IMG_H, IMG_W = 504, 504
X_CH = 8
Y_CH = 1

# Pix2PixHD defaults
NGF = 64
N_RESBLOCKS_G1 = 9
N_DOWNSAMPLE_G1 = 3
N_RESBLOCKS_G2 = 3


def build_pix2pixhd_generator(config=None, use_local_enhancer=True):
    """Build Pix2PixHD generator from pix2pix_hd.py classes."""
    from pix2pix_hd import Pix2PixHDGenerator

    ngf = config.get("ngf", NGF) if config else NGF
    n_resblocks_g1 = config.get("n_resblocks_g1", N_RESBLOCKS_G1) if config else N_RESBLOCKS_G1
    n_downsample_g1 = config.get("n_downsample_g1", N_DOWNSAMPLE_G1) if config else N_DOWNSAMPLE_G1
    n_resblocks_g2 = config.get("n_resblocks_g2", N_RESBLOCKS_G2) if config else N_RESBLOCKS_G2
    use_le = config.get("use_local_enhancer", use_local_enhancer) if config else use_local_enhancer

    model = Pix2PixHDGenerator(
        in_ch=X_CH, out_ch=Y_CH, ngf=ngf,
        use_local_enhancer=use_le,
        n_resblocks_g1=n_resblocks_g1,
        n_downsample_g1=n_downsample_g1,
        n_resblocks_g2=n_resblocks_g2,
    )
    return model


def build_windtransformer_denoiser(config=None, patch_size=None):
    """Build WindTransformer denoiser model."""
    from WindTransformer_windowed import (
        PatchTransformerDenoiser,
        PATCH, EMB, DEPTH, HEADS, MLP_RATIO, DROPOUT,
        ATTN_MODE, WINDOW_SIZE, SHIFT_WINDOWS,
        USE_REFINER, REFINER_USE_YT, REFINER_USE_X, REFINER_X_PROJ_CH,
        recalculate_window_params,
    )

    p = patch_size or (config.get("PATCH", PATCH) if config else PATCH)
    emb = config.get("EMB", EMB) if config else EMB
    depth = config.get("DEPTH", DEPTH) if config else DEPTH
    heads = config.get("HEADS", HEADS) if config else HEADS

    # Update window params if patch size changed
    if p != PATCH:
        recalculate_window_params(p)
        import WindTransformer_windowed as WTW
        ws = WTW.WINDOW_SIZE
    else:
        ws = WINDOW_SIZE

    model = PatchTransformerDenoiser(
        in_ch=X_CH + Y_CH,
        patch=p, emb=emb, depth=depth, heads=heads,
        mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        img_hw=(IMG_H, IMG_W),
        attn_mode=ATTN_MODE, window_size=ws,
        shift_windows=SHIFT_WINDOWS,
        use_refiner=USE_REFINER,
        refiner_use_yt=REFINER_USE_YT,
        refiner_use_x=REFINER_USE_X,
        refiner_x_proj_ch=REFINER_X_PROJ_CH,
    )
    return model


class Pix2PixHDInferenceWrapper(nn.Module):
    """
    Wraps the Pix2PixHD generator with normalization baked in.
    Input:  raw X tensor (8, H, W) - unnormalized
    Output: raw wind field (1, H, W) - denormalized
    """
    def __init__(self, generator, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.generator = generator
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, x):
        # Normalize input
        x_norm = (x - self.x_mean) / self.x_std
        # Forward pass (always use G2 if available)
        y_norm = self.generator(x_norm, use_g2=True)
        # Denormalize output
        y = y_norm * self.y_std + self.y_mean
        return y


class CleanHybridFNOForward(nn.Module):
    """
    Wraps HybridFNO to bypass neuralop's state_dict hooks, which inject dict
    values (like _metadata) that break standard torch.onnx.export (tracing).
    """
    def __init__(self, inner):
        super().__init__()
        # Manually extract submodules to avoid neuralop hooks
        self.fno_embedding = inner.fno.positional_embedding
        self.fno_lifting = inner.fno.lifting
        self.fno_blocks = inner.fno.fno_blocks
        self.fno_projection = inner.fno.projection
        self.attention = inner.attention
        self.refinement = inner.refinement
        self.n_layers = inner.fno.n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sdf = x[:, 0:1]
        baseline = x[:, 3:4]
        
        # FNO forward pass bypassing the top-level FNO module
        x_fno = self.fno_embedding(x)
        x_fno = self.fno_lifting(x_fno)
        for i in range(self.n_layers):
            x_fno = self.fno_blocks(x_fno, i)
        x_fno = self.fno_projection(x_fno)
        
        # HybridFNO refinement
        feat = self.attention(x_fno)
        combined = torch.cat([feat, sdf, baseline], dim=1)
        return self.refinement(combined)


class CleanGeoFNOForward(nn.Module):
    """
    Wraps GeoFNO to bypass neuralop's state_dict hooks and metadata injections.
    """
    def __init__(self, inner):
        super().__init__()
        self.geo_encode = inner.geo_encode
        self.fno_lifting = inner.fno.lifting
        self.fno_blocks = inner.fno.fno_blocks
        self.fno_projection = inner.fno.projection
        self.reconstruct = inner.reconstruct
        self.n_layers = inner.fno.n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sdf = x[:, 0:1, :, :]
        latent_geo = self.geo_encode(x)
        
        x_fno = self.fno_lifting(latent_geo)
        for i in range(self.n_layers):
            x_fno = self.fno_blocks(x_fno, i)
        flow_field = self.fno_projection(x_fno)
        
        return self.reconstruct(flow_field, sdf)


class HybridFNOInferenceWrapper(nn.Module):
    """
    Wraps HybridFNO so it accepts the SAME raw Pix2PixHD-layout input and
    produces the SAME raw wind-speed output.  All preprocessing (channel
    reorder, physical normalization, building-centred coords) and
    postprocessing (delta_u -> mag_U) are baked into the graph.

    Raw input layout  (Pix2PixHD / C# plugin order):
        Ch0: X_coords       (raw metres)
        Ch1: Y_coords       (raw metres)
        Ch2: Z_relative     (raw)
        Ch3: SDF            (raw metres, negative inside buildings)
        Ch4: Bldg_height    (raw metres)
        Ch5: U_over_Uref    (per-pixel, dimensionless)
        Ch6: dir_sin        (-1 to 1)
        Ch7: dir_cos        (-1 to 1)

    FNO internal layout:
        Ch0: SDF / 200
        Ch1: Bldg_height / 50
        Ch2: Z_relative / 10
        Ch3: U_over_Uref * 2
        Ch4: (X - x_centre_of_bldgs) / 500
        Ch5: (Y - y_centre_of_bldgs) / 500
        Ch6: dir_sin
        Ch7: dir_cos

    Output:
        mag_U  =  U_over_Uref * (1 + delta_u)   [raw wind speed, same unit as Pix2PixHD]
    """

    # Raw input channel indices (Pix2PixHD order)
    _R_X   = 0
    _R_Y   = 1
    _R_Z   = 2
    _R_SDF = 3
    _R_BH  = 4
    _R_U   = 5
    _R_SIN = 6
    _R_COS = 7

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, H, W) in raw Pix2PixHD layout

        # --- 1. Extract raw channels ---
        x_coord   = x[:, self._R_X:self._R_X+1]    # (B,1,H,W)
        y_coord   = x[:, self._R_Y:self._R_Y+1]
        z_rel     = x[:, self._R_Z:self._R_Z+1]
        sdf_raw   = x[:, self._R_SDF:self._R_SDF+1]
        bldg_h    = x[:, self._R_BH:self._R_BH+1]
        u_over    = x[:, self._R_U:self._R_U+1]
        dir_sin   = x[:, self._R_SIN:self._R_SIN+1]
        dir_cos   = x[:, self._R_COS:self._R_COS+1]

        # --- 2. Building-centred coordinates ---
        # Compute per-sample building centroid from bldg_h > 0 mask.
        # To keep the graph fully static (no data-dependent indexing which
        # breaks ONNX tracing), we use a weighted-mean formulation:
        #   x_centre = sum(x * mask) / (sum(mask) + eps)
        bldg_mask = (bldg_h > 0).float()                     # (B,1,H,W)
        mask_sum  = bldg_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        x_centre  = (x_coord * bldg_mask).sum(dim=(2, 3), keepdim=True) / mask_sum
        y_centre  = (y_coord * bldg_mask).sum(dim=(2, 3), keepdim=True) / mask_sum

        # Fallback: if no buildings at all, centre on domain mean
        # (rare in practice, but keeps numerics sane)
        no_bldg   = (mask_sum <= 1.0).float()
        domain_xc = x_coord.mean(dim=(2, 3), keepdim=True)
        domain_yc = y_coord.mean(dim=(2, 3), keepdim=True)
        x_centre  = x_centre * (1.0 - no_bldg) + domain_xc * no_bldg
        y_centre  = y_centre * (1.0 - no_bldg) + domain_yc * no_bldg

        # --- 3. Assemble FNO input (channel reorder + physical norm) ---
        fno_input = torch.cat([
            sdf_raw / 200.0,                       # Ch0
            bldg_h  / 50.0,                        # Ch1
            z_rel   / 10.0,                        # Ch2
            u_over  * 2.0,                         # Ch3
            (x_coord - x_centre) / 500.0,          # Ch4
            (y_coord - y_centre) / 500.0,          # Ch5
            dir_sin,                               # Ch6
            dir_cos,                               # Ch7
        ], dim=1)  # (B, 8, H, W)

        # --- 4. Forward through HybridFNO ---
        delta_u = self.model(fno_input)             # (B, 1, H, W)

        # --- 5. Post-process: delta_u -> mag_U ---
        # delta_u = (mag_U - U_ref) / U_ref   =>   mag_U = U_ref * (1 + delta_u)
        # U_ref here is per-pixel U_over_Uref (which IS the local reference speed).
        mag_U = u_over * (1.0 + delta_u)

        return mag_U


def load_stats(stats_path):
    """Load normalization statistics."""
    if not os.path.exists(stats_path):
        print(f"WARNING: Stats file not found at {stats_path}")
        print("  Exporting without baked-in normalization.")
        return None, None, None, None
    sd = torch.load(stats_path, map_location="cpu")
    return sd["x_mean"], sd["x_std"], sd["y_mean"], sd["y_std"]


def export_pix2pixhd(args):
    """Export Pix2PixHD generator to ONNX."""
    print("=" * 60)
    print("Exporting Pix2PixHD Generator to ONNX")
    print("=" * 60)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", {})

    # Build model
    model = build_pix2pixhd_generator(config)

    # Load weights - prefer EMA
    if "ema_generator" in ckpt:
        print("Loading EMA generator weights (best quality).")
        model.load_state_dict(ckpt["ema_generator"])
    elif "generator" in ckpt:
        print("Loading generator weights (no EMA available).")
        model.load_state_dict(ckpt["generator"])
    else:
        raise KeyError("Checkpoint has no 'generator' or 'ema_generator' key.")

    model.eval()

    # Optionally wrap with normalization
    if args.stats:
        x_mean, x_std, y_mean, y_std = load_stats(args.stats)
        if x_mean is not None and args.bake_norm:
            print("Baking normalization into the model (raw input -> raw output).")
            model = Pix2PixHDInferenceWrapper(model, x_mean, x_std, y_mean, y_std)
            model.eval()
        elif x_mean is not None:
            # Save stats alongside ONNX for the API server to use
            stats_out = os.path.splitext(args.output)[0] + "_stats.json"
            stats_dict = {
                "x_mean": x_mean.squeeze().tolist(),
                "x_std": x_std.squeeze().tolist(),
                "y_mean": float(y_mean.squeeze()),
                "y_std": float(y_std.squeeze()),
            }
            with open(stats_out, "w") as f:
                json.dump(stats_dict, f, indent=2)
            print(f"Saved normalization stats to {stats_out}")

    # Create dummy input
    dummy_input = torch.randn(1, X_CH, IMG_H, IMG_W)
    print(f"Input shape: {dummy_input.shape}")

    # Export
    print(f"Exporting to {args.output} ...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["wind_field"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "wind_field": {0: "batch_size"},
        } if args.dynamic_batch else None,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({file_size:.1f} MB)")

    # Verify
    if args.verify:
        verify_onnx(args.output, model, dummy_input)

    # Save config alongside ONNX
    config_out = os.path.splitext(args.output)[0] + "_config.json"
    export_meta = {
        "arch": "pix2pixhd",
        "input_channels": X_CH,
        "output_channels": Y_CH,
        "img_h": IMG_H, "img_w": IMG_W,
        "normalization_baked_in": args.bake_norm,
        "ema_weights": "ema_generator" in ckpt,
        "training_config": config,
    }
    with open(config_out, "w") as f:
        json.dump(export_meta, f, indent=2)
    print(f"Saved export config: {config_out}")


def export_windtransformer(args):
    """Export WindTransformer denoiser to ONNX.

    NOTE: This exports the denoiser network only. The DDIM sampling loop
    must be implemented in your API server code. The denoiser takes:
      - x_cond: (B, 8, H, W) - normalized conditioning input
      - y_t:    (B, 1, H, W) - noisy field at timestep t
      - t:      (B,)         - integer timestep
    And returns:
      - eps:    (B, 1, H, W) - predicted noise
    """
    print("=" * 60)
    print("Exporting WindTransformer Denoiser to ONNX")
    print("=" * 60)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", {})

    # Build model
    model = build_windtransformer_denoiser(config, args.patch_size)

    # Load weights
    if "model_state_dict" in ckpt:
        # WindTransformer uses EMA via apply_to/restore pattern,
        # but best checkpoint usually has the EMA-applied weights
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded model_state_dict.")
    else:
        model.load_state_dict(ckpt)
        print("Loaded legacy state_dict.")

    model.eval()

    # Create dummy inputs
    dummy_x = torch.randn(1, X_CH, IMG_H, IMG_W)      # conditioning
    dummy_yt = torch.randn(1, Y_CH, IMG_H, IMG_W)      # noisy field
    dummy_t = torch.randint(0, 1000, (1,), dtype=torch.long)  # timestep

    print(f"Input shapes: x_cond={dummy_x.shape}, y_t={dummy_yt.shape}, t={dummy_t.shape}")

    # Export
    print(f"Exporting to {args.output} ...")
    torch.onnx.export(
        model,
        (dummy_x, dummy_yt, dummy_t),
        args.output,
        opset_version=args.opset,
        input_names=["x_cond", "y_t", "timestep"],
        output_names=["eps_pred"],
        dynamic_axes={
            "x_cond": {0: "batch_size"},
            "y_t": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "eps_pred": {0: "batch_size"},
        } if args.dynamic_batch else None,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({file_size:.1f} MB)")

    # Verify
    if args.verify:
        verify_onnx(args.output, model, (dummy_x, dummy_yt, dummy_t))

    # Save stats + diffusion schedule for API server
    if args.stats:
        x_mean, x_std, y_mean, y_std = load_stats(args.stats)
        if x_mean is not None:
            stats_out = os.path.splitext(args.output)[0] + "_stats.json"
            stats_dict = {
                "x_mean": x_mean.squeeze().tolist(),
                "x_std": x_std.squeeze().tolist(),
                "y_mean": float(y_mean.squeeze()),
                "y_std": float(y_std.squeeze()),
            }
            with open(stats_out, "w") as f:
                json.dump(stats_dict, f, indent=2)
            print(f"Saved normalization stats: {stats_out}")

    # Save config
    config_out = os.path.splitext(args.output)[0] + "_config.json"
    export_meta = {
        "arch": "windtransformer_denoiser",
        "note": "This is the denoiser only. Implement DDIM sampling loop in your server.",
        "input_channels": X_CH,
        "output_channels": Y_CH,
        "img_h": IMG_H, "img_w": IMG_W,
        "diffusion_timesteps": 1000,
        "recommended_ddim_steps": 50,
        "training_config": config,
    }
    with open(config_out, "w") as f:
        json.dump(export_meta, f, indent=2)
    print(f"Saved export config: {config_out}")


class _RealSpectralConv2d(nn.Module):
    """
    SpectralConv replacement using only real matmul + elementwise ops.

    neuralop's SpectralConv computes  irfft2(W * rfft2(x))  with complex
    weights, which exports to ONNX DFT nodes. DFT is not supported by
    several runtimes (notably DirectML), so instead this module replaces the
    FFT with explicit precomputed DFT basis matrices restricted to the
    (modes1, modes2) selected frequency bins. The resulting graph contains
    only MatMul / Add / Mul / Reshape / Transpose ops - all widely supported.

    Math
    ----
    Let S_u = {-m1/2, ..., m1/2-1} and S_v = {0, ..., m2-1}. We precompute
    four pairs of (cos, sin) basis matrices:

      A_fwd [m1, H]  forward DFT along H (with 1/H factor from norm=forward)
      B_fwd [m2, W]  forward DFT along W (with 1/W factor)
      A_inv [H, m1]  inverse DFT along H
      B_inv [W, m2]  inverse DFT along W

    and split each into its real / imag components. The spectral conv then
    becomes a sequence of batched real matmuls. The implicit Hermitian mirror
    of the one-sided v-spectrum (which irfft2 would fill in automatically)
    is accounted for analytically via a factor-2 weighting on the v > 0
    bins - this makes the real part of the truncated inverse equal to the
    full real-valued inverse transform.
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int,
                 H: int, W: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        self.H = H
        self.W = W
        self.weight_r = nn.Parameter(torch.zeros(in_ch, out_ch, modes1, modes2))
        self.weight_i = nn.Parameter(torch.zeros(in_ch, out_ch, modes1, modes2))
        self.bias: nn.Parameter | None = None

        # Precompute DFT basis matrices (all real, fixed at init)
        u_freqs = torch.arange(modes1, dtype=torch.float32) - (modes1 // 2)
        v_freqs = torch.arange(modes2, dtype=torch.float32)
        h_idx = torch.arange(H, dtype=torch.float32)
        w_idx = torch.arange(W, dtype=torch.float32)
        TWO_PI = 2.0 * torch.pi

        # Forward DFT with norm="forward" bakes 1/(H*W) normalization in.
        #   A_fwd[u, h] = (1/H) * exp(-2pi*i * u * h / H)
        ang_h_fwd = -TWO_PI * u_freqs[:, None] * h_idx[None, :] / H   # [m1, H]
        A_fwd_r = torch.cos(ang_h_fwd) / H
        A_fwd_i = torch.sin(ang_h_fwd) / H
        #   B_fwd[v, w] = (1/W) * exp(-2pi*i * v * w / W)
        ang_w_fwd = -TWO_PI * v_freqs[:, None] * w_idx[None, :] / W   # [m2, W]
        B_fwd_r = torch.cos(ang_w_fwd) / W
        B_fwd_i = torch.sin(ang_w_fwd) / W

        # Inverse DFT (no scaling with norm="forward")
        #   A_inv[h, u] = exp(+2pi*i * u * h / H)
        ang_h_inv = TWO_PI * h_idx[:, None] * u_freqs[None, :] / H    # [H, m1]
        A_inv_r = torch.cos(ang_h_inv)
        A_inv_i = torch.sin(ang_h_inv)
        #   B_inv[w, v] = exp(+2pi*i * v * w / W)
        ang_w_inv = TWO_PI * w_idx[:, None] * v_freqs[None, :] / W    # [W, m2]
        B_inv_r = torch.cos(ang_w_inv)
        B_inv_i = torch.sin(ang_w_inv)

        # Hermitian-mirror doubling: v=0 unchanged, v>0 contributes twice
        # (once as itself, once as conjugate at v' = W-v). Nyquist v=W/2 would
        # also stay at factor 1 but realistic FNO configs never reach it.
        v_scale = torch.full((modes2,), 2.0, dtype=torch.float32)
        v_scale[0] = 1.0

        self.register_buffer("A_fwd_r", A_fwd_r)
        self.register_buffer("A_fwd_i", A_fwd_i)
        self.register_buffer("B_fwd_r", B_fwd_r)
        self.register_buffer("B_fwd_i", B_fwd_i)
        self.register_buffer("A_inv_r", A_inv_r)
        self.register_buffer("A_inv_i", A_inv_i)
        self.register_buffer("B_inv_r", B_inv_r)
        self.register_buffer("B_inv_i", B_inv_i)
        self.register_buffer("v_scale", v_scale)

    @classmethod
    def from_complex(cls, in_ch, out_ch, modes1, modes2, w_complex, H, W):
        """Build from a neuralop complex weight tensor."""
        m = cls(in_ch, out_ch, modes1, modes2, H, W)
        m.weight_r.data.copy_(w_complex.real[:in_ch, :out_ch, :modes1, :modes2].contiguous())
        m.weight_i.data.copy_(w_complex.imag[:in_ch, :out_ch, :modes1, :modes2].contiguous())
        return m

    @property
    def n_modes(self):
        return [self.modes1, self.modes2]

    @n_modes.setter
    def n_modes(self, value):
        # FNOBlocks may try to set n_modes; ignore silently (fixed at init).
        pass

    def transform(self, x: torch.Tensor, output_shape=None) -> torch.Tensor:
        """Passthrough used by FNOBlocks for resolution changes (none here)."""
        return x

    def forward(self, x: torch.Tensor, output_shape=None) -> torch.Tensor:
        B = x.shape[0]
        C, O = self.in_ch, self.out_ch
        m1, m2 = self.modes1, self.modes2

        # 1. Forward DFT along H (real input -> complex coeffs at m1 bins)
        #    x: [B,C,H,W]  ->  [B,C,m1,W]
        x_u_r = torch.matmul(self.A_fwd_r, x)
        x_u_i = torch.matmul(self.A_fwd_i, x)

        # 2. Forward DFT along W (m2 one-sided bins)
        #    x_u [B,C,m1,W] @ B_fwd.T [W,m2]  ->  [B,C,m1,m2]
        B_fwd_r_t = self.B_fwd_r.transpose(0, 1)
        B_fwd_i_t = self.B_fwd_i.transpose(0, 1)
        x_uv_r = torch.matmul(x_u_r, B_fwd_r_t) - torch.matmul(x_u_i, B_fwd_i_t)
        x_uv_i = torch.matmul(x_u_r, B_fwd_i_t) + torch.matmul(x_u_i, B_fwd_r_t)

        # 3. Spectral weight multiply via batched matmul
        #    y[b,o,u,v] = sum_i W[i,o,u,v] * x_uv[b,i,u,v]
        #    Reshape so (u,v) becomes the batch dim of a bmm.
        xp_r = x_uv_r.permute(2, 3, 0, 1).reshape(m1 * m2, B, C)
        xp_i = x_uv_i.permute(2, 3, 0, 1).reshape(m1 * m2, B, C)
        wp_r = self.weight_r.permute(2, 3, 0, 1).reshape(m1 * m2, C, O)
        wp_i = self.weight_i.permute(2, 3, 0, 1).reshape(m1 * m2, C, O)
        yp_r = torch.bmm(xp_r, wp_r) - torch.bmm(xp_i, wp_i)
        yp_i = torch.bmm(xp_r, wp_i) + torch.bmm(xp_i, wp_r)
        y_uv_r = yp_r.reshape(m1, m2, B, O).permute(2, 3, 0, 1)
        y_uv_i = yp_i.reshape(m1, m2, B, O).permute(2, 3, 0, 1)

        # 4. Hermitian mirror doubling for v > 0
        y_uv_r = y_uv_r * self.v_scale
        y_uv_i = y_uv_i * self.v_scale

        # 5. Inverse DFT along W (m2 -> W)
        #    y_uv [B,O,m1,m2] @ B_inv.T [m2,W]  ->  [B,O,m1,W]
        B_inv_r_t = self.B_inv_r.transpose(0, 1)
        B_inv_i_t = self.B_inv_i.transpose(0, 1)
        y_uw_r = torch.matmul(y_uv_r, B_inv_r_t) - torch.matmul(y_uv_i, B_inv_i_t)
        y_uw_i = torch.matmul(y_uv_r, B_inv_i_t) + torch.matmul(y_uv_i, B_inv_r_t)

        # 6. Inverse DFT along H, take real part only (output is real)
        #    A_inv [H,m1] @ y_uw [B,O,m1,W]  ->  [B,O,H,W]
        out = torch.matmul(self.A_inv_r, y_uw_r) - torch.matmul(self.A_inv_i, y_uw_i)

        if self.bias is not None:
            out = out + self.bias
        return out


def _replace_spectral_convs(model, H, W):
    """
    Walk the model tree and replace every neuralop SpectralConv module
    with an equivalent _RealSpectralConv2d whose weights come from the
    original complex parameters. H, W are the fixed spatial dims of the
    FNO grid (needed to precompute the DFT basis).

    Returns the mutated model (in-place).
    """
    try:
        from neuralop.layers.spectral_convolution import SpectralConv
    except ImportError:
        raise RuntimeError("neuralop not installed; cannot find SpectralConv to replace.")

    def _recurse(parent, name, mod):
        for child_name, child in list(mod.named_children()):
            _recurse(mod, child_name, child)

        if isinstance(mod, SpectralConv):
            # neuralop SpectralConv stores weight under .weight (may be a
            # FactorizedTensor or a plain Parameter/tensor). Reconstruct it.
            w = mod.weight
            if hasattr(w, "to_tensor"):          # FactorizedTensor
                w = w.to_tensor()
            elif hasattr(w, "tensor"):            # WrappedParameter
                w = w.tensor
            # w shape: [in_ch, out_ch, modes1, modes2]
            in_ch, out_ch, m1, m2 = w.shape
            real_conv = _RealSpectralConv2d.from_complex(
                in_ch, out_ch, m1, m2, w.detach(), H, W
            )
            if mod.bias is not None:
                real_conv.bias = nn.Parameter(mod.bias.detach().clone())
            setattr(parent, name, real_conv)

    for name, child in list(model.named_children()):
        _recurse(model, name, child)
    return model


def _replace_fno2d_spectral_convs(model, H, W):
    """Replaces fno2d pure torch SpectralConv2d custom blocks with real mats if needed."""
    from core.models.fno2d import SpectralConv2d
    
    def _recurse(parent, name, mod):
        for child_name, child in list(mod.named_children()):
            _recurse(mod, child_name, child)
        if isinstance(mod, SpectralConv2d):
            w = mod.weights1.detach() # complex
            in_ch, out_ch, m1, m2 = w.shape
            w_complex = torch.zeros((in_ch, out_ch, m1, m2), dtype=torch.complex64)
            w_complex.real = w.real
            w_complex.imag = w.imag
            real_conv = _RealSpectralConv2d.from_complex(in_ch, out_ch, m1, m2, w_complex, H, W)
            setattr(parent, name, real_conv)
            
    for name, child in list(model.named_children()):
        _recurse(model, name, child)
    return model


def export_fno_variants(args):
    """Unified Export for Standard, Hybrid, PINN, and Geo FNO Models."""
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from tools.infer_csv import build_model, load_weights

    print("=" * 60)
    print(f"Exporting {args.arch.upper()} FNO to ONNX")
    print("=" * 60)

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_weights(args.checkpoint, "cpu")
    model = build_model(args.arch, state_dict, "cpu")
    model.eval()

    # Verify forward pass works before weight replacement
    dummy_fno_input = torch.randn(1, X_CH, IMG_H, IMG_W)
    print(f"FNO-layout input shape: {dummy_fno_input.shape}")
    with torch.no_grad():
        out_ref = model(dummy_fno_input)
    print(f"Output shape: {out_ref.shape}")

    # NeuralOp or FNO2d Replacement
    print("Replacing complex math operators with real-matmul ONNX layers ...")
    if args.arch == "standard":
        try:
            model = _replace_fno2d_spectral_convs(model, IMG_H, IMG_W)
        except Exception as e:
            print(f"Standard FNO2d Replacement Skipped: {e}")
    else:
        model = _replace_spectral_convs(model, IMG_H, IMG_W)
        
    model.eval()

    # Determine structural interceptor
    if args.arch in ["hybrid", "pinn"]:
        for name, mod in model.named_modules():
            if hasattr(mod, '_save_to_state_dict') and type(mod)._save_to_state_dict is not nn.Module._save_to_state_dict:
                mod._save_to_state_dict = nn.Module._save_to_state_dict.__get__(mod, type(mod))
        clean_model = CleanHybridFNOForward(model) # PINN shares the exact forward loop
    elif args.arch == "geo":
        for name, mod in model.named_modules():
            if hasattr(mod, '_save_to_state_dict') and type(mod)._save_to_state_dict is not nn.Module._save_to_state_dict:
                mod._save_to_state_dict = nn.Module._save_to_state_dict.__get__(mod, type(mod))
        clean_model = CleanGeoFNOForward(model)
    else:
        # Standard FNO is already clean pure torch
        clean_model = model
        
    clean_model.eval()

    # Sanity check validation
    with torch.no_grad():
        out_real = clean_model(dummy_fno_input)
    max_diff = (out_ref - out_real).abs().max().item()
    print(f"  Max diff after replacement: {max_diff:.2e}")
    if max_diff > 1e-1:
        print(f"WARN: Real-matmul replacement produced large error margin ({max_diff:.2e}). Export continuing but check ONNX precision.")

    bake_norm = getattr(args, "bake_norm", False)
    if bake_norm:
        print("Baking preprocessing+postprocessing into the model")
        export_model = HybridFNOInferenceWrapper(clean_model)
        export_model.eval()
    else:
        export_model = clean_model

    dummy_input = torch.randn(1, 8, IMG_H, IMG_W)

    print(f"Exporting to {args.output} ...")
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            args.output,
            opset_version=17,
            input_names=["input"],
            output_names=["mag_U"] if bake_norm else ["delta_u"],
            do_constant_folding=True,
        )

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({file_size:.1f} MB)")

    if args.verify:
        verify_onnx(args.output, export_model, dummy_input)

    # Save metadata
    config_out = os.path.splitext(args.output)[0] + "_config.json"
    export_meta = {
        "arch": args.arch,
        "input_channels": X_CH,
        "output_channels": Y_CH,
        "img_h": IMG_H, "img_w": IMG_W,
        "normalization_baked_in": bake_norm,
    }
    if bake_norm:
        export_meta["note"] = "Input is raw Pix2PixHD-layout. Preprocessing and postprocessing (delta_u -> mag_U) are baked in."
    else:
        export_meta["note"] = "Output is delta_u = (mag_U - U_ref) / U_ref, clipped [-2, 10]"
    with open(config_out, "w") as f:
        json.dump(export_meta, f, indent=2)
    print(f"Saved export config: {config_out}")


def verify_onnx(onnx_path, pytorch_model, dummy_inputs):
    """Verify ONNX output matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nWARN onnxruntime not installed. Skipping verification.")
        print("  Install with: pip install onnxruntime  (CPU)")
        print("            or: pip install onnxruntime-gpu  (GPU)")
        return

    print("\nVerifying ONNX against PyTorch...")

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        if isinstance(dummy_inputs, tuple):
            pt_out = pytorch_model(*dummy_inputs)
        else:
            pt_out = pytorch_model(dummy_inputs)

    if isinstance(pt_out, tuple):
        pt_out = pt_out[0]
    pt_np = pt_out.numpy()

    # ONNX Runtime inference (CPU EP - known to work with all supported ops)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]

    if isinstance(dummy_inputs, tuple):
        ort_inputs = {name: inp.numpy() for name, inp in zip(input_names, dummy_inputs)}
    else:
        ort_inputs = {input_names[0]: dummy_inputs.numpy()}

    ort_out = sess.run(None, ort_inputs)[0]

    # Compare
    max_diff = np.max(np.abs(pt_np - ort_out))
    mean_diff = np.mean(np.abs(pt_np - ort_out))
    print(f"  Max  absolute diff: {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("  OK  ONNX output matches PyTorch (within FP32 tolerance).")
    elif max_diff < 1e-2:
        print("  WARN Small numerical differences (likely FP16 vs FP32). Acceptable for inference.")
    else:
        print("  FAIL Large differences detected! ONNX export may have issues.")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch wind field models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pix2PixHD (with normalization baked in)
  python export_onnx.py --arch pix2pixhd \\
      --checkpoint results_pix2pixhd/checkpoints/ckpt_best.pt \\
      --stats results_pix2pixhd/checkpoints/stats.pt \\
      --bake-norm --verify

  # WindTransformer denoiser
  python export_onnx.py --arch windtransformer \\
      --checkpoint results/checkpoints/wind_model_best.pth \\
      --stats results/checkpoints/stats.pt \\
      --verify
        """
    )
    parser.add_argument("--arch", required=True, choices=["pix2pixhd", "windtransformer", "hybrid", "standard", "geo", "pinn"],
                        help="Model architecture to export")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to .pt or .pth checkpoint file")
    parser.add_argument("--stats", type=str, default=None,
                        help="Path to stats.pt (normalization statistics)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: <arch>_model.onnx)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--dynamic-batch", action="store_true", default=True,
                        help="Enable dynamic batch size (default: True)")
    parser.add_argument("--bake-norm", action="store_true",
                        help="Bake normalization into the model (raw->raw I/O)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output matches PyTorch (requires onnxruntime)")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="[WindTransformer only] Override patch size")

    args = parser.parse_args()

    # Default output name
    if args.output is None:
        args.output = f"{args.arch}_model_export.onnx"

    # Auto-detect stats path if not provided
    if args.stats is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        candidate = os.path.join(ckpt_dir, "stats.pt")
        if os.path.exists(candidate):
            args.stats = candidate
            print(f"Auto-detected stats: {candidate}")

    if args.arch == "pix2pixhd":
        export_pix2pixhd(args)
    elif args.arch == "windtransformer":
        export_windtransformer(args)
    elif args.arch in ["hybrid", "pinn", "geo", "standard"]:
        export_fno_variants(args)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  ONNX model: {args.output}")
    print(f"  Size: {os.path.getsize(args.output) / (1024*1024):.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
