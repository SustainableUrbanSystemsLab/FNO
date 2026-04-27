#!/usr/bin/env python3
"""
FNO CSV Inference Script — 4-Channel Output (U, k, U_roof, k_roof)
=====================================================================
Loads a trained FNO model (Standard, Hybrid, PINN, Geo), reads CSV input(s),
runs inference, and saves:
  1. Grasshopper-compatible prediction CSV
  2. Per-channel 4-panel PNG visualizations (Domain | GT | Pred | Diff+Metrics)
     matching the Conditional Transformer visualization format.

Masking rules (identical to pix2pix_hd_uk_roof.py):
  - ch0,1 (U, k):          mask *building interiors*  — no flow inside solid obstacles
  - ch2,3 (U_roof, k_roof): mask *open ground*        — roof data only exists on buildings

Usage:
  python tools/infer_csv.py --csv test_csv/ML_FormFlux_1_135.csv \\
      --model geo_fno_weights.pth --model_type geo
"""
import os, sys, torch, numpy as np, pandas as pd, glob, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.models.fno2d import FNO2d
from core.models.hybrid import HybridFNO
from core.models.pinn_fno import PINNFNO
from core.models.geo_fno import GeoFNO
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from pipelines.train.distributed import CHANNEL_NAMES, CHANNEL_VMAXES


# ================================================================
# Domain Setup Panel (unchanged from original)
# ================================================================

def plot_domain_panel(ax, x_input):
    H, W = x_input.shape[1], x_input.shape[2]
    cx, cy = W // 2, H // 2
    R = min(H, W) // 2 - 5

    sin_val, cos_val = float(x_input[6].mean()), float(x_input[7].mean())
    mag = np.sqrt(sin_val**2 + cos_val**2)
    if mag < 1e-6: sin_val, cos_val = 1.0, 0.0
    wind_dir_deg = np.degrees(np.arctan2(cos_val, sin_val))
    inlet_center_deg = (wind_dir_deg + 180) % 360

    ax.set_facecolor("white")
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal")

    for i in range(1, 8):
        ax.add_patch(plt.Circle((cx, cy), R * i / 7, fill=False, edgecolor="#d0d0d0", linewidth=0.4, zorder=1))
    for angle_deg in range(0, 360, 30):
        rad = np.radians(angle_deg)
        ax.plot([cx, cx + R * np.cos(rad)], [cy, cy + R * np.sin(rad)], color="#d0d0d0", linewidth=0.4, zorder=1)

    bldg_mask = x_input[1] > 0
    if np.any(bldg_mask):
        bldg_rgba = np.zeros((H, W, 4), dtype=np.float32)
        bldg_rgba[bldg_mask] = [0.15, 0.15, 0.15, 1.0]
        ax.imshow(bldg_rgba, origin="lower", extent=[0, W, 0, H], zorder=2)
        ys, xs = np.where(bldg_mask)
        gan_r = np.clip(np.sqrt(((xs - cx)**2 + (ys - cy)**2).max()) * 1.15, R * 0.35, R * 0.85)
    else: gan_r = R * 0.6

    ax.add_patch(plt.Circle((cx, cy), gan_r, fill=False, edgecolor="goldenrod", linestyle="--", linewidth=1.5, zorder=3))
    ax.add_patch(mpatches.Arc((cx, cy), R * 2, R * 2, angle=0, theta1=inlet_center_deg-90, theta2=inlet_center_deg+90, edgecolor="royalblue", linewidth=2.5, fill=False, zorder=4))
    ax.add_patch(mpatches.Arc((cx, cy), R * 2, R * 2, angle=0, theta1=wind_dir_deg-90, theta2=wind_dir_deg+90, edgecolor="red", linewidth=2.5, fill=False, zorder=4))

    mid_angle = np.radians(inlet_center_deg)
    dx_arrow, dy_arrow = -28 * np.cos(mid_angle), -28 * np.sin(mid_angle)
    for i in range(9):
        frac = (i + 0.5) / 9
        angle = np.radians(inlet_center_deg - 90 + frac * 180)
        xs, ys = cx + R * np.cos(angle), cy + R * np.sin(angle)
        ax.annotate("", xy=(xs + dx_arrow, ys + dy_arrow), xytext=(xs, ys), arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.3), zorder=5)

    ax.text(cx + (R+22)*np.cos(np.radians(inlet_center_deg+45)), cy + (R+22)*np.sin(np.radians(inlet_center_deg+45)), "inlet", color="royalblue", fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8))
    ax.text(cx + (R+22)*np.cos(np.radians(wind_dir_deg+45)), cy + (R+22)*np.sin(np.radians(wind_dir_deg+45)), "outlet", color="red", fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_title("Domain Setup", pad=36, fontsize=15, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


# ================================================================
# Per-Channel Visualization (matching Conditional Transformer)
# ================================================================

def save_pred_vs_true(y_pred, y_true, out_path, x_input=None,
                      has_gt=True, vmax=2.0, channel_label="",
                      extra_mask=None):
    """4-panel visualization per channel.

    Parameters
    ----------
    y_pred, y_true : ndarray (H, W)
    x_input : ndarray (8, H, W) — conditioning input for domain panel
    extra_mask : bool ndarray (H, W) — True where pixels should be hidden (white).
                 For pedestrian channels: building interior.
                 For roof channels: open ground (no building).
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.maximum(y_pred, 0)

    H, W = y_true.shape
    cy_m, cx_m = H // 2, W // 2
    rad_m = min(H, W) // 2 - 5
    Yc_m, Xc_m = np.ogrid[:H, :W]
    outside = np.sqrt((Xc_m - cx_m)**2 + (Yc_m - cy_m)**2) >= rad_m

    # Combined mask: circular boundary + physical mask (building/open-ground)
    hide = outside if extra_mask is None else (outside | extra_mask)

    cmap_field = matplotlib.cm.get_cmap("viridis").copy()
    cmap_field.set_bad("white")
    cmap_rdbu = matplotlib.cm.get_cmap("RdBu").copy()
    cmap_rdbu.set_bad("white")

    y_true_vis = np.ma.masked_where(hide, y_true)
    y_pred_vis = np.ma.masked_where(hide, np.maximum(y_pred, 0))

    suffix = f" ({channel_label})" if channel_label else ""

    if x_input is not None:
        fig = plt.figure(figsize=(24, 6))
        grid_size = (1, 4)
        has_input = True
    else:
        fig = plt.figure(figsize=(15, 5))
        grid_size = (1, 3)
        has_input = False

    plot_idx = 1

    # --- Panel 1: Domain Setup ---
    if has_input:
        ax0 = plt.subplot(grid_size[0], grid_size[1], plot_idx); plot_idx += 1
        plot_domain_panel(ax0, x_input)

    # --- Panel 2: Ground Truth ---
    ax1 = plt.subplot(grid_size[0], grid_size[1], plot_idx); plot_idx += 1
    title_gt = f"Ground Truth{suffix}" if has_gt else f"Ground Truth (N/A){suffix}"
    ax1.set_title(title_gt, pad=36, fontsize=15, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    im1 = ax1.imshow(y_true_vis, cmap=cmap_field, vmin=0.0, vmax=vmax, origin="lower")
    ax1.add_patch(plt.Circle((cx_m, cy_m), rad_m, fill=False, edgecolor="black", linewidth=0.8, zorder=10))
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_visible(False)

    # --- Panel 3: Prediction ---
    ax2 = plt.subplot(grid_size[0], grid_size[1], plot_idx); plot_idx += 1
    ax2.set_title(f"Prediction{suffix}", pad=36, fontsize=15, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    ax2.imshow(y_pred_vis, cmap=cmap_field, vmin=0.0, vmax=vmax, origin="lower")
    ax2.add_patch(plt.Circle((cx_m, cy_m), rad_m, fill=False, edgecolor="black", linewidth=0.8, zorder=10))
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_visible(False)

    # --- Panel 4: Diff + Metrics ---
    ax3 = plt.subplot(grid_size[0], grid_size[1], plot_idx)
    diff = y_pred - y_true
    diff_vis = np.ma.masked_where(hide, diff)

    domain_mask = ~hide

    if has_gt and np.any(domain_mask):
        diff_m = diff[domain_mask]
        abs_m  = np.abs(diff_m)
        mae    = np.mean(abs_m)
        rmse   = np.sqrt(np.mean(diff_m**2))

        gt_m = y_true[domain_mask]
        valid_mape = np.abs(gt_m) > 0.1
        mape = np.mean(abs_m[valid_mape] / np.abs(gt_m[valid_mape])) * 100.0 if np.any(valid_mape) else 0.0

        ss_res = np.sum(diff_m ** 2)
        ss_tot = np.sum((gt_m - np.mean(gt_m)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        try:
            from skimage.metrics import structural_similarity as ssim_func
            pr_in = y_pred[domain_mask]
            data_range = max(gt_m.max(), pr_in.max()) - min(gt_m.min(), pr_in.min())
            if data_range < 1e-8:
                ssim_val = 1.0
            else:
                _, ssim_map = ssim_func(y_true, y_pred, data_range=data_range, full=True)
                ssim_val = float(ssim_map[domain_mask].mean())
            ssim_str = f"{ssim_val:.3f}"
        except Exception:
            ssim_str = "N/A"

        def _grad_corr(pred, true, mask):
            pdx = np.diff(pred, axis=1, prepend=pred[:, :1])
            pdy = np.diff(pred, axis=0, prepend=pred[:1, :])
            tdx = np.diff(true, axis=1, prepend=true[:, :1])
            tdy = np.diff(true, axis=0, prepend=true[:1, :])
            pg = np.concatenate([pdx.flatten(), pdy.flatten()])
            tg = np.concatenate([tdx.flatten(), tdy.flatten()])
            mm = np.concatenate([mask.flatten(), mask.flatten()])
            pg, tg = pg[mm], tg[mm]
            if np.std(pg) < 1e-8 or np.std(tg) < 1e-8: return 0.0
            return float(np.corrcoef(pg, tg)[0, 1])

        grad_corr = _grad_corr(y_pred, y_true, domain_mask)
        line1 = f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | MAPE:{mape:.1f}%"
        line2 = f"SSIM:{ssim_str} | GradCorr:{grad_corr:.3f} | R\u00b2:{r2:.3f}"
    else:
        line1 = "MAE:N/A | RMSE:N/A | MAPE:N/A"
        line2 = "SSIM:N/A | GradCorr:N/A | R\u00b2:N/A"

    im3 = ax3.imshow(diff_vis if has_gt else np.ma.masked_where(hide, np.zeros_like(diff)),
                     cmap=cmap_rdbu, vmin=-vmax, vmax=vmax, origin="lower")
    ax3.add_patch(plt.Circle((cx_m, cy_m), rad_m, fill=False, edgecolor="black", linewidth=0.8, zorder=10))
    ax3.set_title(f"Diff (Pred - GT){suffix}" if has_gt else f"Diff (N/A){suffix}",
                  pad=36, fontsize=15, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values(): sp.set_visible(False)

    # --- Colorbars & Metrics ---
    plt.tight_layout(pad=1.5)
    fig.canvas.draw()

    # Align title heights
    all_axes = [ax0, ax1, ax2, ax3] if has_input else [ax1, ax2, ax3]
    fig_ys = []
    for a in all_axes:
        tx, ty = a.title.get_position()
        fig_y = a.transAxes.transform((tx, ty))[1]
        fig_ys.append(fig_y)
    target_fig_y = max(fig_ys)
    for a in all_axes:
        tx, _ = a.title.get_position()
        new_axes_y = a.transAxes.inverted().transform((0, target_fig_y))[1]
        a.title.set_position((tx, new_axes_y))
    fig.canvas.draw()

    bb1 = ax1.get_position()
    bb2 = ax2.get_position()
    bb3 = ax3.get_position()
    cb_w = bb3.width
    cb_y = min(bb1.y0, bb3.y0) - 0.12

    shared_center = (bb1.x0 + bb2.x1) / 2
    cax_shared = fig.add_axes([shared_center - cb_w/2, cb_y, cb_w, 0.025])
    fig.colorbar(im1, cax=cax_shared, orientation="horizontal")

    diff_center = bb3.x0 + bb3.width / 2
    cax_diff = fig.add_axes([diff_center - cb_w/2, cb_y, cb_w, 0.025])
    fig.colorbar(im3, cax=cax_diff, orientation="horizontal")

    metrics_y = cb_y - 0.10
    fig.text(diff_center, metrics_y, f"{line1}\n{line2}",
             ha="center", va="top", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close("all")
    print(f"  -> Saved: {out_path}")
    if has_gt and np.any(domain_mask):
        print(f"     [{channel_label}] MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.1f}% "
              f"SSIM={ssim_str} GradCorr={grad_corr:.3f} R\u00b2={r2:.4f}")


# ================================================================
# Model Loading
# ================================================================

def load_weights(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    return {k.replace("module.", ""): v for k, v in sd.items() if k != "_metadata"}

def build_model(model_type, state_dict, DEVICE):
    import tomllib
    # Load defaults
    with open(os.path.join(os.path.dirname(__file__), '../config.toml'), "rb") as f:
        config = tomllib.load(f)
    modes = config.get("model", {}).get("modes1", 32)
    width = config.get("model", {}).get("width", 64)
    layers = config.get("model", {}).get("n_layers", 4)

    # Auto-detect parameters from state_dict to match the exact checkpoint
    if "in_proj.weight" in state_dict:
        width = state_dict["in_proj.weight"].shape[0]
        if "fourier_layers.0.0.weights1" in state_dict:
            w_shape = state_dict["fourier_layers.0.0.weights1"].shape
            modes = w_shape[2]

    elif "fno.fno_blocks.convs.0.bias" in state_dict:
        width = state_dict["fno.fno_blocks.convs.0.bias"].shape[0]
        if "fno.fno_blocks.convs.0.weight.tensor" in state_dict:
            modes = state_dict["fno.fno_blocks.convs.0.weight.tensor"].shape[2]

    # Auto-detect out_channels
    out_ch = 1
    for key in ["out_proj.2.weight", "refinement.4.weight", "reconstruct.decode.2.weight"]:
        if key in state_dict:
            out_ch = state_dict[key].shape[0]
            break

    print(f"Building {model_type} architecture -> Modes: {modes}, Width: {width}, OutChannels: {out_ch}")

    # Init Classes
    typ = model_type.lower()
    if typ == "standard":
        model = FNO2d(in_channels=8, out_channels=out_ch, modes1=modes, modes2=modes, width=width, n_layers=layers)
    elif typ == "hybrid":
        model = HybridFNO(in_channels=8, out_channels=out_ch, n_modes=(modes, modes), hidden_channels=width, n_layers=layers)
    elif typ == "pinn":
        model = PINNFNO(in_channels=8, out_channels=out_ch, n_modes=(modes, modes), hidden_channels=width, n_layers=layers)
    elif typ == "geo":
        model = GeoFNO(in_channels=8, out_channels=out_ch, n_modes=(modes, modes), hidden_channels=width, n_layers=layers)
    else:
        raise ValueError("Invalid model type. Choose from: standard, hybrid, pinn, geo")

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model


# ================================================================
# CSV Inference — Multi-Channel
# ================================================================

def _safe_col(df, candidates, default=None):
    """Return the first matching column name from candidates, else default."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _clean_flat(arr):
    """NaN/Inf -> 0 for ground-truth arrays."""
    arr = arr.copy()
    bad = ~np.isfinite(arr)
    if np.any(bad):
        arr[bad] = 0.0
    return arr


def process_single_csv(csv_path, model, DEVICE, output_dir=None):
    """Run inference on a single CSV and produce per-channel visualizations."""
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)

    # Column renaming (Grasshopper / CT exports)
    rename_map = {"X": "X_coords", "Y": "Y_coords", "x": "X_coords", "y": "Y_coords", "U_at_z": "U_over_Uref"}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Build input tensor
    gh = {c: df[c].to_numpy() for c in ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_coords", "Y_coords", "dir_sin", "dir_cos"]}
    X_batch, _ = build_input_tensor_from_gh(gh, device="cpu")
    nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df["X_coords"], df["Y_coords"])
    x_input_np = X_batch[0].numpy()  # (8, H, W)

    # --- Detect how many output channels the model produces ---
    with torch.no_grad():
        X_device = X_batch[0].unsqueeze(0).to(DEVICE)
        pred_all = model(X_device)  # (1, out_ch, H, W)
    out_ch = pred_all.shape[1]
    print(f"  Model produces {out_ch} output channel(s)")

    # --- Collect ground-truth columns if available ---
    gt_columns = {
        0: _safe_col(df, ["mag_U", "actual_U", "mag_U_dimensionless"]),
        1: _safe_col(df, ["k", "k_from_U"]),
        2: _safe_col(df, ["mag_U_roof"]),
        3: _safe_col(df, ["k_roof", "k_roof_from_U"]),
    }

    # --- Per-channel masking (matching CT convention) ---
    bldg_mask_flat = (df["Bldg_height"].to_numpy() > 0)
    is_inside_building_flat = df['Bldg_height'] > (df['Z_relative'] + 0.01)

    # Reference wind speed for delta_u -> mag_U denormalization
    u_ref_flat = df["U_over_Uref"].to_numpy()

    # --- Output paths ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(csv_path))[0]
    else:
        base = os.path.splitext(csv_path)[0]
        output_dir = os.path.dirname(csv_path) or "."

    # --- Denormalize & visualize each channel ---
    df_out = df.copy()
    channel_names = CHANNEL_NAMES[:out_ch]
    channel_vmaxes = CHANNEL_VMAXES[:out_ch]

    # Build per-channel 2D masks for visualization
    # Channel 0,1 (pedestrian): mask building interiors
    # Channel 2,3 (roof):      mask open ground
    bldg_grid = np.zeros((ny, nx), dtype=bool)
    for i, (iy, ix) in enumerate(idx_map):
        bldg_grid[iy, ix] = bldg_mask_flat[i]
    open_gnd_grid = ~bldg_grid
    ch_extra_masks = [bldg_grid, bldg_grid, open_gnd_grid, open_gnd_grid]

    for ch_idx, (ch_name, ch_vmax) in enumerate(zip(channel_names, channel_vmaxes)):
        if ch_idx >= out_ch:
            break

        # Extract raw prediction for this channel
        pred_flat = np.array([pred_all[0, ch_idx, iy, ix].cpu().item() for (iy, ix) in idx_map])

        # Denormalize wind speed channels (ch0=U, ch2=U_roof): delta_u -> mag_U
        if ch_idx in (0, 2):
            pred_mag_flat = np.clip(u_ref_flat * (pred_flat + 1.0), 0.0, None)
        else:
            # TKE channels (ch1=k, ch3=k_roof): already raw, just clip non-negative
            pred_mag_flat = np.maximum(pred_flat, 0.0)

        # Physical masking: zero out inside buildings for pedestrian channels
        if ch_idx in (0, 1) and np.any(is_inside_building_flat):
            pred_mag_flat[is_inside_building_flat] = 0.0

        # Save prediction to CSV
        df_out[f'{ch_name}_pred'] = pred_mag_flat

        # Build 2D grids for visualization
        pred_grid = np.zeros((ny, nx))
        gt_grid = np.zeros((ny, nx))
        has_gt = gt_columns.get(ch_idx) is not None

        if has_gt:
            gt_flat = _clean_flat(df[gt_columns[ch_idx]].to_numpy())

        for i, (iy, ix) in enumerate(idx_map):
            pred_grid[iy, ix] = pred_mag_flat[i]
            if has_gt:
                gt_grid[iy, ix] = gt_flat[i]

        # Save per-channel visualization
        png_out = os.path.join(output_dir, f"{base}_pred_{ch_name}.png")
        save_pred_vs_true(
            pred_grid, gt_grid, png_out,
            x_input=x_input_np,
            has_gt=has_gt,
            vmax=ch_vmax,
            channel_label=ch_name,
            extra_mask=ch_extra_masks[ch_idx] if ch_idx < len(ch_extra_masks) else None,
        )

    # Save combined CSV
    csv_out = os.path.join(output_dir, f"{base}_pred.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"  -> Saved prediction CSV: {csv_out}")
    print(f"  -> Channels visualized: {', '.join(channel_names)}")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="FNO CSV Inference — Multi-Channel")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file or directory containing CSVs")
    parser.add_argument("--model", type=str, required=True, help="Path to your .pth weights file")
    parser.add_argument("--model_type", type=str, required=True, choices=["standard", "hybrid", "pinn", "geo"], help="The architecture used to train the weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save the output CSVs")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = load_weights(args.model, DEVICE)
    model = build_model(args.model_type, state_dict, DEVICE)

    if os.path.isdir(args.csv):
        files = glob.glob(os.path.join(args.csv, "*.csv"))
        files = [f for f in files if "_pred" not in os.path.basename(f)]
        for f in files: process_single_csv(f, model, DEVICE, args.output_dir)
    else:
        process_single_csv(args.csv, model, DEVICE, args.output_dir)

if __name__ == "__main__":
    main()
