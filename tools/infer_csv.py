#!/usr/bin/env python3
"""
FNO CSV Inference Script
=========================
Loads a trained FNO model (Standard, Hybrid, PINN, Geo), reads CSV input(s),
runs inference, and saves the prediction back out as a Grasshopper-compatible
CSV file + a visual PNG plot exactly matching the Conditional Transformer format.

Usage:
  python tools/infer_csv.py --csv test_csv/ML_FormFlux_1_135.csv --model geo_fno_weights.pth --model_type geo
"""
import os, sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, glob, argparse
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.models.fno2d import FNO2d
from core.models.hybrid import HybridFNO
from core.models.pinn_fno import PINNFNO
from core.models.geo_fno import GeoFNO
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

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
        bldg_rgba[bldg_mask] = [0.0, 0.0, 0.0, 1.0]
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

def save_pred_vs_true(y_pred, y_true, out_path, x_input, has_gt=True):
    H, W = y_pred.shape
    cy_m, cx_m = H // 2, W // 2
    Yc_m, Xc_m = np.ogrid[:H, :W]
    outside = np.sqrt((Xc_m - cx_m)**2 + (Yc_m - cy_m)**2) >= (min(H, W)//2 - 5)
    
    y_true_vis = np.ma.masked_where(outside, y_true)
    y_pred_vis = np.ma.masked_where(outside, np.maximum(y_pred, 0))
    diff_vis = np.ma.masked_where(outside, y_pred - y_true)

    fig = plt.figure(figsize=(24, 8)) 
    ax0 = plt.subplot(1, 4, 1); plot_domain_panel(ax0, x_input)
    
    ax1 = plt.subplot(1, 4, 2)
    ax1.set_title("Ground Truth" if has_gt else "Ground Truth (N/A)", pad=36, fontsize=15, fontweight="bold", bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    im1 = ax1.imshow(y_true_vis, cmap="viridis", vmin=0.0, vmax=2.0, origin="lower")
    
    # Use Bldg_height (x_input[1]) for contours to show buildings, scaled back 
    bldg_area = x_input[1] > 0
    if np.any(bldg_area):
        bldg_rgba = np.zeros((H, W, 4), dtype=np.float32)
        bldg_rgba[bldg_area] = [1.0, 1.0, 1.0, 1.0]
        ax1.imshow(bldg_rgba, origin="lower", extent=[0, W, 0, H], zorder=10)

    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_visible(False)

    ax2 = plt.subplot(1, 4, 3)
    ax2.set_title("Prediction", pad=36, fontsize=15, fontweight="bold", bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    ax2.imshow(y_pred_vis, cmap="viridis", vmin=0.0, vmax=2.0, origin="lower")
    if np.any(bldg_area):
        bldg_rgba = np.zeros((H, W, 4), dtype=np.float32)
        bldg_rgba[bldg_area] = [1.0, 1.0, 1.0, 1.0]
        ax2.imshow(bldg_rgba, origin="lower", extent=[0, W, 0, H], zorder=10)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_visible(False)

    ax3 = plt.subplot(1, 4, 4)
    ax3.set_title("Diff (Pred - GT)" if has_gt else "Diff (N/A)", pad=36, fontsize=15, fontweight="bold", bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    im3 = ax3.imshow(diff_vis if has_gt else np.zeros_like(diff_vis), cmap="RdBu", vmin=-1.0, vmax=1.0, origin="lower")
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values(): sp.set_visible(False)

    # Metrics
    if has_gt:
        dm = ~outside
        diff_m, abs_m = diff_vis[dm], np.abs(diff_vis[dm])
        mae = np.mean(abs_m)
        rmse = np.sqrt(np.mean(diff_m**2))
        valid_mape = np.abs(y_true[dm]) > 0.1
        mape = np.mean(abs_m[valid_mape] / np.abs(y_true[dm][valid_mape])) * 100.0 if np.any(valid_mape) else 0.0
        
        gt_m = y_true[dm]
        ss_res = np.sum(diff_m ** 2)
        ss_tot = np.sum((gt_m - np.mean(gt_m)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        try:
            from skimage.metrics import structural_similarity
            ssim_val = structural_similarity(y_true, y_pred, data_range=max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min()))
            ssim_str = f"{ssim_val:.3f}"
        except:
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
            return np.corrcoef(pg, tg)[0, 1]

        grad_corr = _grad_corr(y_pred, y_true, dm)
        line1 = f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | MAPE:{mape:.1f}%"
        line2 = f"SSIM:{ssim_str} | GradCorr:{grad_corr:.3f} | R²:{r2:.3f}"
    else:
        line1 = "MAE:N/A | RMSE:N/A | MAPE:N/A"
        line2 = "SSIM:N/A | GradCorr:N/A | R²:N/A"
    
    # --- COLORBARS AND METRICS ---
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    x_center_top = (pos1.x0 + pos2.x1) / 2.0
    
    cb_w = 0.16 
    cb_h = 0.025
    cb_y = pos1.y0 - 0.16 
    
    cax_shared = fig.add_axes([x_center_top - cb_w/2, cb_y, cb_w, cb_h])
    fig.colorbar(im1, cax=cax_shared, orientation="horizontal")
    
    pos3 = ax3.get_position()
    x_center_diff = pos3.x0 + pos3.width / 2.0 + 0.065
    
    cax_diff = fig.add_axes([x_center_diff - cb_w/2, cb_y, cb_w, cb_h])
    fig.colorbar(im3, cax=cax_diff, orientation="horizontal")

    metrics_y = cb_y - 0.12
    fig.text(x_center_diff, metrics_y, f"{line1}\n{line2}", 
             ha="center", va="top", fontsize=9, family="monospace", 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()


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
    # Check common output layer names for different architectures
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

def process_single_csv(csv_path, model, DEVICE, output_dir=None):
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Format exactly matching Conditional Transformer exports
    # Map lowercase 'x', 'y' to uppercase if export varied
    rename_map = {"X": "X_coords", "Y": "Y_coords", "x": "X_coords", "y": "Y_coords", "U_at_z": "U_over_Uref"}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
            
    try:
        t_col = next((c for c in ["mag_U", "actual_U", "mag_U_dimensionless"] if c in df.columns))
        t_flat = df[t_col].to_numpy().copy()
        # Fix noise in Ground Truth (inf or NaN)
        bad_mask = ~np.isfinite(t_flat)
        if np.any(bad_mask):
            print(f"  Warning: Found {np.sum(bad_mask)} non-finite values in Ground Truth. Zeroing them out for metrics.")
            t_flat[bad_mask] = 0.0
    except StopIteration:
        print("Warning: Ground truth target wind 'mag_U' not found in CSV. Visual Error plot will be skipped.")
        t_flat = None
        
    # Standardize data blocks
    gh = {c: df[c].to_numpy() for c in ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_coords", "Y_coords", "dir_sin", "dir_cos"]}
    X_batch, _ = build_input_tensor_from_gh(gh, device="cpu")
    nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df["X_coords"], df["Y_coords"])
    
    # INFERENCE
    with torch.no_grad():
        X_device = X_batch[0].unsqueeze(0).to(DEVICE)
        pred_delta_t = model(X_device)
    
    # De-Normalize
    p_delta_flat = np.array([pred_delta_t[0, 0, iy, ix].cpu().item() for (iy, ix) in idx_map])
    u_ref_flat = df["U_over_Uref"].to_numpy()
    p_mag_flat = np.clip(u_ref_flat * (p_delta_flat + 1.0), 0.0, None)
    
    # 0. PHYSICAL MASKING (Brutalize ghost winds inside buildings)
    is_inside_building = df['Bldg_height'] > (df['Z_relative'] + 0.01) # Add tiny epsilon
    if np.any(is_inside_building):
        n_masked = np.sum(is_inside_building)
        p_mag_flat[is_inside_building] = 0.0
        print(f"  -> Applied building mask to {n_masked} points.")
    
    # 1. SAVE CSV
    df_out = df.copy()
    df_out['mag_U_pred'] = p_mag_flat  # Use exact same column header as CT
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        csv_out = os.path.join(output_dir, f"{base}_pred.csv")
        png_out = os.path.join(output_dir, f"{base}_pred.png")
    else:
        base = os.path.splitext(csv_path)[0]
        csv_out = f"{base}_pred.csv"
        png_out = f"{base}_pred.png"
        
    df_out.to_csv(csv_out, index=False)
    print(f"  -> Saved Grasshopper CSV Map: {csv_out}")

    # 2. SAVE PNG VISUALS 
    p_mag_grid = np.zeros((ny, nx))
    target_mag = np.zeros((ny, nx)) if t_flat is not None else None
    
    for i, (iy, ix) in enumerate(idx_map):
        p_mag_grid[iy, ix] = p_mag_flat[i]
        if target_mag is not None:
            target_mag[iy, ix] = t_flat[i]
            
    # Always save visuals, even if GT is missing
    if target_mag is None:
        target_mag = np.zeros_like(p_mag_grid)
        has_gt = False
    else:
        has_gt = True
        
    save_pred_vs_true(p_mag_grid, target_mag, png_out, X_batch[0].numpy(), has_gt=has_gt)
    print(f"  -> Saved Visual Plot: {png_out}")


def main():
    parser = argparse.ArgumentParser(description="FNO CSV Exporter matching Transformer Format")
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
