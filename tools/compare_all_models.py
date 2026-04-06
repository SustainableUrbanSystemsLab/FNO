import os, sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, tomllib
import argparse
from matplotlib.colors import TwoSlopeNorm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.models.fno2d import FNO2d
from core.models.hybrid import HybridFNO
from core.models.pinn_fno import PINNFNO
from core.models.geo_fno import GeoFNO
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from pipelines.train.distributed import NpyDataset
from tools.diagnose_fno_wakes import analyze_prediction_quality

def load_config():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config.toml'))
    if os.path.exists(path):
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}

def auto_detect_and_build_model(model_class, model_name, state_dict, in_channels, default_modes, default_width, default_layers):
    """Automatically parses architecture size from the state_dict and builds the appropriate model class."""
    modes1, modes2, width, layers = default_modes, default_modes, default_width, default_layers
    
    # Vanilla FNO logic
    if "in_proj.weight" in state_dict:
        width = state_dict["in_proj.weight"].shape[0]
        if "fourier_layers.0.0.weights1" in state_dict:
            w_shape = state_dict["fourier_layers.0.0.weights1"].shape
            modes1 = w_shape[2]
            modes2 = w_shape[3]
            
    # NeuralOp (Hybrid/PINN) logic
    elif "fno.fno_blocks.convs.0.bias" in state_dict:
        width = state_dict["fno.fno_blocks.convs.0.bias"].shape[0]
        if "fno.fno_blocks.convs.0.weight.tensor" in state_dict:
            w_shape = state_dict["fno.fno_blocks.convs.0.weight.tensor"].shape
            modes1 = w_shape[2]
            modes2 = (w_shape[3] - 1) * 2
            
    print(f"[{model_name}] Instantiating size: modes=({modes1},{modes2}), width={width}")
    
    if model_class == FNO2d:
        model = FNO2d(in_channels=in_channels, out_channels=1, modes1=modes1, modes2=modes2, width=width, n_layers=layers)
    else:
        model = model_class(in_channels=in_channels, n_modes=(modes1, modes2), hidden_channels=width, n_layers=layers)
        
    model.load_state_dict(state_dict, strict=True) # or False for PINN variations
    model.eval()
    return model

def load_weights(path, device):
    if not os.path.exists(path): return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    return {k.replace("module.", ""): v for k, v in sd.items() if k != "_metadata"}

def run_comparison(data_path, sample_idx, out_image="model_comparison.png", save_metrics="comparison_metrics.txt"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE}...")

    # Load Data Target
    target_mag, sdf, valid_mask, X_tensor, u_ref = None, None, None, None, None
    config = load_config()
    
    # 1. LOAD DATASET (Supports NPY Dir or CSV directly)
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        df.rename(columns={"X": "X_coords", "Y": "Y_coords", "x": "X_coords", "y": "Y_coords", "U_at_z": "U_over_Uref"}, inplace=True)
        t_col = next((c for c in ["mag_U", "actual_U", "mag_U_dimensionless"] if c in df.columns), None)
        if not t_col: raise RuntimeError("Missing target mag_U in CSV.")
            
        required = ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_coords", "Y_coords", "dir_sin", "dir_cos"]
        gh = {c: df[c].to_numpy() for c in required}
        X_batch, _ = build_input_tensor_from_gh(gh, device="cpu")
        X_tensor = X_batch[0]
        nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df["X_coords"], df["Y_coords"])
        
        target_mag = np.zeros((ny, nx))
        sdf = np.zeros((ny, nx))
        t_flat, u_ref_flat = df[t_col].to_numpy(), df["U_over_Uref"].to_numpy()
        
        for i, (iy, ix) in enumerate(idx_map):
            target_mag[iy, ix] = t_flat[i]
            sdf[iy, ix] = df["SDF"].iloc[i] * 200.0
            
        u_ref_grid = np.zeros((ny, nx))
        for i, (iy, ix) in enumerate(idx_map): u_ref_grid[iy, ix] = u_ref_flat[i]
        u_ref = u_ref_grid
        valid_mask = u_ref_grid > 1e-6

    else:
        dataset = NpyDataset(os.path.join(data_path, "X.npy"), os.path.join(data_path, "Y.npy"), augment=False)
        X_tensor, Y_tensor = dataset[sample_idx]
        u_ref = X_tensor[3].numpy() / 2.0
        target_delta = Y_tensor[0].numpy()
        target_mag = u_ref * (target_delta + 1.0)
        sdf = X_tensor[0].numpy() * 200.0
        valid_mask = u_ref > 1e-6
        
    target_mag[~valid_mask] = 0.0
    
    # 2. RUN INFERENCE FOR ALL 3 MODELS
    defaults = {
        'modes': config.get("model", {}).get("modes1", 32),
        'width': config.get("model", {}).get("width", 64),
        'layers': config.get("model", {}).get("n_layers", 4)
    }
    
    models_to_test = {
        "Standard": ("fno_test_weights.pth", FNO2d),
        "Hybrid": ("hybrid_fno_weights.pth", HybridFNO),
        "Strict PINN": ("pinn_fno_weights.pth", PINNFNO),
        "Geo FNO": ("geo_fno_weights.pth", GeoFNO)
    }
    
    predictions = {}
    metrics_map = {}
    
    X_device = X_tensor.unsqueeze(0).to(DEVICE)

    for name, (weight_file, ModelClass) in models_to_test.items():
        sd = load_weights(weight_file, DEVICE)
        if sd is None:
            print(f"Skipping {name} (Could not find {weight_file})")
            continue
            
        model = auto_detect_and_build_model(
            ModelClass, name, sd, X_tensor.shape[0], 
            defaults['modes'], defaults['width'], defaults['layers']
        ).to(DEVICE)
        
        with torch.no_grad():
            pred_t = model(X_device)
            
        p_delta = pred_t[0, 0].cpu().numpy()
        if data_path.endswith(".csv"):
            p_mag = np.zeros_like(target_mag)
            p_delta_flat = np.array([pred_t[0, 0, iy, ix].cpu().item() for (iy, ix) in idx_map])
            p_mag_flat = np.clip(u_ref_flat * (p_delta_flat + 1.0), 0.0, None)
            for i, (iy, ix) in enumerate(idx_map): p_mag[iy, ix] = p_mag_flat[i]
        else:
            p_mag = np.clip(u_ref * (p_delta + 1.0), 0.0, None)
            
        p_mag[~valid_mask] = 0.0
        predictions[name] = p_mag
        
        # Implement Pix2PixHD strict circular domain mask constraint
        H_grid, W_grid = target_mag.shape
        cy_m, cx_m = H_grid // 2, W_grid // 2
        rad_m = min(H_grid, W_grid) // 2 - 5
        Yc_m, Xc_m = np.ogrid[:H_grid, :W_grid]
        outside = np.sqrt((Xc_m - cx_m)**2 + (Yc_m - cy_m)**2) >= rad_m
        
        # Override valid_mask with rigorous circular inner-domain
        has_data = valid_mask & (~outside)
        
        if has_data.sum() > 0:
            ss_res = np.sum((target_mag[has_data] - p_mag[has_data]) ** 2)
            ss_tot = np.sum((target_mag[has_data] - np.mean(target_mag[has_data])) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            mape_mask = (has_data) & (target_mag > 0.1)
            if mape_mask.sum() > 0:
                mape_score = np.mean(np.abs((target_mag[mape_mask] - p_mag[mape_mask]) / target_mag[mape_mask])) * 100.0
            else:
                mape_score = 0.0
            
            # Match exact Grad Corr from Conditional Transformer (Directional Flat)
            def _grad_corr(pred, true, mask):
                pdx = np.diff(pred, axis=1, prepend=pred[:, :1])
                pdy = np.diff(pred, axis=0, prepend=pred[:1, :])
                tdx = np.diff(true, axis=1, prepend=true[:, :1])
                tdy = np.diff(true, axis=0, prepend=true[:1, :])
                pg = np.concatenate([pdx.flatten(), pdy.flatten()])
                tg = np.concatenate([tdx.flatten(), tdy.flatten()])
                mm = np.concatenate([mask.flatten(), mask.flatten()])
                pg, tg = pg[mm], tg[mm]
                if np.std(pg) < 1e-8 or np.std(tg) < 1e-8:
                    return 0.0
                return np.corrcoef(pg, tg)[0, 1]

            grad_corr = _grad_corr(p_mag, target_mag, has_data)
            
            try:
                from skimage.metrics import structural_similarity
                rng = target_mag[has_data].max() - target_mag[has_data].min()
                ssim_score = structural_similarity(target_mag, p_mag, data_range=rng)
            except ImportError:
                ssim_score = -1.0
        else:
            r2_score = 0.0
            mape_score = 0.0
            grad_corr = 0.0
            ssim_score = 0.0
            
        mets = analyze_prediction_quality(p_mag, target_mag)
        mets['r2'] = r2_score
        mets['mape'] = mape_score
        mets['grad_corr'] = grad_corr
        mets['ssim'] = ssim_score
        metrics_map[name] = mets

    if not predictions:
        print("No models found! Train them first.")
        return

    # 3. PRINT ANALYSIS TEXT
    print("\n" + "="*50)
    print("MODEL COMPARISON METRICS (Wind Speed MAE / RMSE)")
    print("="*50)
    for name, mets in metrics_map.items():
        print(f"--- {name.upper()} ---")
        print(f"R-squared (R^2):  {mets['r2']:.4f}")
        print(f"SSIM:             {mets['ssim']:.4f}")
        print(f"Gradient Corr:    {mets['grad_corr']:.4f}")
        print(f"Overall MAE:      {mets['overall_mae']:.4f}")
        print(f"Overall RMSE:     {mets['overall_rmse']:.4f}")
        print(f"overall MAPE:     {mets['mape']:.2f}%")
        print(f"Wake Region MAE:  {mets['wake_mae']:.4f}")
        print(f"Peak Region MAE:  {mets['peak_mae']:.4f}")
        print(f"Gradient Error:   {mets['gradient_mae']:.4f} \n")

    # Write textual summary to file
    with open(save_metrics, "w") as f:
        f.write("Model Comparison Metrics\n========================\n")
        f.write("Target Data: " + data_path + f" (Sample {sample_idx})\n\n")
        for name, mets in metrics_map.items():
            f.write(f"--- {name} ---\n")
            f.write(f"R-squared:   {mets['r2']:.4f}\n")
            f.write(f"SSIM:        {mets['ssim']:.4f}\n")
            f.write(f"Grad Corr:   {mets['grad_corr']:.4f}\n")
            f.write(f"Overall MAE: {mets['overall_mae']:.4f}\n")
            f.write(f"overall RMSE:{mets['overall_rmse']:.4f}\n")
            f.write(f"overall MAPE:{mets['mape']:.2f}%\n")
            f.write(f"Wake MAE:    {mets['wake_mae']:.4f}\n")
            f.write(f"Peak MAE:    {mets['peak_mae']:.4f}\n")
            f.write(f"Grad Error:  {mets['gradient_mae']:.4f}\n\n")

    # 4. PLOTTING VISUAL COMPARISON
    num_plots = len(predictions) + 1
    fig, axes = plt.subplots(3, num_plots, figsize=(4.5 * num_plots, 10))
    fig.subplots_adjust(wspace=0.15, hspace=0.25)
    
    # Colormaps
    vmax = np.max(target_mag)
    cmap_wind = "viridis"
    norm_diff = TwoSlopeNorm(vcenter=0.0, vmin=-0.5, vmax=0.5)

    def plot_field(ax, field, title, is_diff=False):
        ax.imshow(sdf, origin='lower', cmap='binary', alpha=0.9)
        if is_diff:
            im = ax.imshow(field, origin='lower', cmap='RdBu_r', norm=norm_diff)
        else:
            im = ax.imshow(field, origin='lower', cmap=cmap_wind, vmin=0, vmax=vmax, alpha=0.85)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot Ground Truth
    plot_field(axes[0, 0], target_mag, "Ground Truth (CFD)")
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    # Plot Model Results
    idx = 1
    for name, p_mag in predictions.items():
        diff = p_mag - target_mag
        metrics = metrics_map[name]
        
        plot_field(axes[0, idx], p_mag, f"{name} Prediction")
        plot_field(axes[1, idx], diff, f"Error (Pred - Truth)", is_diff=True)
        
        # Overlay textual metrics instead of a 3rd plot
        axes[2, idx].axis('off')
        textstr = '\n'.join((
            "METRICS:",
            f"R^2 Score: {metrics['r2']:.3f}",
            f"SSIM: {metrics['ssim']:.3f}",
            f"Grad Corr: {metrics['grad_corr']:.3f}",
            f"Sys MAE: {metrics['overall_mae']:.3f}",
            f"Sys RMSE: {metrics['overall_rmse']:.3f}",
            f"Sys MAPE: {metrics['mape']:.1f}%",
            f"Wake MAE: {metrics['wake_mae']:.3f}",
            f"Peak MAE: {metrics['peak_mae']:.3f}",
            f"Grad MAE: {metrics['gradient_mae']:.3f}"
        ))
        axes[2, idx].text(0.1, 0.5, textstr, transform=axes[2, idx].transAxes, 
                          fontsize=12, verticalalignment='center')
        
        idx += 1

    plt.suptitle(f"Model Architecture Comparison (M={defaults['modes']}, W={defaults['width']})", fontsize=16, y=0.98)
    plt.savefig(out_image, bbox_inches='tight', dpi=150)
    print(f"Comparison image successfully saved to: {out_image}")
    print(f"Metrics saved to: {save_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to NPY directory or specific CSV file")
    parser.add_argument("--sample", type=int, default=15, help="Index of sample if using NPY directory")
    parser.add_argument("--output", type=str, default="model_comparison.png")
    args = parser.parse_args()
    
    run_comparison(args.data, args.sample, args.output)
