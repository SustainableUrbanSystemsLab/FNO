"""
Evaluate Model on the 90/10 Validation Subset — Multi-Channel (U, k, U_roof, k_roof)
======================================================================================
Reproduces the exact same 90/10 train/val split used during training (fixed seed 42)
and runs the full evaluation suite on the hold-out 10% subset.

For models with >1 output channel, metrics are computed per-channel with the correct
masking strategy:
  - ch0,1 (U, k):          mask building interiors
  - ch2,3 (U_roof, k_roof): mask open ground
"""
import os, sys, torch, numpy as np, pandas as pd, argparse, json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.utils.config_loader import load_config
from pipelines.train.distributed import NpyDataset, CHANNEL_NAMES, CHANNEL_VMAXES
from tools.infer_csv import build_model, save_pred_vs_true


def get_metrics_for_2d(y_pred, y_true, mask):
    """Compute regression + spatial metrics for a single (H, W) pair."""
    if mask is None or np.sum(mask) == 0:
        return {'MAE': 0, 'RMSE': 0, 'MAPE': 0, 'R2': 0, 'SSIM': 0, 'GradCorr': 0}

    diff = y_pred[mask] - y_true[mask]
    abs_d = np.abs(diff)

    mae = float(np.mean(abs_d))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    gt = y_true[mask]
    valid_mape = np.abs(gt) > 0.1
    mape = float(np.mean(abs_d[valid_mape] / np.abs(gt[valid_mape])) * 100.0) if np.any(valid_mape) else 0.0

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # SSIM
    try:
        from skimage.metrics import structural_similarity as ssim_func
        pr_in = y_pred[mask]
        dr = max(gt.max(), pr_in.max()) - min(gt.min(), pr_in.min())
        if dr < 1e-8:
            ssim_val = 1.0
        else:
            _, smap = ssim_func(y_true, y_pred, data_range=dr, full=True)
            ssim_val = float(smap[mask].mean())
    except Exception:
        ssim_val = 0.0

    # GradCorr
    def _gc(p, t, m):
        pdx = np.diff(p, axis=1, prepend=p[:, :1])
        pdy = np.diff(p, axis=0, prepend=p[:1, :])
        tdx = np.diff(t, axis=1, prepend=t[:, :1])
        tdy = np.diff(t, axis=0, prepend=t[:1, :])
        pg = np.concatenate([pdx.flatten(), pdy.flatten()])
        tg = np.concatenate([tdx.flatten(), tdy.flatten()])
        mm = np.concatenate([m.flatten(), m.flatten()])
        pg, tg = pg[mm], tg[mm]
        if np.std(pg) < 1e-8 or np.std(tg) < 1e-8: return 0.0
        return float(np.corrcoef(pg, tg)[0, 1])

    grad_corr = _gc(y_pred, y_true, mask)

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'SSIM': ssim_val, 'GradCorr': grad_corr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth weights")
    parser.add_argument("--model_type", type=str, required=True, choices=["standard", "hybrid", "pinn", "geo"])
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--out", type=str, default="val_evaluation_report")
    parser.add_argument("--n_vis", type=int, default=5, help="Number of per-channel validation images to save")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Detect Data Folder (same logic as training)
    if sys.platform == 'win32':
        DATA_FOLDER = config.get('paths', {}).get('data_folder_windows', 'train_csv')
    else:
        DATA_FOLDER = config.get('paths', {}).get('data_folder_ice', config.get('paths', {}).get('data_folder_linux', 'train_csv'))

    x_path = os.path.join(DATA_FOLDER, 'X.npy')
    y_path = os.path.join(DATA_FOLDER, 'Y.npy')

    # 2. Re-create the 90/10 Split
    print(f"Loading dataset from {DATA_FOLDER}...")
    full_dataset = NpyDataset(x_path, y_path, augment=False)
    VAL_SPLIT = config.get('training', {}).get('val_split', 0.1)
    total_samples = len(full_dataset)
    train_size = int((1.0 - VAL_SPLIT) * total_samples)

    # Seed 42 is critical to match the training hold-out set exactly
    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
    val_idx = indices[train_size:]
    val_subset = Subset(full_dataset, val_idx)

    # Detect output channels
    sample_x, sample_y = val_subset[0]
    out_ch = sample_y.shape[0]
    channel_names = CHANNEL_NAMES[:out_ch]

    print(f"Total samples: {total_samples}")
    print(f"Validation hold-out: {len(val_subset)} samples, {out_ch} output channel(s)")
    print(f"Channels: {', '.join(channel_names)}")

    # 3. Load Model
    print(f"Building {args.model_type} model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model = build_model(args.model_type, state_dict, device)
    model.eval()

    # 4. Run Evaluation — per-channel
    # Accumulate per-channel metrics
    per_channel_results = {cn: [] for cn in channel_names}
    vis_dir = os.path.join(os.path.dirname(args.out) or ".", "val_images")
    os.makedirs(vis_dir, exist_ok=True)
    n_vis_left = args.n_vis

    for i in tqdm(range(len(val_subset)), desc="Evaluating Val Subset"):
        xb, yb = val_subset[i]
        xb_dev = xb.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(xb_dev)  # (1, out_ch, H, W)

        x_np = xb.numpy()  # (8, H, W)
        H, W = x_np.shape[1], x_np.shape[2]

        # Circular domain mask
        cy_m, cx_m = H // 2, W // 2
        Yc, Xc = np.ogrid[:H, :W]
        outside = np.sqrt((Xc - cx_m)**2 + (Yc - cy_m)**2) >= (min(H, W) // 2 - 5)

        # Building mask from Bldg_height (ch1 in FNO format)
        bldg_mask = x_np[1] > 0  # (H, W)
        open_gnd_mask = ~bldg_mask
        ch_extra_masks = [bldg_mask, bldg_mask, open_gnd_mask, open_gnd_mask]

        for ch_idx, cn in enumerate(channel_names):
            if ch_idx >= out_ch:
                break

            y_pred_ch = pred[0, ch_idx].cpu().numpy()
            y_true_ch = yb[ch_idx].numpy()

            # Physical masking: hide = outside + channel-specific mask
            extra = ch_extra_masks[ch_idx] if ch_idx < len(ch_extra_masks) else None
            hide = outside if extra is None else (outside | extra)
            domain_mask = ~hide

            metrics = get_metrics_for_2d(y_pred_ch, y_true_ch, domain_mask)
            metrics['sample_idx'] = val_idx[i]
            per_channel_results[cn].append(metrics)

            # Save a few sample visualizations
            if n_vis_left > 0 and ch_idx == 0:  # only decrement on first channel
                pass  # handled below after all channels
            if i < args.n_vis:
                png_path = os.path.join(vis_dir, f"val_{i:04d}_{cn}.png")
                save_pred_vs_true(
                    np.maximum(y_pred_ch, 0), y_true_ch, png_path,
                    x_input=x_np, has_gt=True,
                    vmax=CHANNEL_VMAXES[ch_idx] if ch_idx < len(CHANNEL_VMAXES) else 2.0,
                    channel_label=cn,
                    extra_mask=extra,
                )

    # 5. Aggregate & Print Report
    print("\n" + "=" * 80)
    print("      VALIDATION SUBSET (TEST) RESULTS — PER CHANNEL      ")
    print("=" * 80)
    print(f"{'Channel':<10} {'MAE':>10} {'RMSE':>10} {'MAPE%':>8} {'R2':>8} {'SSIM':>8} {'GradCorr':>10}")
    print("-" * 80)

    all_summaries = {}
    for cn in channel_names:
        df_ch = pd.DataFrame(per_channel_results[cn])
        summary = df_ch.drop(columns=['sample_idx'], errors='ignore').mean().to_dict()
        all_summaries[cn] = summary
        print(f"{cn:<10} {summary['MAE']:>10.4f} {summary['RMSE']:>10.4f} {summary['MAPE']:>8.2f} "
              f"{summary['R2']:>8.4f} {summary['SSIM']:>8.4f} {summary['GradCorr']:>10.4f}")

    print("=" * 80)

    # 6. Save Report
    json_out = f"{args.out}.json"
    csv_out = f"{args.out}.csv"

    with open(json_out, "w") as f:
        json.dump({
            "model": args.model_path,
            "type": args.model_type,
            "data_folder": DATA_FOLDER,
            "val_samples": len(val_subset),
            "out_channels": out_ch,
            "per_channel_metrics": all_summaries,
        }, f, indent=4)

    # Save flat CSV with all channels
    rows = []
    for cn in channel_names:
        for m in per_channel_results[cn]:
            row = {'channel': cn}
            row.update(m)
            rows.append(row)
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"Report saved to {json_out} and {csv_out}")
    if args.n_vis > 0:
        print(f"Sample images saved to {vis_dir}")

if __name__ == "__main__":
    main()
