"""
PINN-FNO Inference Pipeline
============================
Runs inference using the trained PINN-FNO model on test CSV files.

Usage:
    uv run python pipelines/inference/pinn.py
    # or run a file directly:
    uv run python pipelines/inference/pinn.py --csv test_csv/my_case.csv
"""

import os, glob, sys, torch, numpy as np, pandas as pd, tomllib, argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.models.pinn_fno import PINNFNO
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

# ============ Config ============
CONFIG_FILE = 'config.toml'

def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', CONFIG_FILE))
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    return {}

config   = load_config()
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'pinn_fno_weights.pth'


def run_inference(csv_path, model, out_suffix='_pinn_pred'):
    df = pd.read_csv(csv_path)
    df.rename(columns={'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords',
                       'y': 'Y_coords', 'U_at_z': 'U_over_Uref'}, inplace=True)

    required = ['SDF', 'Bldg_height', 'Z_relative', 'U_over_Uref',
                'X_coords', 'Y_coords', 'dir_sin', 'dir_cos']
    if any(c not in df.columns for c in required):
        print(f"  SKIP {csv_path} — missing columns"); return

    gh = {c: df[c].to_numpy() for c in required}
    X, _ = build_input_tensor_from_gh(gh, device=DEVICE)

    with torch.no_grad():
        delta_pred = model(X)[0, 0].cpu().numpy()

    _, _, _, _, idx_map = infer_grid_from_coords_simple(df['X_coords'], df['Y_coords'])
    delta_flat   = np.array([delta_pred[iy, ix] for iy, ix in idx_map])
    baseline     = df['U_over_Uref'].to_numpy()
    mag_final    = np.clip(baseline * (delta_flat + 1.0), 0.0, None)

    df['delta_pred'] = np.round(delta_flat, 6)
    df['mag_U']      = np.round(mag_final, 6)

    df.rename(columns={'X_coords': 'x', 'Y_coords': 'y'}, inplace=True)
    out_path = csv_path.replace('.csv', f'{out_suffix}.csv')
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None,
                        help='Single CSV file. If not set, runs all files in test_folder.')
    args = parser.parse_args()

    # Load model
    MODES1   = config.get('model', {}).get('modes1', 32)
    MODES2   = config.get('model', {}).get('modes2', 32)
    WIDTH    = config.get('model', {}).get('width', 64)
    N_LAYERS = config.get('model', {}).get('n_layers', 4)

    # Determine input channels from a sample file
    test_folder = config.get('paths', {}).get('test_folder', 'test_csv')
    sample_files = sorted(glob.glob(os.path.join(test_folder, '*.csv')))
    if not sample_files:
        raise RuntimeError(f"No CSV files found in {test_folder}")

    sample_df = pd.read_csv(sample_files[0])
    sample_df.rename(columns={'X': 'X_coords', 'Y': 'Y_coords'}, inplace=True)
    required = ['SDF', 'Bldg_height', 'Z_relative', 'U_over_Uref',
                'X_coords', 'Y_coords', 'dir_sin', 'dir_cos']
    gh_sample = {c: sample_df[c].to_numpy() for c in required}
    X_sample, _ = build_input_tensor_from_gh(gh_sample, device='cpu')
    in_channels = X_sample.shape[1]

    model = PINNFNO(
        in_channels=in_channels,
        n_modes=(MODES1, MODES2),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model weights not found at {MODEL_PATH}. Train first!")

    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    sd = {k.replace('module.', ''): v for k, v in sd.items() if k != '_metadata'}
    model.load_state_dict(sd, strict=False)
    model.eval()

    print(f"PINN-FNO loaded from {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    if args.csv:
        files = [args.csv]
    else:
        files = [f for f in sample_files if not f.endswith('_pinn_pred.csv')]

    print(f"Running inference on {len(files)} files...")
    for csv in tqdm(files):
        try:
            out = run_inference(csv, model)
            if out: print(f"  Saved: {out}")
        except Exception as e:
            print(f"  FAILED {csv}: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
