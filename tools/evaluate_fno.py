import os, sys, torch, numpy as np, matplotlib.pyplot as plt, tomllib
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.models.fno2d import FNO2d
from pipelines.train.distributed import NpyDataset
from tools.diagnose_fno_wakes import analyze_prediction_quality, create_diagnostic_plot

def load_config(config_path='config.toml'):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', config_path))
    if os.path.exists(path):
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}

def run_diagnostics_on_npy(model_path, x_path, y_path, sample_idx=0, output_name="fno_diagnostic.png"):
    print(f"--- FNO2d NPY Diagnostic: {os.path.basename(model_path)} ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data
    print(f"Loading dataset from: \n  X: {x_path}\n  Y: {y_path}")
    dataset = NpyDataset(x_path, y_path, augment=False)
    
    if sample_idx >= len(dataset):
        print(f"Error: sample_idx {sample_idx} is out of bounds (max {len(dataset)-1}).")
        return
        
    print(f"Extracting sample {sample_idx}...")
    # Get the remapped data directly from the dataset logic
    X_tensor, Y_tensor = dataset[sample_idx]
    
    # 2. Load Checkpoint
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Handle DDP and legacy formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if k != "_metadata"}

    config = load_config()
    MODES1 = config.get("model", {}).get("modes1", 32)
    MODES2 = config.get("model", {}).get("modes2", 32)
    WIDTH = config.get("model", {}).get("width", 64)
    N_LAYERS = config.get("model", {}).get("n_layers", 4)
    
    # Auto-detect modes/width if possible to avoid crashes
    if "in_proj.weight" in state_dict:
        new_width = state_dict["in_proj.weight"].shape[0]
        if new_width != WIDTH:
            WIDTH = new_width
            
    if "fourier_layers.0.0.weights1" in state_dict:
        w_shape = state_dict["fourier_layers.0.0.weights1"].shape
        new_modes1 = w_shape[2]
        new_modes2 = w_shape[3]
        if new_modes1 != MODES1 or new_modes2 != MODES2:
            MODES1, MODES2 = new_modes1, new_modes2
            
    count = 0
    while f"fourier_layers.{count}.0.weights1" in state_dict:
        count += 1
    if count > 0 and count != N_LAYERS:
        N_LAYERS = count

    # Create FNO2d model
    model = FNO2d(
        in_channels=X_tensor.shape[0],
        out_channels=1,
        modes1=MODES1,
        modes2=MODES2,
        width=WIDTH,
        n_layers=N_LAYERS
    ).to(DEVICE)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Inference
    X_batch = X_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_tensor = model(X_batch)

    # 4. Scale back to Absolute Magnitude (mag_U)
    u_ref = X_tensor[3].numpy() / 2.0
    pred_delta = pred_tensor[0, 0].cpu().numpy()
    target_delta = Y_tensor[0].numpy()
    
    pred_mag = np.clip(u_ref * (pred_delta + 1.0), 0.0, None)
    target_mag = u_ref * (target_delta + 1.0)
    sdf = X_tensor[0].numpy() * 200.0 # Re-scale SDF for contour plotting
    
    valid_mask = u_ref > 1e-6
    pred_mag[~valid_mask] = 0
    target_mag[~valid_mask] = 0

    # 5. Analyze & Plot
    print("\nCalculating metrics...")
    metrics = analyze_prediction_quality(pred_mag, target_mag)

    print("\nFNO2d METRICS (Real magnitudes):")
    print(f"  Overall MAE:       {metrics['overall_mae']:.4f}")
    print(f"  Wake MAE:          {metrics['wake_mae']:.4f}")

    create_diagnostic_plot(pred_mag, target_mag, sdf=sdf, save_path=output_name)
    print(f"\nDiagnostic plot saved to: {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fno_test_weights.pth")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing X.npy and Y.npy")
    parser.add_argument("--sample", type=int, default=15, help="Index of the dataset sample to visualize")
    parser.add_argument("--output", type=str, default="fno_diagnostic.png")
    args = parser.parse_args()

    x_path = os.path.join(args.data_dir, "X.npy")
    y_path = os.path.join(args.data_dir, "Y.npy")
    
    run_diagnostics_on_npy(args.model, x_path, y_path, args.sample, args.output)
