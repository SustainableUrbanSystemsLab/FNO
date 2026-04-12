"""
Evaluate Model on the 90/10 Validation Subset
=============================================
This script reproduces the exact same 90/10 train/val split used during training
(fixed seed 42) and runs the full evaluation suite on the hold-out 10% subset.
"""
import os, sys, torch, numpy as np, pandas as pd, argparse, json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.utils.config_loader import load_config
from pipelines.train.distributed import NpyDataset
from tools.infer_csv import build_model
from evaluate import get_metrics_for_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth weights")
    parser.add_argument("--model_type", type=str, required=True, choices=["standard", "hybrid", "pinn", "geo"])
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--out", type=str, default="val_evaluation_report")
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
    
    print(f"Total samples: {total_samples}")
    print(f"Evaluation will be performed on the 10% ({len(val_subset)} samples) validation hold-out.")

    # 3. Load Model
    print(f"Building {args.model_type} model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    # We need to infer architecture from config or saved payload if possible, 
    # but build_model uses config.toml by default.
    model = build_model(args.model_type, state_dict, device)
    model.eval()

    # 4. Run Evaluation
    results = []
    for i in tqdm(range(len(val_subset)), desc="Evaluating Val Subset"):
        xb, yb = val_subset[i]
        xb = xb.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(xb)
            
        y_pred = pred[0, 0].cpu().numpy()
        y_true = yb[0].cpu().numpy()
        
        # Binary mask for the actual fluid domain (SDF > 0)
        sdf = xb[0, 0].cpu().numpy()
        mask = sdf > 0
        
        metrics = get_metrics_for_sample(y_pred, y_true, mask, device=device.type)
        metrics['sample_idx'] = val_idx[i]
        results.append(metrics)

    # 5. Save Report
    df = pd.DataFrame(results)
    summary = df.drop(columns=['sample_idx']).mean().to_dict()
    
    print("\n" + "="*50)
    print("      VALIDATION SUBSET (TEST) RESULTS      ")
    print("="*50)
    for k, v in summary.items():
        print(f"{k.rjust(15)} : {v:.4f}")
    print("="*50)
    
    json_out = f"{args.out}.json"
    csv_out = f"{args.out}.csv"
    
    with open(json_out, "w") as f:
        json.dump({
            "model": args.model_path,
            "type": args.model_type,
            "data_folder": DATA_FOLDER,
            "val_samples": len(val_subset),
            "metrics": summary
        }, f, indent=4)
        
    df.to_csv(csv_out, index=False)
    print(f"Report saved to {json_out} and {csv_out}")

if __name__ == "__main__":
    main()
