#!/usr/bin/env python3
"""
Universal FNO Inference CLI
============================
Reads a Grasshopper CSV file, runs it through the requested FNO Scenario 
(Standard, Hybrid, PINN, Geo), and exports the exact identical Point-Cloud CSV 
and a High-Fidelity 4-Panel Visualization.

Usage:
  python run_inference.py --scenario GEO --csv test_csv/ML_FormFlux_1_135.csv
"""
import os
import sys
import argparse
import torch
import glob

# Ensure the tools directory is recognized so we can reuse the 4-panel inference logic
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tools.infer_csv import process_single_csv, load_weights, build_model

SCENARIOS = {
    "STANDARD": {
        "name": "Standard FNO",
        "checkpoint": "fno_test_weights.pth",
        "type": "standard",
        "desc": "Baseline High-Resolution Fourier Operator"
    },
    "HYBRID": {
        "name": "Hybrid U-FNO",
        "checkpoint": "hybrid_fno_weights.pth",
        "type": "hybrid",
        "desc": "Fourier Operator wrapped in L1 Boundary Convolutional Skips"
    },
    "PINN": {
        "name": "Strict PINN-FNO",
        "checkpoint": "pinn_fno_weights.pth",
        "type": "pinn",
        "desc": "FNO bounded by explicit Navier-Stokes Mass & Momentum tracking"
    },
    "GEO": {
        "name": "Geometric Latent FNO",
        "checkpoint": "geo_fno_weights.pth",
        "type": "geo",
        "desc": "Boundary-warped Fourier operator evaluating in latent space"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Universal FNO Inference CLI")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()), required=True, help="Scenario to run")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file or directory of CSVs")
    parser.add_argument("--output_dir", type=str, default="results/inference", help="Target directory for results")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override default checkpoint path for this scenario")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    conf = SCENARIOS[args.scenario]
    ckpt_path = args.checkpoint if args.checkpoint else conf["checkpoint"]
    if not os.path.exists(ckpt_path):
        # Smart fallback detection for standard model naming variations
        fallbacks = []
        if args.scenario == "STANDARD":
            fallbacks = ["fno_mag_weights.pth", "fno_weights.pth", "fno_test_weights.pth", "fno_model.pth"]
        elif args.scenario == "HYBRID":
            fallbacks = ["hybrid_weights.pth", "hybrid_fno_weights.pth", "fno_hybrid_weights.pth"]
        elif args.scenario == "PINN":
            fallbacks = ["pinn_weights.pth", "pinn_fno_weights.pth", "fno_pinn_weights.pth"]
        elif args.scenario == "GEO":
            fallbacks = ["geo_fno_weights.pth", "geo_weights.pth", "fno_geo_weights.pth", "fno_weights.pth"]

        # Check explicit fallbacks first
        for fb in fallbacks:
            if os.path.exists(fb):
                print(f"[Info] Default checkpoint '{ckpt_path}' not found. Auto-detected fallback: '{fb}'")
                ckpt_path = fb
                break

        # Check inside epochs/ folder if still not found
        if not os.path.exists(ckpt_path) and os.path.exists("epochs"):
            ep_prefix = args.scenario.lower()
            matching_eps = sorted(glob.glob(f"epochs/{ep_prefix}_*.pth") + glob.glob("epochs/checkpoint_*.pth"))
            if matching_eps:
                ckpt_path = matching_eps[-1]  # Pick latest epoch checkpoint
                print(f"[Info] Auto-detected latest epoch checkpoint: '{ckpt_path}'")

    if not os.path.exists(ckpt_path):
        print(f"Error: Could not find checkpoint file '{ckpt_path}'. Specify your model with --checkpoint <path.pth>")
        sys.exit(1)

    device = torch.device(args.device)

    print(f"\n>>> Running FNO Inference: {conf['name']}")
    print(f">>> Scenario Info: {conf['desc']}")
    print(f">>> Target Checkpoint: {ckpt_path}")
    print(f">>> Hardware Device: {device}\n")

    # 1. Load Model Framework
    state_dict = load_weights(ckpt_path, device)
    model = build_model(conf["type"], state_dict, device)

    # 2. Standardized Timestamped Results Folder
    from datetime import datetime
    import json

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name_clean = os.path.splitext(os.path.basename(ckpt_path))[0]
    folder_name = f"{args.scenario}_{ckpt_name_clean}_{timestamp_str}"
    
    # If user provided default results/inference or a specific root folder, nest timestamped subfolder
    if args.output_dir == "results/inference":
        target_out_dir = os.path.join("results/inference", folder_name)
    else:
        target_out_dir = os.path.join(args.output_dir, folder_name)

    os.makedirs(target_out_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f" STANDARDIZED INFERENCE RUN INITIALIZED")
    print(f"==================================================")
    print(f"  Scenario:      {conf['name']} ({args.scenario})")
    print(f"  Checkpoint:    {os.path.abspath(ckpt_path)}")
    ckpt_stat = os.stat(ckpt_path)
    ckpt_size_mb = ckpt_stat.st_size / (1024 * 1024)
    ckpt_mtime = datetime.fromtimestamp(ckpt_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Checkpoint Sz: {ckpt_size_mb:.2f} MB (Modified: {ckpt_mtime})")
    print(f"  Input Data:    {os.path.abspath(args.csv)}")
    print(f"  Output Dir:    {os.path.abspath(target_out_dir)}")
    print(f"  Device:        {device}")
    print(f"==================================================\n")

    # 3. Execute Inference
    if os.path.isdir(args.csv):
        files = glob.glob(os.path.join(args.csv, "*.csv"))
        files = [f for f in files if "_pred" not in os.path.basename(f)]
        for f in files: 
            process_single_csv(f, model, device, target_out_dir)
    else:
        process_single_csv(args.csv, model, device, target_out_dir)

    # 4. Generate Metadata Summary File inside output directory
    generated_files = sorted(os.listdir(target_out_dir))
    
    summary_txt_path = os.path.join(target_out_dir, "inference_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write("======================================================================\n")
        f.write(" FNO INFERENCE RUN SUMMARY & METADATA LOG\n")
        f.write("======================================================================\n")
        f.write(f"Run Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scenario Name:        {conf['name']} ({args.scenario})\n")
        f.write(f"Scenario Description: {conf['desc']}\n")
        f.write(f"Checkpoint File:      {os.path.abspath(ckpt_path)}\n")
        f.write(f"Checkpoint Size:      {ckpt_size_mb:.2f} MB\n")
        f.write(f"Checkpoint Modified:  {ckpt_mtime}\n")
        f.write(f"Input Data Source:    {os.path.abspath(args.csv)}\n")
        f.write(f"Hardware Device:      {device}\n")
        f.write(f"Output Directory:     {os.path.abspath(target_out_dir)}\n")
        f.write("======================================================================\n")
        f.write("Generated Output Files:\n")
        for gf in generated_files:
            f.write(f"  - {gf}\n")
        f.write("======================================================================\n")

    metadata_json_path = os.path.join(target_out_dir, "metadata.json")
    metadata_dict = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "scenario": args.scenario,
        "scenario_name": conf['name'],
        "checkpoint_file": os.path.abspath(ckpt_path),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "checkpoint_modified": ckpt_mtime,
        "input_csv": os.path.abspath(args.csv),
        "device": str(device),
        "output_directory": os.path.abspath(target_out_dir),
        "generated_files": generated_files
    }
    with open(metadata_json_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    print(f"\n[SUCCESS] Inference run complete!")
    print(f"  -> Results folder: {os.path.abspath(target_out_dir)}")
    print(f"  -> Summary file:   {os.path.abspath(summary_txt_path)}")
    print(f"  -> Metadata JSON:  {os.path.abspath(metadata_json_path)}")

if __name__ == "__main__":
    main()
