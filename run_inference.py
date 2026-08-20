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

    # 2. Process Data dynamically
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.isdir(args.csv):
        files = glob.glob(os.path.join(args.csv, "*.csv"))
        files = [f for f in files if "_pred" not in os.path.basename(f)]
        for f in files: 
            process_single_csv(f, model, device, args.output_dir)
    else:
        process_single_csv(args.csv, model, device, args.output_dir)
        
    print(f"\nDone! Results successfully populated in -> {args.output_dir}")

if __name__ == "__main__":
    main()
