# ICE Cluster Command Cheat Sheet

Make sure to run `git pull` before running any tests.

### 🗺️ 1. Compute Node Access (Interactive)
To start a session with a GPU for testing:
```bash
salloc -N 1 -c 4 --mem=16G --partition=coe-gpu --gres=gpu:1 -t 2:00:00
```

---

### 🚀 2. Training Models (Distributed)
Launch multi-GPU training jobs using the SLURM wrapper.

**Standard FNO:**
```bash
bash slurm/deploy_ice.sh --config config.toml --script pipelines/train/distributed.py --gpu h100 --ngpus 2 --fresh
```

**Hybrid U-FNO (Boundary-Aware):**
```bash
bash slurm/deploy_ice.sh --config config.toml --script pipelines/train/hybrid.py --gpu h100 --ngpus 2 --fresh
```

**PINN-FNO (Physics-Informed):**
```bash
bash slurm/deploy_ice.sh --config config.toml --script pipelines/train/pinn.py --gpu h100 --ngpus 2 --fresh
```

**Geo-FNO (Geometry-Aware):**
```bash
bash slurm/deploy_ice.sh --config config.toml --script pipelines/train/geo.py --gpu h100 --ngpus 2 --fresh
```

---

### 🏁 3. Universal Inference (New CLI)
Use these for high-fidelity 4-panel diagnostic plots and Grasshopper CSVs.

**Standard:**
```bash
uv run python run_inference.py --scenario STANDARD --csv test_csv/ML_FormFlux_1_135.csv
```

**Hybrid:**
```bash
uv run python run_inference.py --scenario HYBRID --csv test_csv/ML_FormFlux_1_135.csv
```

**PINN:**
```bash
uv run python run_inference.py --scenario PINN --csv test_csv/ML_FormFlux_1_135.csv
```

**Geo:**
```bash
uv run python run_inference.py --scenario GEO --csv test_csv/ML_FormFlux_1_135.csv
```

**Batch Inference (All Files in Folder):**
```bash
uv run python run_inference.py --scenario GEO --csv test_csv/ --output_dir results/batch_eval/
```

---

### 📊 4. Master Evaluation & Performance
Compare architectures and visualize convergence.

**Master Side-By-Side Comparison (6-Column Plot):**
```bash
uv run python tools/compare_all_models.py --data test_csv/ML_FormFlux_1_135.csv --output full_comparison.png
```

**Training Curves Comparison (Loss History):**
```bash
uv run python tools/plot_comparison_curves.py
```
*(Produces: `model_training_comparison.png`)*
