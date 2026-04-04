# ICE Cluster Command Cheat Sheet

Make sure to run `git pull` before running any tests.

### 1. Standard Distributed FNO 
**Train:**
```bash
bash slurm/deploy_ice.sh --config configs/config_test.toml --script pipelines/train/distributed.py --gpu h200 --ngpus 1 --fresh
```
**Evaluate:**
```bash
uv run python tools/evaluate_fno.py --model fno_test_weights.pth --data test_csv/ML_FormFlux_1_135.csv --output dist_diagnostic.png
```
*(Produces: `dist_diagnostic.png`)*

---

### 2. Hybrid FNO (Physics + Data)
**Train:**
```bash
bash slurm/deploy_ice.sh --config configs/config_test.toml --script pipelines/train/hybrid.py --gpu h200 --ngpus 1 --fresh
```
**Evaluate:**
```bash
uv run python tools/diagnose_hybrid.py --model hybrid_fno_weights.pth --data test_csv/ML_FormFlux_1_135.csv --output hybrid_diagnostic.png
```
*(Produces: `hybrid_diagnostic.png`)*

---

### 3. PINN FNO (Strict Physics Informed)
**Train:**
```bash
bash slurm/deploy_ice.sh --config configs/config_test.toml --script pipelines/train/pinn.py --gpu h200 --ngpus 1 --fresh
```
**Evaluate:**
```bash
uv run python tools/diagnose_pinn.py --model pinn_fno_weights.pth --data test_csv/ML_FormFlux_1_135.csv --output pinn_diagnostic.png
```
*(Produces: `pinn_diagnostic.png`)*

---

### 4. Geo-FNO (Geometry-Aware Deep Deformation)
**Train:**
```bash
bash slurm/deploy_ice.sh --config configs/config_test.toml --script pipelines/train/geo.py --gpu h200 --ngpus 1 --fresh
```
**Evaluate:**
```bash
uv run python tools/diagnose_geo.py --model geo_fno_weights.pth --data test_csv/ML_FormFlux_1_135.csv --output geo_diagnostic.png
```
*(Produces: `geo_diagnostic.png`)*

---

### 5. Standard Non-Distributed FNO (Legacy)
**Train:**
```bash
bash slurm/deploy_ice.sh --config configs/config_test.toml --script pipelines/train/standard.py --gpu h200 --ngpus 1 --fresh
```
**Evaluate:**
```bash
uv run python tools/evaluate_fno.py --model fno_mag_weights.pth --data test_csv/ML_FormFlux_1_135.csv --output standard_diagnostic.png
```
*(Produces: `standard_diagnostic.png`)*
