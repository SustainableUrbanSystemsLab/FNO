# FNO Mag Training & Inference (Grasshopper -> CSV)
This package contains scripts to train a Fourier Neural Operator (FNO) to predict **dimensionless wind speed magnitude** (mag_U) from Grasshopper exports (CSV).

## Files
- `fno2d_model.py` : FNO model and loss utility
- `gh_to_fno.py` : helpers to build (1,C,H,W) input tensor from GH CSV columns
- `train_fno_mag.py` : training script. Expects a folder `train_csv/` with one CSV per sample.
- `run_inference_mag.py` : inference script that reads a single CSV and writes predictions
- `README.md` : this file

## Input CSV format (rows = points)
Required input columns (per sample):
- `SDF`, `Bldg_height`, `Z_relative`, `U_at_z`, `X_coords`, `Y_coords`, `dir_sin`, `dir_cos`

Required target (dimensionless magnitude) **one of**:
- `mag_U` (already dimensionless) OR `mag_U_dimensionless` OR `mag_dimensionless`
  (the scripts prefer dimensionless mag; they will NOT divide by U_ref)
- Alternatively you may provide `Ux_dimensionless`, `Uy_dimensionless`, `Uz_dimensionless` and the script will compute mag.

Optional columns:
- `is_sensor` : 1 for real sensor locations (loss will be applied there), 0 otherwise. Default=1
- `U_ref` : only used at inference to output physical mag (optional)

## How to use
1. Put training CSV files into `train_csv/` (one file per sample). The files can be sparse or masked (points don't need to form a perfect rectangle).
2. Install requirements:
   ```bash
   pip install torch pandas numpy
   ```
   Or usage with `uv` (Recommended Python 3.10-3.12):
   ```bash
   uv python pin 3.12
   uv sync
   # or
   uv pip install -r requirements.txt
   ```
3. Train:
   ```bash
   python train_fno_mag.py
   ```
   Model saved as `fno_mag_weights.pth`.
   Intermediate epoch checkpoints are saved in the `epochs/` folder.
   
   **Note**: The training script supports sparse/masked grids (where points are missing). Infinite target values are automatically ignored in the loss.
4. Inference:
   - Edit `run_inference_mag.py` to set `CSV` path and `MODEL` path if needed.
   - Run:
   ```bash
   python run_inference_mag.py
   ```
   Output file `*_mag_pred.csv` will contain `mag_U_pred_dimensionless` (and `mag_U_pred` if `U_ref` exists).

## Notes
- The code assumes inputs are **dimensionless** where indicated (you said you'll provide dimensionless `U_at_z` and `mag_U`). The scripts do not re-normalize `U_at_z`.
- If your grid ordering differs, provide explicit `H,W` in the scripts or include integer grid indices in CSV and adapt `gh_to_fno.py` indexing.
- For vector outputs (Ux,Uy,Uz) or multi-task training, use the other training scripts provided earlier (not included in this ZIP).

## License
Use as you like. If you want changes (e.g., multi-task, sensor-only loss tweaks, augmentation), tell me the specifics.
