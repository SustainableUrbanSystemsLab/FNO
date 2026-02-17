# FNO Model Data Structure

This document defines the Inputs (X) and Target (Y) used for the Wind Prediction FNO Model.

## 1. Input Tensor (X)
**Shape:** `[8, Height, Width]`
The input consists of **8 Channels** representing the physical geometry and wind conditions.

| Channel Index | Name | Description | Normalization | Range (Approx) |
| :--- | :--- | :--- | :--- | :--- |
| **0** | `SDF` | Signed Distance Field (dist to nearest wall) | Divided by **200.0** | `[0.0, 1.0]` |
| **1** | `Bldg_height` | Height of the building at this pixel | Divided by **50.0** | `[0.0, 1.0]` |
| **2** | `Z_relative` | Height slice of the simulation (Z coord) | Divided by **10.0** | `[0.0, 1.0]` |
| **3** | `U_over_Uref` | Background wind ratio (inlet profile) | Multiplied by **2.0** | `[0.2, 2.0]` |
| **4** | `X_local` | X dist from building center | Divided by **500.0** | `[-1.0, 1.0]` |
| **5** | `Y_local` | Y dist from building center | Divided by **500.0** | `[-1.0, 1.0]` |
| **6** | `dir_sin` | Sine of Wind Direction | None | `[-1.0, 1.0]` |
| **7** | `dir_cos` | Cosine of Wind Direction | None | `[-1.0, 1.0]` |

## 2. Target Tensor (Y)
**Shape:** `[1, Height, Width]`
The model predicts a single scalar value representing the **Wake Deficit**.

| Name | Formula | Description |
| :--- | :--- | :--- |
| **Target_Delta_U** | `(Mag_U - U_ref) / U_ref` | Fractional change in wind speed. |

### Interpretation:
- **0.0**: No change (Wind speed = U_ref).
- **-0.5**: 50% drop in speed (Strong Wake).
- **-1.0**: Stagnation (Speed = 0).

### Reconstruction:
To get the physical wind speed (`Mag_U`) from the model prediction:
```python
Mag_U = (Prediction * U_ref) + U_ref
```
