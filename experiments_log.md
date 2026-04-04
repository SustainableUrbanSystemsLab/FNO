# FNO Training Experiments Log

> [!IMPORTANT]
> This is the official record of the architectural tuning and configuration testing designed to help FNO capture sharp wake boundaries and overcome spectral blurring (Gibbs Phenomenon). Use this to track the performance of `Standard`, `Hybrid`, `PINN`, and `Geo-FNO` models against the `Conditional Transformer`.

| Experiment Scenario | Architecture | Modes | Width | Wake Weight | Grad Weight | Spectral W. | Status / Visual Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01: Baseline Dist** | Standard | 32 | 64 | 0.0 | 0.0 | 0.0 | **Archived** (Pure MSE, blurry wakes, no physics) |
| **02: Baseline Hybrid**| Hybrid | 32 | 64 | 0.3 | 0.5 | 0.05 | **Archived** (Better spatial attention, soft edges) |
| **03: Baseline PINN** | PINN | 32 | 64 | 1.0 | 2.0 | 0.05 | **Archived** (Strict math boundaries, hard to train) |
| **04: High Capacity** | Standard | 48 | 128 | 1.0 | 1.5 | 0.001 | **Waiting on Cluster** (Testing high-freq wave retention) |
| **05: Implicit Mapping** | Geo-FNO | 48 | 128 | 1.0 | 1.5 | 0.001 | **Waiting on Cluster** (Testing latent geometric deformation) |

## Key Insights & Theoretical Causes of Blur

1. **The Fourier Truncation Problem (Gibbs Phenomenon)**
   The standard FNO processes wind fields as global sine/cosine waves rather than localized pixels. Truncating these waves (e.g., `modes=32`) physically deletes the ultra-high-frequency components required to draw a perfectly sharp vertical edge (like a building wall terminating into a wake). This forces the model mathematically to draw blurry, smooth approximations. Increasing modes to `48` fights this directly.

2. **Spectral Bias Constraint**
   A high `spectral_weight` (e.g., 0.05, typical in older configs) actively punishes the network for drawing sharp edges, enforcing low-frequency smooth solutions to the Navier-Stokes equations. Crushing this value down to `0.001` unlocks aerodynamic sharpness.

3. **Loss Function Geometry (L2 vs L1)**
   Image architectures (Transformers/Pix2Pix) typically utilize L1 Loss (Absolute Error) which encourages sharp spatial distinctions. FNO defaults to L2 (Mean Squared Error) which calculates the easiest mathematical path by just projecting a blurry gradient instead of stepping down strictly. Combining massive `gradient_weights` (10x higher) simulates L1 boundary pressure.

4. **Information Capacity (Width)**
   Image networks utilize immense channel widths. By pumping the FNO width from `64 -> 128`, the model drastically increases the internal memory it uses to track micro-velocities inside the wake core.

## Scheduled / Future Technical Fixes to try

- **[ ] L1 Wake-Boundary Injection (Sobel Filtering)**
  - *Plan:* Modify the inner FNO loss loop (`sensor_weighted_mse`) to explicitly calculate an L1 (Absolute) Gradient Loss strictly evaluated *inside* the semantic wake mask. This will punish the FNO severely for blurring across the building step.
  
- **[ ] SDF-Thresholding Intervention**
  - *Plan:* An immediate, cheap "hack" during inference. Read the existing input `SDF` (Signed Distance Function), and automatically brutalize the FNO's output prediction to $0.0$ if the grid cell is $<= 0$ meters from the building surface, wiping out ghost winds.

- **[ ] U-Net Convolutional Recovery (True U-FNO)**
  - *Plan:* If the implicit Geo-FNO fails, build a literal `U-FNO` block that bypasses the Spectral layers entirely via Convolutional Skip Connections. The FNO will handle background global wind flow, and the CNN skips will map the sharp geometric edges at the very final layer exactly like the Conditional Transformer does.
