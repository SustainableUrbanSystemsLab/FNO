# FNO Architectures: Technical Breakdown

This document provides a highly specialized breakdown of the 4 unique Fourier Neural Operator (FNO) models developed and tested in this repository. It explains the structural flow of data, the theoretical purpose behind each architecture, and their mathematical relationship to analyzing aerodynamic wakes.

---

## 1. Standard FNO (`core/models/fno2d.py`)
**"The Baseline Global Solver"**

### How it Works:
The Standard FNO completely ignores the concept of localized "pixels." Instead of looking at adjacent grid cells like a Convolutional Neural Network (CNN) does, it takes the entire $128 \times 128$ wind field grid and applies a Fast Fourier Transform (FFT) across it. 
It converts the spatial mapping of wind into a collection of sine and cosine waves (frequencies). It multiplies these frequency waves by learnable weights, and then runs an Inverse-FFT back to physical space.

### Structural Flow:
1. **Linear Lift**: Input $(8, H, W)$ is mapped up to the hidden `width` dimension $(128, H, W)$.
2. **Spectral Layers (x4)**: The data splits. One half goes through standard spatial linear weights. The other half goes into the Fourier Domain, gets truncated to the first `48` modes, multiplied by complex weights, and transformed back. They are added together.
3. **Projection**: It steps back down to $(1, H, W)$ for the wind magnitude.

### Mathematical Benefits & Drawbacks:
**✅ Benefit (Resolution Invariance):** The math is strictly continuous. Because it learns weights in the frequency domain $k$, you can train the model on a $64 \times 64$ grid, and evaluate it on a $1024 \times 1024$ grid perfectly without retraining.
**❌ Drawback (The Gibbs Phenomenon):** Mathematically, representing a completely sharp, sheer drop (like the physical wall of a building) requires a Heaviside Step Function $H(x)$. Trying to draw a step function using Sine waves requires an infinite limit of frequencies. Because we truncate to $k \le 48$ modes to save GPU memory, the model physically deletes the high-frequency waves, leaving a blurry, rippling approximation of the wakes.

---

## 2. Hybrid FNO (`core/models/hybrid.py`)
**"The U-Net Structural Compromise"**

### How it Works:
The Hybrid FNO explicitly attempts to cure the Standard FNO's blurry boundaries. It merges the global physics-solving ability of the FNO with the sharp, edge-detecting ability of localized CNN layers (similar to a Pix2Pix or U-Net).

### Structural Flow:
1. **Local Convolution Encoder**: The $(8, H, W)$ input goes through local CNNs to extract structural boundary features.
2. **Spectral Core**: The FNO operates on the encoded state, solving the global pressure and velocity physics.
3. **Skip Connections**: The sharp boundary data from Step 1 is copied and passed completely *around* the FNO block, merging directly with the FNO's output.
4. **Decoder**: A final CNN layer takes the global wind profile and mathematically fuses it with the sharp structural edges from the skip connections.

### Mathematical Benefits & Drawbacks:
**✅ Benefit (Sharp Edge Recovery):** CNN filters $w \ast x$ operate in the spatial domain, allowing them to perfectly mimic L1 boundaries and easily reconstruct high-frequency step-functions without the infinite-wave math required by Fourier filters.
**❌ Drawback (Receptive Field Bottleneck):** While the CNN fixes the wake boundaries, it destroys the FNO's *Resolution Invariance*. CNN math is heavily tied to the specific pixel size of the training grid. Additionally, if environmental pressure changes thousands of meters away, the CNN requires $O(n)$ layers to mathematically propagate that pressure to the building, limiting its global fluid understanding.

---

## 3. Strict PINN-FNO (`core/models/pinn_fno.py`)
**"The Mathematical Disciplinarian"**

### How it Works:
The fundamental architecture is an FNO, but it interacts with a massive, over-engineered internal loss loop that actively calculates multi-dimensional calculus during training. It doesn't just guess wind speed; it calculates the divergence ($\nabla \cdot U$) and gradients ($\nabla U$) of its *own* predictions.

### Structural Flow:
1. **Forward Pass**: Standard FNO projection.
2. **Physics Calculator**: While analyzing the output map, it computes:
    - **Continuity Loss**: (Mass Conservation). Ensures air isn't magically appearing or disappearing into nothing.
    - **Momentum Loss**: Ensures the velocity gradients conform to the Navier-Stokes pressure shifts.
    - **Wake Peak Guard**: Specifically penalizes high velocity values if they occur directly in the wake zone.
    
### Mathematical Benefits & Drawbacks:
**✅ Benefit (Out-of-Distribution Robustness):** Standard models can memorize the training data and fail horribly if given a slightly taller building. The PINN is mathematically bound by the Divergence Theorem ($\nabla \cdot \mathbf{U} = 0$). Even if you give it a bizarre building shape, the network is physically restricted from drawing impossible "phantom winds."
**❌ Drawback (Gradient Stiffening & Conflict):** The optimizer is trying to minimize a multi-objective loss function: $\mathcal{L} = \mathcal{L}_{mse} + \lambda_1 \|\nabla \cdot \mathbf{U}\|^2$. The gradients required to minimize the data error (MSE) often point in the exact mathematically opposite direction as the Navier-Stokes physics constraints. This causes "Gradient Stiffening", where the optimizer gets trapped in a local minimum, leading to flat, uninspired predictions because it gives up trying to satisfy both equations.

---

## 4. Geo-FNO (`core/models/geo_fno.py`)
**"The Implicit Coordinate Warper"**

### How it Works:
Standard FNOs fundamentally assume the physical grid is completely uniform and empty. They don't mathematically understand that a solid, physical building obstacle exists *inside* the grid layout. Geo-FNO solves this by deforming the input space based on the building boundary.

### Structural Flow:
1. **Local Geometry Encoder**: It reads the Signed Distance Function (SDF). Before the FFTs activate, this encoder calculates a non-linear latent grid mapping. It stretches and compresses the grid coordinates so the "building" is mathematically pushed out of the computational domain.
2. **Fourier Transform**: The spectral convolutions happen in this newly deformed, empty space where physics flows smoothly.
3. **Boundary Reconstructor**: After returning to physical space, an SDF-conditioned mask guarantees razor-sharp drops in velocity precisely where the geometry layer dictates a wall exists.

### Mathematical Benefits & Drawbacks:
**✅ Benefit (Topological Purity):** By mapping the physical world $x$ into a latent space $\xi$, the Fourier Transform operates over an uninterrupted, continuous uniform space without hitting sharp boundaries. The model effectively solves the fluid dynamics in an "empty room", and the boundary reconstructor forcefully translates it back against the physical wall, eliminating the spectral blurring.
**❌ Drawback (Jacobian Singularity):** The mathematical transformation from physical space to the latent mapped space relies on the condition of the Jacobian matrix ($J = \frac{\partial \xi}{\partial x}$). If you have a highly complex, rugged, or concave building geometry, the spatial deformation becomes too extreme, the Jacobian approaches $0$, and the FFT mathematically collapses into severe floating-point errors.
