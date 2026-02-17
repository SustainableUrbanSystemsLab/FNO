import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_sample():
    try:
        # Load data
        data = torch.load('sample_data.pt', map_location='cpu')
        X = data['X'].squeeze(0) if data['X'].dim() == 4 else data['X'] # Ensure [C, H, W]
        Y = data['Y'].squeeze(0) if data['Y'].dim() == 4 else data['Y'] # Ensure [1, H, W]
        names = data['channel_names']
        
        # Prepare Plot
        num_channels = X.shape[0]
        num_plots = num_channels + 1 # +1 for Target
        
        cols = 3
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        axes = axes.flatten()
        
        # Plot Inputs
        for i in range(num_channels):
            ax = axes[i]
            img = X[i].numpy()
            im = ax.imshow(img, cmap='viridis', origin='lower')
            ax.set_title(f"Input: {names[i]}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
            
        # Plot Target
        ax_t = axes[num_channels]
        img_t = Y[0].numpy()
        im_t = ax_t.imshow(img_t, cmap='plasma', origin='lower')
        ax_t.set_title("Target: Delta U\n(Simulated for Training)")
        plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)
        ax_t.axis('off')
        
        # Hide extra
        for j in range(num_channels + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        out_file = "sample_data_viz.png"
        plt.savefig(out_file, dpi=150)
        print(f"Saved visualization to {out_file}")
        
    except FileNotFoundError:
        print("Error: sample_data.pt not found. Run export_sample_tensors.py first.")

if __name__ == "__main__":
    plot_sample()
