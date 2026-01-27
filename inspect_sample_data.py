import torch

def inspect():
    try:
        print("Loading sample_data.pt...")
        # Map location to cpu to avoid CUDA errors if saved on GPU
        data = torch.load('sample_data.pt', map_location='cpu')
        
        X = data['X']
        Y = data['Y']
        names = data['channel_names']

        print("\n=== Sample Data Inspection ===")
        print(f"X (Input) Tensor Shape:  {tuple(X.shape)}")
        print(f"Y (Target) Tensor Shape: {tuple(Y.shape)}")
        print("-" * 65)
        
        # Print stats per channel
        print(f"{'Channel Name':<20} | {'Min':<10} | {'Max':<10} | {'Mean':<10}")
        print("-" * 65)
        
        # Input Channels
        # X is [1, C, H, W] or [C, H, W]? Let's handle both.
        if X.dim() == 4:
            X_data = X[0]
        else:
            X_data = X

        for i, name in enumerate(names):
            vals = X_data[i].float()
            print(f"{name:<20} | {vals.min():<10.4f} | {vals.max():<10.4f} | {vals.mean():<10.4f}")
            
        print("-" * 65)
        # Target
        y_vals = Y.float()
        print(f"{'Target (Delta U)':<20} | {y_vals.min():<10.4f} | {y_vals.max():<10.4f} | {y_vals.mean():<10.4f}")
        print("-" * 65)
        
    except FileNotFoundError:
        print("Error: 'sample_data.pt' not found. Please run 'export_sample_tensors.py' first.")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect()
