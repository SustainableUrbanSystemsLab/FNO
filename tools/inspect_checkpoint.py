import torch
import os

path = 'fno_mag_weights.pth'
if not os.path.exists(path):
    print(f"{path} not found")
else:
    try:
        sd = torch.load(path, map_location='cpu')
        print(f"Loaded {path}")
        
        # Check specific keys
        if 'out_proj.0.weight' in sd:
            print(f"out_proj.0.weight: {sd['out_proj.0.weight'].shape}")
        
        if 'in_proj.weight' in sd:
            print(f"in_proj.weight: {sd['in_proj.weight'].shape}")
            
        # Infer width
        # in_proj is (width, in_channels, 1, 1)
        # out_proj.0 is (128, width, 1, 1)
        
    except Exception as e:
        print(f"Error loading: {e}")
