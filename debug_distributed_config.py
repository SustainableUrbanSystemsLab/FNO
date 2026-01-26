
import os
import sys
import torch
from train_fno_distributed import load_config, PEAK_WEIGHT, GRAD_WEIGHT, SPECTRAL_WEIGHT
from fno2d_model import physics_loss

print(f"Loaded PEAK_WEIGHT: {PEAK_WEIGHT}")
print(f"Loaded GRAD_WEIGHT: {GRAD_WEIGHT}")
print(f"Loaded SPECTRAL_WEIGHT: {SPECTRAL_WEIGHT}")

# Mock run of physics_loss to checks if it produces non-zero peak loss
print("\nTesting physics_loss with dummy data...")
B, C, H, W = 2, 1, 32, 32
pred = torch.rand(B, C, H, W)
target = torch.rand(B, C, H, W)
# Make some high values in target to trigger peak loss
target[0, 0, 10:15, 10:15] = 5.0 

loss, components = physics_loss(
    pred, target, 
    grad_weight=GRAD_WEIGHT, 
    spectral_weight=SPECTRAL_WEIGHT, 
    peak_weight=PEAK_WEIGHT, 
    return_components=True
)

print(f"Total Loss: {loss.item()}")
print("Components:", components)

if components['peak_loss'] == 0.0 and PEAK_WEIGHT > 0:
    print("FAIL: Peak weight is > 0 but peak_loss is 0.0!")
elif components['peak_loss'] > 0:
    print("SUCCESS: Peak loss is being calculated.")
else:
    print(f"Ambiguous: Peak weight is {PEAK_WEIGHT}")
