from __future__ import absolute_import
from __future__ import print_function

import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the perturbation
pert = torch.load('./attacks/pert')

# Convert to numpy array and adjust the shape
pert = pert.numpy()

# Check the shape of pert
if pert.shape[0] == 3:  # If in [C, H, W] format
    pert = np.transpose(pert, [1, 2, 0])  # Convert to [H, W, C] for visualization

# Normalize the perturbation to the range [0, 1] for proper display
pert = (pert - pert.min()) / (pert.max() - pert.min())

# Plot the perturbation
plt.axis('off')
plt.imshow(pert)
plt.show()
