# utils/privacy.py
import torch
import numpy as np
from typing import Optional

def add_laplacian_noise(tensor: torch.Tensor, lambda_param: float,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """Add zero-mean Laplacian noise for Local Differential Privacy"""
    if lambda_param <= 0:
        return tensor.clone()
    
    if device is None:
        device = tensor.device
    
    # ✅ Fixed: Generate noise on correct device
    noise = torch.from_numpy(
        np.random.laplace(loc=0.0, scale=lambda_param, size=tensor.shape)
    ).to(device).float()
    
    return tensor + noise