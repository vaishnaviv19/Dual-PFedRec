# utils/privacy.py
"""
Privacy-Preserving Mechanisms (Paper Section 6.6)
Local Differential Privacy (LDP) with Laplacian noise
"""

import torch
import numpy as np
from typing import Optional


def add_laplacian_noise(tensor: torch.Tensor, 
                       lambda_param: float,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Add zero-mean Laplacian noise for Local Differential Privacy
    
    θₘ = θₘ + Laplace(0, λ)  [Paper Section 6.6]
    
    Args:
        tensor: Tensor to add noise to (e.g., item embeddings)
        lambda_param: Noise scale parameter λ (higher = more privacy, less utility)
        device: Target device for noise tensor
        
    Returns:
        noisy_tensor: Tensor with Laplacian noise added
    """
    if lambda_param <= 0:
        return tensor.clone()
    
    if device is None:
        device = tensor.device
    
    # Generate Laplacian noise with same shape as tensor
    noise = torch.from_numpy(
        np.random.laplace(loc=0.0, scale=lambda_param, size=tensor.shape)
    ).float().to(device)
    
    return tensor + noise


def compute_privacy_budget(epsilon: float, delta: float = 1e-5, 
                          num_rounds: int = 100) -> Dict[str, float]:
    """
    Estimate cumulative privacy budget using advanced composition
    
    Args:
        epsilon: Per-round privacy parameter
        delta: Failure probability
        num_rounds: Number of federated rounds
        
    Returns:
        budget: Dictionary with total epsilon and delta
    """
    # Advanced composition theorem (simplified)
    # ε_total ≈ √(2k log(1/δ')) * ε + kε(e^ε - 1)
    import math
    
    k = num_rounds
    delta_prime = delta
    
    # First term: Gaussian-like composition
    term1 = math.sqrt(2 * k * math.log(1/delta_prime)) * epsilon
    
    # Second term: Linear composition with amplification
    term2 = k * epsilon * (math.exp(epsilon) - 1)
    
    total_epsilon = term1 + term2
    
    return {
        "per_round_epsilon": epsilon,
        "total_epsilon": total_epsilon,
        "delta": delta,
        "num_rounds": num_rounds
    }