"""
Step 4: consensus_loss()

Implements the consensus loss function for training discriminators.

The loss encourages discriminators to align with the BT consensus distribution
while maintaining reasonable entropy (avoiding overconfidence).
"""

import torch
from typing import Dict, Any


def consensus_loss(p_k_step: torch.Tensor,
                   p_group: torch.Tensor,
                   hyperparams: Dict[str, Any]) -> torch.Tensor:
    """
    Compute consensus loss for a single discriminator's step distribution.
    
    Loss = beta_cons * KL(p_group || p_k) + alpha_entropy * (-H(p_k))
    
    Where:
    - KL(p_group || p_k) = sum_t p_group(t) * (log p_group(t) - log p_k(t))
    - H(p_k) = -sum_t p_k(t) * log p_k(t) (entropy)
    - The entropy term is subtracted to encourage higher entropy (less overconfident)
    
    Args:
        p_k_step: Discriminator k's distribution over steps for a single log, shape [T]
        p_group: BT consensus distribution over steps for the same log, shape [T]
        hyperparams: Dictionary containing:
            - "beta_cons": weight for KL divergence term
            - "alpha_entropy": weight for entropy regularization (typically small, e.g. 0.01)
    
    Returns:
        Scalar loss tensor suitable for loss.backward()
    """
    # Validate inputs
    if p_k_step.shape != p_group.shape:
        raise ValueError(f"p_k_step and p_group must have same shape. "
                        f"Got {p_k_step.shape} and {p_group.shape}")
    
    if p_k_step.dim() != 1:
        raise ValueError(f"p_k_step must be 1D tensor, got shape {p_k_step.shape}")
    
    if p_group.dim() != 1:
        raise ValueError(f"p_group must be 1D tensor, got shape {p_group.shape}")
    
    # Extract hyperparameters
    beta_cons = hyperparams.get("beta_cons", 1.0)
    alpha_entropy = hyperparams.get("alpha_entropy", 0.01)
    
    # Small epsilon to avoid log(0)
    epsilon = 1e-8
    
    # Add epsilon to probabilities to avoid numerical issues
    p_k_safe = p_k_step + epsilon
    p_group_safe = p_group + epsilon
    
    # Normalize to ensure they sum to 1 (after adding epsilon)
    p_k_safe = p_k_safe / p_k_safe.sum()
    p_group_safe = p_group_safe / p_group_safe.sum()
    
    # Core term: KL divergence KL(p_group || p_k)
    # KL(p_group || p_k) = sum_t p_group(t) * (log p_group(t) - log p_k(t))
    log_p_group = torch.log(p_group_safe)
    log_p_k = torch.log(p_k_safe)
    
    # Compute KL divergence
    L_KL = torch.sum(p_group_safe * (log_p_group - log_p_k))
    
    # Entropy bonus: -H(p_k) to encourage higher entropy
    # H(p_k) = -sum_t p_k(t) * log p_k(t)
    # We subtract H(p_k) in the loss to encourage higher entropy
    H_p_k = -torch.sum(p_k_safe * log_p_k)
    L_ent = -H_p_k  # Negative entropy (subtract to encourage higher entropy)
    
    # Final loss: weighted combination
    loss = beta_cons * L_KL + alpha_entropy * L_ent
    
    return loss

