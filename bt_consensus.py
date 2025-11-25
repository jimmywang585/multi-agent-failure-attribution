"""
Step 3: BT_consensus()

Implements Bradley-Terry consensus aggregation of multiple step probability distributions.

Converts K probability distributions over steps into a single consensus distribution
using pairwise preference aggregation and BT score fitting.
"""

import torch
import torch.nn as nn
from typing import List


def BT_consensus(p_steps_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute Bradley-Terry consensus distribution from multiple step probability distributions.
    
    Input: list of K tensors p_k[t] over steps (each length T)
    Output: p_group[t] = consensus distribution over steps
    
    Steps:
        1) Convert each p_k into pairwise preferences q_k(s > t)
        2) Average to get q_bar(s > t)
        3) Fit BT scores r_t using gradient ascent
        4) Convert r_t into the final distribution: p_group[t] = softmax(r_t)
    
    Args:
        p_steps_list: List of K probability tensors, each of shape [T] where T is number of steps.
                     Each tensor should be a valid probability distribution (sums to 1).
    
    Returns:
        p_group: Consensus probability distribution over steps, shape [T]
    """
    if not p_steps_list:
        raise ValueError("p_steps_list cannot be empty")
    
    # Get number of discriminators K and number of steps T
    K = len(p_steps_list)
    T = p_steps_list[0].shape[0]
    
    # Validate all distributions have same length
    for k, p_k in enumerate(p_steps_list):
        if p_k.shape[0] != T:
            raise ValueError(f"All distributions must have same length. p_{k} has length {p_k.shape[0]}, expected {T}")
        if p_k.dim() != 1:
            raise ValueError(f"Each distribution must be 1D tensor. p_{k} has shape {p_k.shape}")
    
    # Stack all distributions: [K, T]
    p_steps = torch.stack(p_steps_list, dim=0)  # [K, T]
    
    # Step 1: Convert each p_k into pairwise preferences q_k(s > t)
    # Preference q_k(s > t) = probability that step s is preferred over step t
    # Using logistic form: q = p_k[s] / (p_k[s] + p_k[t])
    # This gives us q_k(s > t) for all pairs (s, t)
    
    # Expand dimensions for pairwise comparison
    # p_steps: [K, T]
    # p_s: [K, T, 1] - probability of step s
    # p_t: [K, 1, T] - probability of step t
    p_s = p_steps.unsqueeze(-1)  # [K, T, 1]
    p_t = p_steps.unsqueeze(-2)  # [K, 1, T]
    
    # Compute pairwise preferences: q_k(s > t) = p_k[s] / (p_k[s] + p_k[t])
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    q_k = p_s / (p_s + p_t + epsilon)  # [K, T, T]
    
    # Step 2: Average to get q_bar(s > t)
    # Average over K discriminators
    q_bar = q_k.mean(dim=0)  # [T, T]
    
    # Step 3: Fit BT scores r_t using gradient ascent
    # BT model: q_bar(s > t) ≈ σ(r_s - r_t) where σ is sigmoid
    # We optimize: sum_{s < t} [ q_bar * log σ(r_s - r_t) + (1 - q_bar) * log σ(r_t - r_s) ]
    
    # Initialize BT scores r_t (learnable parameters)
    r = nn.Parameter(torch.zeros(T, requires_grad=True))
    
    # Optimizer for gradient ascent (maximize log-likelihood)
    optimizer = torch.optim.Adam([r], lr=0.1)
    
    # Number of optimization iterations
    num_iterations = 100
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Compute log-likelihood (negative for minimization, so we'll negate it)
        # We want to maximize: sum_{s,t} [ q_bar(s>t) * log σ(r_s - r_t) + (1 - q_bar(s>t)) * log σ(r_t - r_s) ]
        
        # Expand r for pairwise comparison
        r_s = r.unsqueeze(-1)  # [T, 1]
        r_t = r.unsqueeze(-2)  # [1, T]
        
        # Compute log probabilities
        # log σ(r_s - r_t) and log σ(r_t - r_s)
        log_sigma_s_gt_t = torch.nn.functional.logsigmoid(r_s - r_t)  # [T, T]
        log_sigma_t_gt_s = torch.nn.functional.logsigmoid(r_t - r_s)  # [T, T]
        
        # Compute log-likelihood
        # For each pair (s, t) where s < t, we have:
        # q_bar(s>t) * log σ(r_s - r_t) + (1 - q_bar(s>t)) * log σ(r_t - r_s)
        log_likelihood = q_bar * log_sigma_s_gt_t + (1 - q_bar) * log_sigma_t_gt_s
        
        # Sum only over pairs where s < t (upper triangle, excluding diagonal)
        # Create mask for upper triangle (s < t)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=r.device), diagonal=1)
        log_likelihood_masked = log_likelihood[mask]
        
        # Total log-likelihood (we want to maximize this)
        total_log_likelihood = log_likelihood_masked.sum()
        
        # Negate for minimization (since optimizers minimize)
        loss = -total_log_likelihood
        
        # Backward pass
        loss.backward()
        
        # Gradient ascent step
        optimizer.step()
    
    # Step 4: Convert r_t into the final distribution
    # p_group[t] = softmax(r_t)
    # Detach r from computation graph since optimization is complete
    r_final = r.detach()
    p_group = torch.softmax(r_final, dim=0)  # [T]
    
    return p_group

