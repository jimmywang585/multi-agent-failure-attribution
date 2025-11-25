"""
Reward functions for System 1: Online Consensus Training

Implements the full reward structure including:
- Supervised rewards (when labels available)
- Unsupervised rewards (pairwise BT alignment, step-level, agent-level)
- Entropy and abstention rewards
"""

import torch
from typing import List, Dict, Any, Optional


def compute_agent_distribution(p_step: torch.Tensor, agent_ids_i: List[str]) -> Dict[str, torch.Tensor]:
    """
    Aggregate step probabilities into agent probabilities.
    
    For each unique agent, sum the probabilities of all steps assigned to that agent.
    Returns tensors to maintain gradient flow.
    
    Args:
        p_step: Step probability distribution, shape [T]
        agent_ids_i: List of agent IDs for each step, length T
    
    Returns:
        Dictionary mapping agent_id -> probability tensor (with gradients)
    """
    if len(p_step) != len(agent_ids_i):
        raise ValueError(f"p_step length ({len(p_step)}) must match "
                        f"agent_ids_i length ({len(agent_ids_i)})")
    
    P_agent = {}
    
    # Get unique agent IDs
    unique_agents = list(set(agent_ids_i))
    
    # For each unique agent, sum probabilities of steps assigned to that agent
    for agent_id in unique_agents:
        # Find all steps assigned to this agent
        agent_mask = torch.tensor(
            [agent_ids_i[t] == agent_id for t in range(len(agent_ids_i))],
            device=p_step.device,
            dtype=torch.bool
        )
        # Sum probabilities for this agent's steps
        P_agent[agent_id] = torch.sum(p_step[agent_mask])
    
    return P_agent


def supervised_reward(
    p_step: torch.Tensor,
    P_agent: Dict[str, torch.Tensor],
    gt_step: Optional[int],
    gt_agent: Optional[str],
    hyperparams: Dict[str, Any]
) -> torch.Tensor:
    """
    Compute supervised reward from ground-truth labels.
    
    R_sup = alpha_s * log p_step[gt_step] + alpha_a * log P_agent[gt_agent]
    
    Args:
        p_step: Step probability distribution, shape [T]
        P_agent: Agent probability distribution (dict mapping agent_id -> prob tensor)
        gt_step: Ground-truth step index (can be None)
        gt_agent: Ground-truth agent ID (can be None)
        hyperparams: Dictionary with:
            - "alpha_s": weight for step-level reward (default: 1.0)
            - "alpha_a": weight for agent-level reward (default: 1.0)
    
    Returns:
        Scalar reward tensor (can be zero if no labels)
    """
    alpha_s = hyperparams.get("alpha_s", 1.0)
    alpha_a = hyperparams.get("alpha_a", 1.0)
    epsilon = 1e-8
    
    # Initialize with zeros connected to computation graph (use p_step as base)
    R_sup_step = torch.zeros_like(p_step[0:1])  # [1] tensor connected to graph
    R_sup_agent = torch.zeros_like(p_step[0:1])  # [1] tensor connected to graph
    
    # Step-level reward
    if gt_step is not None and 0 <= gt_step < len(p_step):
        R_sup_step = torch.log(p_step[gt_step:gt_step+1] + epsilon)
    
    # Agent-level reward
    if gt_agent is not None and gt_agent in P_agent:
        P_gt_agent = P_agent[gt_agent]
        # Ensure P_gt_agent is a scalar tensor, then take log
        if P_gt_agent.dim() == 0:
            R_sup_agent = torch.log(P_gt_agent.unsqueeze(0) + epsilon)
        else:
            R_sup_agent = torch.log(P_gt_agent + epsilon)
    
    R_sup = alpha_s * R_sup_step.squeeze() + alpha_a * R_sup_agent.squeeze()
    
    return R_sup


def unsupervised_reward(
    p_k_step: torch.Tensor,
    P_k_agent: Dict[str, torch.Tensor],
    p_group_step: torch.Tensor,
    P_group_agent: Dict[str, torch.Tensor],
    hyperparams: Dict[str, Any]
) -> torch.Tensor:
    """
    Compute unsupervised reward combining pairwise BT alignment and distributional alignment.
    
    R_unsup = beta_pair * R_pair + beta_s * R_step + beta_a * R_agent
    
    Where:
    - R_pair: Pairwise BT alignment (discriminator preferences vs group BT preferences)
    - R_step: Step-level alignment (sum_t p_k[t] * log p_group[t])
    - R_agent: Agent-level alignment (sum_i P_k[i] * log P_group[i])
    
    Args:
        p_k_step: Discriminator k's step distribution, shape [T]
        P_k_agent: Discriminator k's agent distribution (dict)
        p_group_step: Group consensus step distribution, shape [T]
        P_group_agent: Group consensus agent distribution (dict)
        hyperparams: Dictionary with:
            - "beta_pair": weight for pairwise BT term (default: 1.0)
            - "beta_s": weight for step-level term (default: 1.0)
            - "beta_a": weight for agent-level term (default: 1.0)
    
    Returns:
        Scalar reward tensor
    """
    beta_pair = hyperparams.get("beta_pair", 1.0)
    beta_s = hyperparams.get("beta_s", 1.0)
    beta_a = hyperparams.get("beta_a", 1.0)
    epsilon = 1e-8
    
    T = len(p_k_step)
    
    # 1) Pairwise BT alignment term R_pair
    # Compute local pairwise preferences q_k(s>t) from p_k_step
    p_s = p_k_step.unsqueeze(-1)  # [T, 1]
    p_t = p_k_step.unsqueeze(-2)  # [1, T]
    q_k = p_s / (p_s + p_t + epsilon)  # [T, T]
    
    # Approximate group BT prob P_BT(s>t) from p_group_step
    log_p_group_s = torch.log(p_group_step + epsilon).unsqueeze(-1)  # [T, 1]
    log_p_group_t = torch.log(p_group_step + epsilon).unsqueeze(-2)  # [1, T]
    diff = log_p_group_s - log_p_group_t  # [T, T]
    P_BT = torch.sigmoid(diff)  # [T, T]
    
    # R_pair = avg over pairs (s < t) of: q_k * log P_BT + (1 - q_k) * log (1 - P_BT)
    log_P_BT = torch.log(P_BT + epsilon)
    log_1_minus_P_BT = torch.log(1 - P_BT + epsilon)
    
    R_pair_matrix = q_k * log_P_BT + (1 - q_k) * log_1_minus_P_BT  # [T, T]
    
    # Sum only over pairs where s < t (upper triangle, excluding diagonal)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=p_k_step.device), diagonal=1)
    R_pair = R_pair_matrix[mask].mean()
    
    # 2) Step-level alignment R_step
    # R_step = sum_t p_k_step[t] * log p_group_step[t]
    R_step = torch.sum(p_k_step * torch.log(p_group_step + epsilon))
    
    # 3) Agent-level alignment R_agent
    # R_agent = sum_i P_k_agent[i] * log P_group_agent[i]
    # Initialize from p_k_step to ensure gradient connection
    R_agent = torch.zeros_like(p_k_step[0:1])  # [1] tensor connected to graph
    
    for agent_id in P_k_agent:
        if agent_id in P_group_agent:
            P_k_i = P_k_agent[agent_id]
            P_group_i = P_group_agent[agent_id]
            # Ensure tensors are properly shaped
            if P_k_i.dim() == 0:
                P_k_i = P_k_i.unsqueeze(0)
            if P_group_i.dim() == 0:
                P_group_i = P_group_i.unsqueeze(0)
            R_agent = R_agent + (P_k_i * torch.log(P_group_i + epsilon))
    
    R_agent = R_agent.squeeze()
    
    # Combine terms
    R_unsup = beta_pair * R_pair + beta_s * R_step + beta_a * R_agent
    
    return R_unsup


def entropy_and_abstention_reward(
    p_k_step: torch.Tensor,
    P_k_agent: Dict[str, torch.Tensor],
    a: torch.Tensor,
    hyperparams: Dict[str, Any]
) -> torch.Tensor:
    """
    Compute entropy and abstention reward.
    
    R_stab = gamma_s * H_step + gamma_a * H_agent - lambda_abs * a
    
    Where:
    - H_step: Entropy of step distribution
    - H_agent: Entropy of agent distribution
    - a: Abstention rate
    
    Args:
        p_k_step: Step probability distribution, shape [T]
        P_k_agent: Agent probability distribution (dict)
        a: Abstention rate scalar in [0, 1]
        hyperparams: Dictionary with:
            - "gamma_s": weight for step entropy (default: 0.01)
            - "gamma_a": weight for agent entropy (default: 0.01)
            - "lambda_abs": weight for abstention penalty (default: 0.1)
    
    Returns:
        Scalar reward tensor
    """
    gamma_s = hyperparams.get("gamma_s", 0.01)
    gamma_a = hyperparams.get("gamma_a", 0.01)
    lambda_abs = hyperparams.get("lambda_abs", 0.1)
    epsilon = 1e-8
    
    # Step entropy: H_step = -sum_t p_k_step[t] * log p_k_step[t]
    H_step = -torch.sum(p_k_step * torch.log(p_k_step + epsilon))
    
    # Agent entropy: H_agent = -sum_i P_k_agent[i] * log P_k_agent[i]
    # Initialize from p_k_step to ensure gradient connection
    H_agent = torch.zeros_like(p_k_step[0:1])  # [1] tensor connected to graph
    for agent_id in P_k_agent:
        P_i = P_k_agent[agent_id]
        # Ensure tensor is properly shaped
        if P_i.dim() == 0:
            P_i = P_i.unsqueeze(0)
        H_agent = H_agent - (P_i * torch.log(P_i + epsilon))
    
    H_agent = H_agent.squeeze()
    
    # Entropy rewards (we want to maximize entropy, so add them)
    R_ent = gamma_s * H_step + gamma_a * H_agent
    
    # Abstention penalty (we want to minimize abstention, so subtract it)
    R_abs = -lambda_abs * a
    
    R_stab = R_ent + R_abs
    
    return R_stab

