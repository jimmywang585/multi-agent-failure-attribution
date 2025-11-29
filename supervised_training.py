"""
System 3: Supervised Training (One Round)

Implements one epoch of supervised training using ground-truth labels:
- mistake_step[i]: decisive error step index
- mistake_agent[i]: decisive error agent ID

This is the standard ground-truth supervised baseline.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple

from discriminator import Discriminator, initialize_discriminators
from bt_consensus import BT_consensus


def aggregate_by_agent(p_step: torch.Tensor, agent_ids_i: List[str]) -> Dict[str, torch.Tensor]:
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


def supervised_train_one_round(
        L: List[List[Any]],
        agent_ids: List[List[str]],
        mistake_agent: List[Optional[str]],
        mistake_step: List[Optional[int]],
        K: int,
        model_config: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, Any]] = None
) -> List[Discriminator]:
    """
    Train K discriminators for one epoch using supervised ground-truth labels.
    
    Process:
    1. Initialize K discriminators
    2. Initialize optimizers for each discriminator
    3. Loop over each log once (1 epoch)
    4. For each discriminator and each log:
       - Run forward to get p_step, logits
       - Compute supervised loss (cross-entropy on true decisive step)
       - Optionally include agent-level loss
       - Backprop + optimizer step
    
    Args:
        L: List of log sequences (each is a list of steps)
        agent_ids: List of agent ID lists, one per log
                  (agent_ids[i] corresponds to L[i])
        mistake_agent: List of ground-truth agent IDs (one per log, can be None)
        mistake_step: List of ground-truth step indices (one per log, can be None)
        K: Number of discriminators to train
        model_config: Configuration dictionary for discriminators
        hyperparams: Dictionary with hyperparameters:
            - "use_agent_loss": bool, whether to include agent-level loss (default: False)
            - "lr": float, learning rate for optimizers (default: 0.001)
    
    Returns:
        List of K trained Discriminator instances
    
    Raises:
        ValueError: If input lengths don't match or if no valid labels found
    """
    # Validate inputs
    n_logs = len(L)
    if len(agent_ids) != n_logs:
        raise ValueError(f"L length ({n_logs}) must match agent_ids length ({len(agent_ids)})")
    if len(mistake_agent) != n_logs:
        raise ValueError(f"L length ({n_logs}) must match mistake_agent length ({len(mistake_agent)})")
    if len(mistake_step) != n_logs:
        raise ValueError(f"L length ({n_logs}) must match mistake_step length ({len(mistake_step)})")
    
    # Extract hyperparameters
    if hyperparams is None:
        hyperparams = {}
    use_agent_loss = hyperparams.get("use_agent_loss", False)
    lr = hyperparams.get("lr", 0.001)
    
    # Step 1: Initialize K discriminators
    discriminators = initialize_discriminators(K, model_config)
    
    # Step 2: Initialize optimizers for each discriminator
    optimizers = []
    for discriminator in discriminators:
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
        optimizers.append(optimizer)
    
    # Set discriminators to training mode
    for discriminator in discriminators:
        discriminator.train()
    
    # Step 3: Loop over each log once (1 epoch)
    for i in range(n_logs):
        log_steps = L[i]
        agent_ids_i = agent_ids[i]
        gt_agent = mistake_agent[i]
        gt_step = mistake_step[i]
        
        # Skip if no ground truth available
        if gt_step is None:
            continue
        
        # Validate ground truth step index
        if gt_step < 0 or gt_step >= len(log_steps):
            continue  # Skip invalid labels
        
        # Validate agent_ids_i length matches log_steps length
        if len(agent_ids_i) != len(log_steps):
            continue  # Skip if mismatch
        
        # For each discriminator
        for k, discriminator in enumerate(discriminators):
            optimizer = optimizers[k]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Run forward pass: p_step, logits = D_k.forward(L[i])
            p_step, logits = discriminator.forward(log_steps)
            
            # Compute supervised loss
            # Core term: cross-entropy on the true decisive step
            # loss_step = -log(p_step[gt_step])
            # Use negative log likelihood (cross-entropy with one-hot target)
            loss_step = -torch.log(p_step[gt_step] + 1e-8)  # Add epsilon to avoid log(0)
            
            # Optional: agent-level loss
            loss_agent = None
            if use_agent_loss and gt_agent is not None:
                # Aggregate step probabilities into agent probabilities
                P_agent = aggregate_by_agent(p_step, agent_ids_i)
                
                # Check if gt_agent exists in the agent probabilities
                if gt_agent in P_agent:
                    P_gt_agent = P_agent[gt_agent]
                    # loss_agent = -log(P_agent[gt_agent])
                    loss_agent = -torch.log(P_gt_agent + 1e-8)
                else:
                    # If gt_agent not found, skip agent loss for this example
                    loss_agent = None
            
            # Combine losses
            if loss_agent is not None:
                loss = loss_step + loss_agent
            else:
                loss = loss_step
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
    
    # Set discriminators back to evaluation mode
    for discriminator in discriminators:
        discriminator.eval()
    
    return discriminators


def supervised_predict_failure(log_steps: List[Any],
                               agent_ids_i: List[str],
                               discriminators: List[Discriminator],
                               use_ensemble: bool = True) -> Tuple[int, str]:
    """
    Predict failure step and agent using supervised-trained discriminators.
    
    Args:
        log_steps: Log sequence (list of steps) for a single example
        agent_ids_i: List of agent IDs corresponding to each step in the log
        discriminators: List of K trained Discriminator instances
        use_ensemble: If True, use BT consensus to ensemble predictions.
                     If False, use only the first discriminator.
    
    Returns:
        t_hat: Predicted step index (0-indexed)
        i_hat: Predicted agent ID (string)
    
    Raises:
        ValueError: If agent_ids_i length doesn't match number of steps
    """
    if not discriminators:
        raise ValueError("discriminators list cannot be empty")
    
    if not log_steps:
        raise ValueError("log_steps cannot be empty")
    
    num_steps = len(log_steps)
    
    if len(agent_ids_i) != num_steps:
        raise ValueError(f"agent_ids_i length ({len(agent_ids_i)}) must match "
                        f"number of steps ({num_steps})")
    
    if use_ensemble:
        # Ensemble using BT consensus or simple averaging
        all_p = []
        
        for discriminator in discriminators:
            p_k_step = discriminator.predict_step_distribution(log_steps)
            all_p.append(p_k_step)
        
        # Check if we should use simple averaging instead of BT
        use_bt = True  # Default to BT consensus
        if hasattr(supervised_predict_failure, '_use_simple_avg'):
            use_bt = not supervised_predict_failure._use_simple_avg
        
        if use_bt:
            p_group = BT_consensus(all_p)  # [T] tensor
        else:
            # Simple averaging
            p_stacked = torch.stack(all_p, dim=0)  # [K, T]
            p_group = p_stacked.mean(dim=0)  # [T]
        
        t_hat = torch.argmax(p_group).item()
    else:
        # Use only the first discriminator
        p_step = discriminators[0].predict_step_distribution(log_steps)
        t_hat = torch.argmax(p_step).item()
    
    # Validate t_hat is within bounds
    if t_hat < 0 or t_hat >= num_steps:
        raise ValueError(f"Predicted step index {t_hat} is out of bounds [0, {num_steps})")
    
    # Map to agent ID
    i_hat = agent_ids_i[t_hat]
    
    return t_hat, i_hat


def supervised_predict_all(logs: List[List[Any]],
                          agent_ids_list: List[List[str]],
                          discriminators: List[Discriminator],
                          use_ensemble: bool = True) -> List[Tuple[int, str]]:
    """
    Predict failures for all logs using supervised-trained discriminators.
    
    Helper function for batch inference/testing.
    
    Args:
        logs: List of log sequences (each is a list of steps)
        agent_ids_list: List of agent ID lists, one per log
        discriminators: List of K trained Discriminator instances
        use_ensemble: If True, use BT consensus ensemble. If False, use first discriminator only.
    
    Returns:
        List of (t_hat, i_hat) tuples, one per log
    """
    if len(logs) != len(agent_ids_list):
        raise ValueError(f"logs length ({len(logs)}) must match "
                        f"agent_ids_list length ({len(agent_ids_list)})")
    
    results = []
    
    for i in range(len(logs)):
        t_hat, i_hat = supervised_predict_failure(
            logs[i],
            agent_ids_list[i],
            discriminators,
            use_ensemble=use_ensemble
        )
        results.append((t_hat, i_hat))
    
    return results

