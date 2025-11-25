"""
System 1: Online Consensus Training (Semi-supervised, Multi-pass)

Implements semi-supervised consensus training that trains K discriminators
using a full reward structure including:
- Supervised rewards (when labels available)
- Unsupervised rewards (pairwise BT alignment, step-level, agent-level)
- Entropy and abstention rewards

This is a multi-pass training system that can use both labeled and unlabeled logs.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple

from discriminator import Discriminator, initialize_discriminators
from bt_consensus import BT_consensus
from rewards import (
    compute_agent_distribution,
    supervised_reward,
    unsupervised_reward,
    entropy_and_abstention_reward
)


def online_consensus_train(
    L: List[List[Any]],
    agent_ids: List[List[str]],
    K: int,
    model_config: Optional[Dict[str, Any]] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    mistake_agent: Optional[List[Optional[str]]] = None,
    mistake_step: Optional[List[Optional[int]]] = None
) -> List[Discriminator]:
    """
    Train K discriminators using online consensus training (semi-supervised).
    
    Process:
    1. Initialize K discriminators and optimizers
    2. For each epoch:
       - For each log:
         - All discriminators predict p_k_step and abstention rates a_k
         - Build BT consensus p_group
         - Compute agent-level distributions P_k and P_group
         - Each discriminator updates using full reward structure:
           - Supervised reward (if labels available)
           - Unsupervised reward (pairwise BT, step-level, agent-level)
           - Entropy and abstention rewards
    3. Stop early if loss stabilizes or max_epochs reached
    
    Args:
        L: List of log sequences (each is a list of steps)
        agent_ids: List of agent ID lists, one per log
                  (agent_ids[i] corresponds to L[i])
        K: Number of discriminators to train
        model_config: Configuration dictionary for discriminators
        hyperparams: Dictionary with hyperparameters:
            - "lr": learning rate for optimizers (default: 0.001)
            - "num_epochs": max number of epochs (default: 10)
            - "loss_tolerance": threshold for early stopping (default: 1e-5)
            - "alpha_s", "alpha_a": supervised reward weights (default: 1.0)
            - "beta_pair", "beta_s", "beta_a": unsupervised reward weights (default: 1.0)
            - "gamma_s", "gamma_a": entropy reward weights (default: 0.01)
            - "lambda_abs": abstention penalty weight (default: 0.1)
            - "lambda_mix": mixing weight for supervised vs unsupervised on labeled logs (default: 0.5)
        mistake_agent: List of ground-truth agent IDs (one per log, can be None)
        mistake_step: List of ground-truth step indices (one per log, can be None)
    
    Returns:
        List of K trained Discriminator instances
    """
    # Validate inputs
    n_logs = len(L)
    if len(agent_ids) != n_logs:
        raise ValueError(f"L length ({n_logs}) must match agent_ids length ({len(agent_ids)})")
    
    # Handle optional labels (semi-supervised)
    if mistake_agent is None:
        mistake_agent = [None] * n_logs
    if mistake_step is None:
        mistake_step = [None] * n_logs
    
    if len(mistake_agent) != n_logs:
        raise ValueError(f"mistake_agent length ({len(mistake_agent)}) must match L length ({n_logs})")
    if len(mistake_step) != n_logs:
        raise ValueError(f"mistake_step length ({len(mistake_step)}) must match L length ({n_logs})")
    
    # Extract hyperparameters with defaults
    if hyperparams is None:
        hyperparams = {}
    
    lr = float(hyperparams.get("lr", 0.001))
    num_epochs = int(hyperparams.get("num_epochs", 10))
    loss_tolerance = float(hyperparams.get("loss_tolerance", 1e-5))
    lambda_mix = float(hyperparams.get("lambda_mix", 0.5))
    
    # Step 1: Initialize K discriminators + optimizers
    print(f"Initializing {K} discriminators...")
    discriminators = initialize_discriminators(K, model_config)
    print(f"✓ Initialized {K} discriminators")
    
    optimizers = []
    for k, D_k in enumerate(discriminators):
        optimizer_k = torch.optim.Adam(D_k.parameters(), lr=lr)
        optimizers.append(optimizer_k)
    
    # Set discriminators to training mode
    for D_k in discriminators:
        D_k.train()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"GPU memory after initialization: {allocated:.2f} GB")
    
    previous_average_loss = float('inf')
    
    # Step 2: Multi-pass training over epochs
    print(f"Starting training for up to {num_epochs} epochs on {len(L)} logs...")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        total_loss = 0.0
        num_logs = 0
        
        # Loop over each log
        for i in range(len(L)):
            # Print progress every 10 logs
            if (i + 1) % 10 == 0 or (i + 1) == len(L):
                print(f"  Processing log {i+1}/{len(L)}...", end='\r', flush=True)
            log_steps = L[i]
            agent_ids_i = agent_ids[i]
            gt_step = mistake_step[i]
            gt_agent = mistake_agent[i]
            
            # Validate log_steps and agent_ids_i match
            if len(log_steps) != len(agent_ids_i):
                continue  # Skip if mismatch
            
            # ---- Forward pass: all discriminators on this log ----
            p_steps_list = []
            a_list = []
            for D_k in discriminators:
                p_k_step, _, a_k = D_k.forward_with_abstention(log_steps)
                p_steps_list.append(p_k_step)
                a_list.append(a_k)
            
            # ---- BT consensus over these K distributions ----
            # Detach inputs to BT_consensus since it does its own optimization
            # and we use the consensus as a target, not for backprop
            p_steps_list_detached = [p_k.detach() for p_k in p_steps_list]
            p_group = BT_consensus(p_steps_list_detached)
            
            # ---- Agent-level group distribution ----
            P_group_agent = compute_agent_distribution(p_group, agent_ids_i)
            
            # ---- Each discriminator gets its own reward and update ----
            for k, D_k in enumerate(discriminators):
                optimizer_k = optimizers[k]
                p_k_step = p_steps_list[k]
                a_k = a_list[k]
                
                # Compute agent-level distribution for discriminator k
                P_k_agent = compute_agent_distribution(p_k_step, agent_ids_i)
                
                # Compute rewards
                R_sup = supervised_reward(p_k_step, P_k_agent, gt_step, gt_agent, hyperparams)
                R_unsup = unsupervised_reward(
                    p_k_step, P_k_agent, p_group, P_group_agent, hyperparams
                )
                R_stab = entropy_and_abstention_reward(p_k_step, P_k_agent, a_k, hyperparams)
                
                # Mix supervised and unsupervised rewards if labels available
                if gt_step is not None or gt_agent is not None:
                    R_core = lambda_mix * R_sup + (1 - lambda_mix) * R_unsup
                else:
                    R_core = R_unsup
                
                # Total reward (we want to maximize R_total → minimize loss = -R_total)
                R_total = R_core + R_stab
                loss = -R_total
                
                # Update discriminator k
                optimizer_k.zero_grad()
                loss.backward()
                optimizer_k.step()
                
                # Accumulate loss for early stopping
                total_loss += loss.item()
                num_logs += 1
        
        # ---- Compute average loss and early stopping ----
        average_loss = total_loss / max(num_logs, 1)
        
        # Print epoch summary
        print(f"\n  Epoch {epoch} complete: Average loss = {average_loss:.6f}")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Check for early stopping (loss has stabilized)
        if abs(previous_average_loss - average_loss) < loss_tolerance:
            # Loss has stabilized; stop early
            print(f"  Early stopping: loss change ({abs(previous_average_loss - average_loss):.8f}) < tolerance ({loss_tolerance})")
            break
        
        previous_average_loss = average_loss
    
    # Put discriminators in eval mode before returning
    for D_k in discriminators:
        D_k.eval()
    
    return discriminators


def consensus_predict_failure(log_steps: List[Any],
                              agent_ids_i: List[str],
                              discriminators: List[Discriminator]) -> Tuple[int, str]:
    """
    Predict failure step and agent using consensus-trained discriminators.
    
    Uses BT consensus to ensemble predictions from all discriminators.
    
    Args:
        log_steps: Log sequence (list of steps) for a single example
        agent_ids_i: List of agent IDs corresponding to each step in the log
        discriminators: List of K trained Discriminator instances
    
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
    
    # Get predictions from all discriminators
    all_p = [D_k.predict_step_distribution(log_steps) for D_k in discriminators]
    
    # Combine using BT consensus
    p_group = BT_consensus(all_p)
    
    # Find argmax
    t_hat = torch.argmax(p_group).item()
    
    # Validate t_hat is within bounds
    if t_hat < 0 or t_hat >= num_steps:
        raise ValueError(f"Predicted step index {t_hat} is out of bounds [0, {num_steps})")
    
    # Map to agent ID
    i_hat = agent_ids_i[t_hat]
    
    return t_hat, i_hat


def consensus_predict_all(logs: List[List[Any]],
                         agent_ids_list: List[List[str]],
                         discriminators: List[Discriminator]) -> List[Tuple[int, str]]:
    """
    Predict failures for all logs using consensus-trained discriminators.
    
    Helper function for batch inference/testing.
    
    Args:
        logs: List of log sequences (each is a list of steps)
        agent_ids_list: List of agent ID lists, one per log
        discriminators: List of K trained Discriminator instances
    
    Returns:
        List of (t_hat, i_hat) tuples, one per log
    """
    if len(logs) != len(agent_ids_list):
        raise ValueError(f"logs length ({len(logs)}) must match "
                        f"agent_ids_list length ({len(agent_ids_list)})")
    
    results = []
    
    for i in range(len(logs)):
        t_hat, i_hat = consensus_predict_failure(
            logs[i],
            agent_ids_list[i],
            discriminators
        )
        results.append((t_hat, i_hat))
    
    return results

