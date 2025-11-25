"""
System 2: Baseline Inference (No Training)

Implements baseline failure attribution using untrained discriminators.
Uses BT consensus to combine predictions from multiple discriminators.

This module is structured to allow plugging in:
- System 1: Online Consensus Training
- System 3: Supervised Training
"""

import torch
from typing import List, Dict, Any, Tuple, Optional

from discriminator import Discriminator, initialize_discriminators
from bt_consensus import BT_consensus
from consensus_training import online_consensus_train
from supervised_training import supervised_train_one_round


# System type constants
SYSTEM_BASELINE = "baseline"
SYSTEM_CONSENSUS = "consensus"  # System 1 - Online Consensus Training
SYSTEM_SUPERVISED = "supervised"  # System 3 - Supervised Training


def baseline_initialize(K: int, model_config: Optional[Dict[str, Any]] = None) -> List[Discriminator]:
    """
    Initialize K untrained discriminators for baseline inference.
    
    Creates discriminators but does NOT train them or create optimizers.
    Simply returns the untrained discriminators.
    
    Args:
        K: Number of discriminators to create
        model_config: Configuration dictionary for discriminators
                     (see Discriminator.__init__ for details)
    
    Returns:
        List of K untrained Discriminator instances [D_1, ..., D_K]
    """
    # Use the existing initialization function
    discriminators = initialize_discriminators(K, model_config)
    
    # Set all discriminators to evaluation mode (no training)
    for discriminator in discriminators:
        discriminator.eval()
    
    return discriminators


def baseline_predict_failure(log_steps: List[Any],
                             agent_ids_i: List[str],
                             discriminators: List[Discriminator]) -> Tuple[int, str]:
    """
    Predict failure step and agent using baseline (untrained) discriminators.
    
    Process:
    1. Each discriminator predicts p_k_step (step probability distribution)
    2. Combine predictions using BT consensus to get p_group
    3. Find argmax to get predicted step index t_hat
    4. Map to agent ID: i_hat = agent_ids_i[t_hat]
    
    Args:
        log_steps: Log sequence (list of steps) for a single example
        agent_ids_i: List of agent IDs corresponding to each step in the log
                    (length should match number of steps in log_steps)
        discriminators: List of K untrained Discriminator instances
    
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
    
    # Step 1: Get predictions from each discriminator
    all_p = []  # List of p_k_step tensors
    
    for discriminator in discriminators:
        # Get step probability distribution from discriminator k
        p_k_step = discriminator.predict_step_distribution(log_steps)
        all_p.append(p_k_step)
    
    # Step 2: Combine predictions using BT consensus
    p_group = BT_consensus(all_p)  # [T] tensor
    
    # Step 3: Find argmax to get predicted step index
    t_hat = torch.argmax(p_group).item()  # Convert to Python int
    
    # Validate t_hat is within bounds
    if t_hat < 0 or t_hat >= num_steps:
        raise ValueError(f"Predicted step index {t_hat} is out of bounds [0, {num_steps})")
    
    # Step 4: Map to agent ID
    i_hat = agent_ids_i[t_hat]
    
    return t_hat, i_hat


def baseline_predict_all(logs: List[List[Any]],
                        agent_ids_list: List[List[str]],
                        discriminators: List[Discriminator]) -> List[Tuple[int, str]]:
    """
    Predict failures for all logs using baseline discriminators.
    
    Helper function for batch inference/testing.
    
    Args:
        logs: List of log sequences (each is a list of steps)
        agent_ids_list: List of agent ID lists, one per log
                       (agent_ids_list[i] corresponds to logs[i])
        discriminators: List of K untrained Discriminator instances
    
    Returns:
        List of (t_hat, i_hat) tuples, one per log
    """
    if len(logs) != len(agent_ids_list):
        raise ValueError(f"logs length ({len(logs)}) must match "
                        f"agent_ids_list length ({len(agent_ids_list)})")
    
    results = []
    
    for i in range(len(logs)):
        t_hat, i_hat = baseline_predict_failure(
            logs[i],
            agent_ids_list[i],
            discriminators
        )
        results.append((t_hat, i_hat))
    
    return results


def build_system(system_type: str,
                 K: int,
                 model_config: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Any:
    """
    System factory: Build and return the requested system.
    
    This function provides a unified interface for creating different systems:
    - System 2 (baseline): Untrained discriminators
    - System 1 (consensus): Online consensus training (no labels)
    - System 3 (supervised): Supervised training (with ground truth)
    
    Args:
        system_type: One of SYSTEM_BASELINE, SYSTEM_CONSENSUS, SYSTEM_SUPERVISED
        K: Number of discriminators
        model_config: Configuration for discriminators
        **kwargs: Additional arguments for specific systems:
            - SYSTEM_CONSENSUS: requires 'L', 'agent_ids', optional 'hyperparams'
            - SYSTEM_SUPERVISED: requires 'L', 'agent_ids', 'mistake_agent', 'mistake_step', optional 'hyperparams'
    
    Returns:
        System-specific return value:
        - SYSTEM_BASELINE: List of untrained Discriminator instances
        - SYSTEM_CONSENSUS: List of trained Discriminator instances (consensus-trained)
        - SYSTEM_SUPERVISED: List of trained Discriminator instances (supervised-trained)
    """
    if system_type == SYSTEM_BASELINE:
        return baseline_initialize(K, model_config)
    
    elif system_type == SYSTEM_CONSENSUS:
        # System 1: Online Consensus Training (semi-supervised)
        # Expects L, agent_ids, hyperparams in kwargs
        # Optionally accepts mistake_agent, mistake_step for semi-supervised training
        L = kwargs.get("L")
        agent_ids = kwargs.get("agent_ids")
        hyperparams = kwargs.get("hyperparams", None)
        mistake_agent = kwargs.get("mistake_agent", None)
        mistake_step = kwargs.get("mistake_step", None)
        
        if L is None or agent_ids is None:
            raise ValueError(f"SYSTEM_CONSENSUS requires 'L' and 'agent_ids' in kwargs")
        
        return online_consensus_train(
            L, agent_ids, K, model_config, hyperparams, mistake_agent, mistake_step
        )
    
    elif system_type == SYSTEM_SUPERVISED:
        # System 3: Supervised Training
        # Expects L, agent_ids, mistake_agent, mistake_step, hyperparams in kwargs
        L = kwargs.get("L")
        agent_ids = kwargs.get("agent_ids")
        mistake_agent = kwargs.get("mistake_agent")
        mistake_step = kwargs.get("mistake_step")
        hyperparams = kwargs.get("hyperparams", None)
        
        if L is None or agent_ids is None or mistake_agent is None or mistake_step is None:
            raise ValueError(f"SYSTEM_SUPERVISED requires 'L', 'agent_ids', 'mistake_agent', "
                           f"and 'mistake_step' in kwargs")
        
        return supervised_train_one_round(
            L, agent_ids, mistake_agent, mistake_step, K, model_config, hyperparams
        )
    
    else:
        raise ValueError(
            f"Unknown system type: {system_type}. "
            f"Must be one of: {SYSTEM_BASELINE}, {SYSTEM_CONSENSUS}, {SYSTEM_SUPERVISED}"
        )

