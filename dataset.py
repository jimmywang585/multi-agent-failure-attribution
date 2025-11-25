"""
Step 1: process_dataset()

Load Who&When logs from Hugging Face and produce L (list of log sequences) and agent_ids.
For now, ignore ground truth labels.

Dataset: https://huggingface.co/datasets/Kevin355/Who_and_When
"""

from typing import List, Tuple, Any
from datasets import load_dataset


def process_dataset(split: str = "train", config: str = None) -> Tuple[List[Any], List[str]]:
    """
    Load Who&When logs from Hugging Face and produce L and agent_ids.
    
    Args:
        split: Dataset split to load (default: "train")
        config: Dataset config name - "Algorithm-Generated", "Hand-Crafted", or None to merge both (default: None)
        
    Returns:
        L: List of log sequences (each sequence is a list of steps from the 'history' column)
        agent_ids: List of unique agent identifiers found in the dataset columns
    """
    if config is None:
        # Load both configs separately and merge the results
        # (Can't concatenate directly due to schema differences)
        L_alg, agent_ids_alg = _process_single_config("Algorithm-Generated", split)
        L_hand, agent_ids_hand = _process_single_config("Hand-Crafted", split)
        
        # Merge the results
        L = L_alg + L_hand
        agent_ids_set = set(agent_ids_alg) | set(agent_ids_hand)
        agent_ids = sorted(list(agent_ids_set))
    else:
        L, agent_ids = _process_single_config(config, split)
    
    if not L:
        raise ValueError("No log sequences found in the dataset")
    
    if not agent_ids:
        raise ValueError("No agent IDs found in the dataset")
    
    return L, agent_ids


def _process_single_config(config: str, split: str) -> Tuple[List[Any], List[str]]:
    """
    Process a single dataset config.
    
    Args:
        config: Dataset config name - "Algorithm-Generated" or "Hand-Crafted"
        split: Dataset split to load
        
    Returns:
        L: List of log sequences
        agent_ids: List of unique agent identifiers
    """
    dataset = load_dataset("Kevin355/Who_and_When", config, split=split)
    
    L = []  # List of log sequences
    agent_ids_set = set()  # Use set to collect unique agent IDs
    
    # Metadata columns to exclude when identifying agents
    metadata_columns = {'is_correct', 'question', 'question_ID', 'ground_truth', 'history', 
                       'is_corrected', 'mistake_agent', 'mistake_step', 'mistake_reason', 
                       'mistake_type', 'groundtruth'}
    
    # Extract agent IDs from column names (exclude metadata columns)
    all_columns = set(dataset.column_names)
    agent_columns = all_columns - metadata_columns
    
    # Add agent IDs from column names
    for agent_id in agent_columns:
        agent_ids_set.add(agent_id)
    
    # Extract log sequences from the 'history' column
    for example in dataset:
        history = example.get('history', [])
        if history:
            L.append(history)
            
            for step in history:
                if isinstance(step, dict):
                    # Try different possible keys for agent ID
                    agent_id = (step.get('agent') or step.get('agent_id') or 
                               step.get('who') or step.get('agent_name') or 
                               step.get('name') or step.get('role'))
                    if agent_id:
                        agent_ids_set.add(str(agent_id))
    
    # Convert set to sorted list for consistent ordering
    agent_ids = sorted(list(agent_ids_set))
    
    return L, agent_ids

