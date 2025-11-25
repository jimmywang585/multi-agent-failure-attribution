"""
Metrics computation for multi-agent failure attribution system.

Computes accuracy metrics comparing predictions against ground truth labels.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def extract_ground_truth(split: str = "train", config: Optional[str] = None) -> Tuple[List[Optional[int]], List[Optional[str]]]:
    """
    Extract ground truth labels from the Who&When dataset.
    
    Args:
        split: Dataset split to load (default: "train")
        config: Dataset config name - "Algorithm-Generated", "Hand-Crafted", or None to merge both
        
    Returns:
        mistake_step: List of ground truth step indices (one per log, can be None)
        mistake_agent: List of ground truth agent IDs (one per log, can be None)
    """
    from datasets import load_dataset
    
    mistake_step = []
    mistake_agent = []
    
    if config is None:
        # Process both configs
        configs = ["Algorithm-Generated", "Hand-Crafted"]
    else:
        configs = [config]
    
    for cfg in configs:
        try:
            dataset = load_dataset("Kevin355/Who_and_When", cfg, split=split)
            
            for example in dataset:
                # Extract mistake_step (may be string or int)
                step_val = example.get('mistake_step', None)
                if step_val is not None and step_val != "":
                    try:
                        # Try to convert to int if it's a string
                        if isinstance(step_val, str):
                            # Handle string representations like "0", "1", etc.
                            step_val = int(step_val.strip())
                        elif isinstance(step_val, (int, float)):
                            step_val = int(step_val)
                        else:
                            step_val = None
                    except (ValueError, TypeError, AttributeError):
                        step_val = None
                else:
                    step_val = None
                mistake_step.append(step_val)
                
                # Extract mistake_agent
                agent_val = example.get('mistake_agent', None)
                if agent_val is not None and agent_val != "":
                    mistake_agent.append(str(agent_val).strip())
                else:
                    mistake_agent.append(None)
        except Exception as e:
            # If one config fails, continue with the other
            print(f"Warning: Could not load config {cfg}: {e}")
            continue
    
    return mistake_step, mistake_agent


def compute_metrics(
    predictions: List[Tuple[int, str]],
    ground_truth_step: List[Optional[int]],
    ground_truth_agent: List[Optional[str]],
    agent_ids: List[List[str]]
) -> Dict[str, Any]:
    """
    Compute accuracy metrics comparing predictions to ground truth.
    
    Args:
        predictions: List of (predicted_step, predicted_agent) tuples
        ground_truth_step: List of ground truth step indices (can be None)
        ground_truth_agent: List of ground truth agent IDs (can be None)
        agent_ids: List of agent ID lists (one per log, for step-to-agent mapping)
    
    Returns:
        Dictionary containing various accuracy metrics
    """
    if len(predictions) != len(ground_truth_step) or len(predictions) != len(ground_truth_agent):
        raise ValueError(f"Predictions length ({len(predictions)}) must match ground truth length "
                        f"({len(ground_truth_step)}, {len(ground_truth_agent)})")
    
    # Filter to only examples with ground truth
    valid_indices = []
    for i in range(len(predictions)):
        if ground_truth_step[i] is not None or ground_truth_agent[i] is not None:
            valid_indices.append(i)
    
    if not valid_indices:
        return {
            "step_accuracy": None,
            "agent_accuracy": None,
            "exact_match": None,
            "num_valid": 0,
            "num_total": len(predictions)
        }
    
    # Step-level accuracy
    step_correct = 0
    step_total = 0
    for i in valid_indices:
        if ground_truth_step[i] is not None:
            step_total += 1
            if predictions[i][0] == ground_truth_step[i]:
                step_correct += 1
    
    step_accuracy = step_correct / step_total if step_total > 0 else None
    
    # Agent-level accuracy
    agent_correct = 0
    agent_total = 0
    for i in valid_indices:
        if ground_truth_agent[i] is not None:
            agent_total += 1
            # Compare predicted agent (normalize strings for comparison)
            pred_agent = str(predictions[i][1]).strip()
            gt_agent = str(ground_truth_agent[i]).strip()
            if pred_agent == gt_agent:
                agent_correct += 1
    
    agent_accuracy = agent_correct / agent_total if agent_total > 0 else None
    
    # Exact match (both step and agent correct)
    exact_correct = 0
    exact_total = 0
    for i in valid_indices:
        if ground_truth_step[i] is not None and ground_truth_agent[i] is not None:
            exact_total += 1
            pred_agent = str(predictions[i][1]).strip()
            gt_agent = str(ground_truth_agent[i]).strip()
            if (predictions[i][0] == ground_truth_step[i] and 
                pred_agent == gt_agent):
                exact_correct += 1
    
    exact_match = exact_correct / exact_total if exact_total > 0 else None
    
    # Step distance (how far off the predicted step is)
    step_distances = []
    for i in valid_indices:
        if ground_truth_step[i] is not None:
            distance = abs(predictions[i][0] - ground_truth_step[i])
            step_distances.append(distance)
    
    avg_step_distance = np.mean(step_distances) if step_distances else None
    median_step_distance = np.median(step_distances) if step_distances else None
    
    return {
        "step_accuracy": step_accuracy,
        "agent_accuracy": agent_accuracy,
        "exact_match": exact_match,
        "avg_step_distance": avg_step_distance,
        "median_step_distance": median_step_distance,
        "num_valid": len(valid_indices),
        "num_total": len(predictions),
        "step_correct": step_correct,
        "step_total": step_total,
        "agent_correct": agent_correct,
        "agent_total": agent_total,
        "exact_correct": exact_correct,
        "exact_total": exact_total
    }


def print_metrics(metrics: Dict[str, Any], system_name: str = "System"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary returned by compute_metrics
        system_name: Name of the system (for display)
    """
    print(f"\n{'='*60}")
    print(f"Metrics for {system_name}")
    print(f"{'='*60}")
    
    if metrics["num_valid"] == 0:
        print("⚠ No ground truth labels available for evaluation")
        return
    
    print(f"Valid examples: {metrics['num_valid']} / {metrics['num_total']}")
    print()
    
    if metrics["step_accuracy"] is not None:
        print(f"Step Accuracy: {metrics['step_accuracy']:.4f} ({metrics['step_correct']}/{metrics['step_total']})")
    
    if metrics["agent_accuracy"] is not None:
        print(f"Agent Accuracy: {metrics['agent_accuracy']:.4f} ({metrics['agent_correct']}/{metrics['agent_total']})")
    
    if metrics["exact_match"] is not None:
        print(f"Exact Match (Step + Agent): {metrics['exact_match']:.4f} ({metrics['exact_correct']}/{metrics['exact_total']})")
    
    if metrics["avg_step_distance"] is not None:
        print(f"Average Step Distance: {metrics['avg_step_distance']:.2f}")
        print(f"Median Step Distance: {metrics['median_step_distance']:.2f}")
    
    print(f"{'='*60}\n")

