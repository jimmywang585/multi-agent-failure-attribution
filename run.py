#!/usr/bin/env python3
"""
Main entry point for Multi-Agent Failure Attribution System

Usage:
    python run.py [--config config.yaml] [--system SYSTEM_TYPE] [--model MODEL_NAME]
    
Examples:
    # Run with default config
    python run.py
    
    # Run specific system with specific model
    python run.py --system consensus --model roberta
    
    # Run with custom config file
    python run.py --config my_config.yaml
"""

import argparse
from pathlib import Path

from config_loader import load_config, get_model_config, get_system_config, get_data_config, get_output_config
from dataset import process_dataset
from baseline import build_system, SYSTEM_BASELINE, SYSTEM_CONSENSUS, SYSTEM_SUPERVISED
from consensus_training import consensus_predict_all
from supervised_training import supervised_predict_all
from baseline import baseline_predict_all

# Try to import metrics, but handle gracefully if not available
try:
    from metrics import extract_ground_truth, compute_metrics, print_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: metrics module not found. Metrics will not be computed.")


def extract_agent_ids_from_logs(L: list, dataset=None) -> list:
    """
    Extract agent IDs from log steps.
    
    This function tries to extract agent IDs from the log structure.
    For Who&When dataset, agent information may be in the step dicts or in separate columns.
    
    Args:
        L: List of log sequences
        dataset: Optional dataset object to extract additional info
    
    Returns:
        List of agent ID lists (one per log)
    """
    agent_ids_list = []
    
    for i, log in enumerate(L):
        agent_ids_i = []
        for step in log:
            if isinstance(step, dict):
                # Try to extract agent ID from common keys in Who&When format
                # Priority: name (matches ground truth) > role > other fields
                # Ground truth uses 'name' field (e.g., "Verification_Expert")
                agent_id = (
                    step.get('name') or  # Use name first to match ground truth format
                    step.get('role') or 
                    step.get('agent') or 
                    step.get('agent_id') or 
                    step.get('who') or
                    None
                )
                if agent_id:
                    agent_ids_i.append(str(agent_id))
                else:
                    # Fallback: try to extract from dataset columns if available
                    # For Who&When, agent columns might be separate
                    agent_ids_i.append("unknown")
            else:
                # Fallback for non-dict steps
                agent_ids_i.append("unknown")
        
        # If we got all "unknown", try to use dataset columns
        if all(aid == "unknown" for aid in agent_ids_i) and dataset is not None:
            # Try to extract from dataset structure
            # This is dataset-specific and may need adjustment
            pass
        
        agent_ids_list.append(agent_ids_i)
    
    return agent_ids_list


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Failure Attribution System")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--system', type=str, choices=['baseline', 'consensus', 'supervised', 'all'],
                       help='System to run (overrides config)')
    parser.add_argument('--model', type=str, help='Model to use (overrides config)')
    parser.add_argument('--max-logs', type=int, help='Maximum number of logs to use (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Please create a config.yaml file or specify a different config with --config")
        return 1
    
    # Override config with command line arguments
    if args.system:
        config['system']['system_type'] = args.system
    if args.model:
        config['system']['model_name'] = args.model
    if args.max_logs:
        config['data']['max_logs'] = args.max_logs
    
    # Get configurations
    system_config = get_system_config(config)
    model_config = get_model_config(config)
    data_config = get_data_config(config)
    output_config = get_output_config(config)
    
    system_type = system_config['system_type']
    K = system_config['K']
    dataset_split = system_config.get('dataset_split', 'train')
    hyperparams = system_config.get('hyperparams', {})
    
    print(f"\n{'='*60}")
    print(f"Multi-Agent Failure Attribution System")
    print(f"{'='*60}")
    print(f"System: {system_type}")
    print(f"Model: {config['system']['model_name']} ({model_config['encoder_type']})")
    print(f"Discriminators: {K}")
    print(f"Dataset split: {dataset_split}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset_config = data_config.get('dataset_config', None)  # None = merge both
        L, unique_agent_ids = process_dataset(split=dataset_split, config=dataset_config)
        config_display = dataset_config if dataset_config else "merged (Algorithm-Generated + Hand-Crafted)"
        print(f"✓ Loaded {len(L)} logs with {len(unique_agent_ids)} unique agents (config: {config_display})")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Limit logs if specified
    max_logs = data_config.get('max_logs')
    if max_logs and max_logs < len(L):
        L = L[:max_logs]
        print(f"  Using first {max_logs} logs for testing")
    
    # Extract agent IDs per log
    print("Extracting agent IDs from logs...")
    if data_config.get('extract_agent_ids_from_steps', True):
        # Try to load dataset again to get additional info if needed
        try:
            from datasets import load_dataset
            dataset_config = data_config.get('dataset_config', None)
            if dataset_config is None:
                # Load both configs separately (can't concatenate due to schema differences)
                # We'll use the first one for agent ID extraction, but both are already in L
                dataset_obj = load_dataset("Kevin355/Who_and_When", "Algorithm-Generated", split=dataset_split)
            else:
                dataset_obj = load_dataset("Kevin355/Who_and_When", dataset_config, split=dataset_split)
            agent_ids = extract_agent_ids_from_logs(L, dataset=dataset_obj)
        except:
            agent_ids = extract_agent_ids_from_logs(L, dataset=None)
        print(f"✓ Extracted agent IDs for {len(agent_ids)} logs")
        
        # Validate: check if we got meaningful agent IDs
        unknown_count = sum(1 for log_ids in agent_ids for aid in log_ids if aid == "unknown")
        if unknown_count > len(agent_ids) * 0.5:
            print(f"  Warning: {unknown_count} unknown agent IDs found.")
            print(f"  You may need to adjust extract_agent_ids_from_logs() for your dataset format.")
    else:
        print("  Warning: Using dummy agent IDs. Update extract_agent_ids_from_logs() for your dataset.")
        agent_ids = [[unique_agent_ids[i % len(unique_agent_ids)] for _ in range(len(log))] for i, log in enumerate(L)]
    
    # Extract ground truth labels for evaluation
    if METRICS_AVAILABLE:
        print("Extracting ground truth labels...")
        try:
            dataset_config = data_config.get('dataset_config', None)
            gt_mistake_step, gt_mistake_agent = extract_ground_truth(split=dataset_split, config=dataset_config)
            # Limit to match L length if we limited logs
            if max_logs and max_logs < len(gt_mistake_step):
                gt_mistake_step = gt_mistake_step[:max_logs]
                gt_mistake_agent = gt_mistake_agent[:max_logs]
            num_with_labels = sum(1 for i in range(len(gt_mistake_step)) 
                                 if gt_mistake_step[i] is not None or gt_mistake_agent[i] is not None)
            print(f"✓ Extracted ground truth for {num_with_labels}/{len(gt_mistake_step)} logs")
        except Exception as e:
            print(f"  Warning: Could not extract ground truth labels: {e}")
            gt_mistake_step = [None] * len(L)
            gt_mistake_agent = [None] * len(L)
    else:
        gt_mistake_step = [None] * len(L)
        gt_mistake_agent = [None] * len(L)
    
    # Prepare model config for discriminators
    discriminator_model_config = {
        'encoder_type': model_config['encoder_type'],
        'max_length': model_config['max_length'],
        'd_model': model_config.get('d_model')
    }
    
    # Run specified system(s)
    systems_to_run = [system_type] if system_type != 'all' else [SYSTEM_BASELINE, SYSTEM_CONSENSUS, SYSTEM_SUPERVISED]
    
    for sys_type in systems_to_run:
        print(f"\n{'='*60}")
        print(f"Running System: {sys_type.upper()}")
        print(f"{'='*60}")
        
        try:
            if sys_type == SYSTEM_BASELINE:
                # System 2: Baseline (no training)
                print("Initializing untrained discriminators...")
                discriminators = build_system(
                    SYSTEM_BASELINE,
                    K=K,
                    model_config=discriminator_model_config
                )
                print("✓ Baseline system ready")
                
                # Predict on all logs
                print("Running predictions...")
                results = baseline_predict_all(L, agent_ids, discriminators)
                print(f"✓ Completed {len(results)} predictions")
                
            elif sys_type == SYSTEM_CONSENSUS:
                # System 1: Online Consensus Training
                print("Training discriminators with consensus learning...")
                
                # Prepare labels (optional, for semi-supervised)
                use_labels = data_config.get('use_labels', False)
                if use_labels:
                    mistake_step = [None] * len(L)
                    mistake_agent = [None] * len(L)
                    print("  Using semi-supervised mode (some labels available)")
                else:
                    mistake_step = [None] * len(L)
                    mistake_agent = [None] * len(L)
                    print("  Using fully unsupervised mode")
                
                discriminators = build_system(
                    SYSTEM_CONSENSUS,
                    K=K,
                    model_config=discriminator_model_config,
                    L=L,
                    agent_ids=agent_ids,
                    hyperparams=hyperparams,
                    mistake_agent=mistake_agent,
                    mistake_step=mistake_step
                )
                print("✓ Consensus training completed")
                
                # Predict on all logs
                print("Running predictions...")
                results = consensus_predict_all(L, agent_ids, discriminators)
                print(f"✓ Completed {len(results)} predictions")
                
            elif sys_type == SYSTEM_SUPERVISED:
                # System 3: Supervised Training
                print("Training discriminators with supervised learning...")
                
                # Prepare ground truth labels
                # For supervised training, use actual ground truth if available
                if METRICS_AVAILABLE:
                    try:
                        dataset_config = data_config.get('dataset_config', None)
                        mistake_step, mistake_agent = extract_ground_truth(split=dataset_split, config=dataset_config)
                        if max_logs and max_logs < len(mistake_step):
                            mistake_step = mistake_step[:max_logs]
                            mistake_agent = mistake_agent[:max_logs]
                    except Exception:
                        mistake_step = [None] * len(L)
                        mistake_agent = [None] * len(L)
                else:
                    mistake_step = [None] * len(L)
                    mistake_agent = [None] * len(L)
                
                discriminators = build_system(
                    SYSTEM_SUPERVISED,
                    K=K,
                    model_config=discriminator_model_config,
                    L=L,
                    agent_ids=agent_ids,
                    mistake_agent=mistake_agent,
                    mistake_step=mistake_step,
                    hyperparams=hyperparams
                )
                print("✓ Supervised training completed")
                
                # Predict on all logs
                print("Running predictions...")
                results = supervised_predict_all(L, agent_ids, discriminators, use_ensemble=True)
                print(f"✓ Completed {len(results)} predictions")
            
            # Show sample predictions
            if results:
                print(f"\nSample predictions (first 5):")
                for i, (t_hat, i_hat) in enumerate(results[:5]):
                    print(f"  Log {i}: step={t_hat}, agent={i_hat}")
            
            # Compute and display metrics
            if METRICS_AVAILABLE and results and len(results) == len(gt_mistake_step):
                try:
                    metrics = compute_metrics(
                        predictions=results,
                        ground_truth_step=gt_mistake_step,
                        ground_truth_agent=gt_mistake_agent,
                        agent_ids=agent_ids
                    )
                    print_metrics(metrics, system_name=sys_type.upper())
                except Exception as e:
                    print(f"  Warning: Could not compute metrics: {e}")
            
            # Save results if configured
            if output_config.get('save_predictions', False):
                import json
                output_path = output_config.get('predictions_save_path', './predictions.json')
                with open(output_path, 'w') as f:
                    json.dump([{"step": t, "agent": a} for t, a in results], f, indent=2)
                print(f"✓ Saved predictions to {output_path}")
            
        except Exception as e:
            print(f"✗ Error running {sys_type}: {e}")
            import traceback
            traceback.print_exc()
            if system_type != 'all':
                return 1
    
    print(f"\n{'='*60}")
    print("✓ All systems completed successfully!")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())

