"""
Quick experiment to compare BT consensus vs simple averaging for failure attribution.
"""

import torch
from dataset import process_dataset
from baseline import baseline_initialize, baseline_predict_all
from supervised_training import supervised_train_one_round, supervised_predict_all
from metrics import extract_ground_truth, compute_metrics, print_metrics
from config_loader import load_config, get_model_config, get_system_config, get_data_config

def simple_average_predictions(all_p):
    """Simple averaging of probability distributions."""
    p_stacked = torch.stack(all_p, dim=0)  # [K, T]
    return p_stacked.mean(dim=0)  # [T]

def predict_with_averaging(log_steps, agent_ids_i, discriminators):
    """Predict using simple averaging instead of BT consensus."""
    all_p = []
    for discriminator in discriminators:
        p_k_step = discriminator.predict_step_distribution(log_steps)
        all_p.append(p_k_step)
    
    p_group = simple_average_predictions(all_p)
    t_hat = torch.argmax(p_group).item()
    i_hat = agent_ids_i[t_hat]
    return t_hat, i_hat

def predict_all_with_averaging(logs, agent_ids_list, discriminators):
    """Predict all logs using simple averaging."""
    results = []
    for i in range(len(logs)):
        t_hat, i_hat = predict_with_averaging(logs[i], agent_ids_list[i], discriminators)
        results.append((t_hat, i_hat))
    return results

def main():
    print("="*60)
    print("BT Consensus vs Simple Averaging Comparison")
    print("="*60)
    
    # Load config
    config = load_config('config.yaml')
    model_config = get_model_config(config)
    system_config = get_system_config(config)
    data_config = get_data_config(config)
    
    K = system_config['K']
    dataset_split = system_config.get('dataset_split', 'train')
    dataset_config = data_config.get('dataset_config', None)
    
    # Load dataset
    print("\nLoading dataset...")
    L, unique_agent_ids = process_dataset(split=dataset_split, config=dataset_config)
    print(f"✓ Loaded {len(L)} logs")
    
    # Extract agent IDs
    from run import extract_agent_ids_from_logs
    agent_ids = extract_agent_ids_from_logs(L, dataset=None)
    
    # Extract ground truth
    gt_mistake_step, gt_mistake_agent = extract_ground_truth(split=dataset_split, config=dataset_config)
    
    # Prepare model config
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator_model_config = {
        'encoder_type': model_config['encoder_type'],
        'max_length': model_config['max_length'],
        'd_model': model_config.get('d_model'),
        'device': device
    }
    
    print(f"\nUsing {K} discriminators with {model_config['encoder_type']}")
    print(f"Device: {device}")
    
    # Test 1: Baseline with BT consensus
    print("\n" + "="*60)
    print("Test 1: Baseline with BT Consensus")
    print("="*60)
    discriminators_bt = baseline_initialize(K, discriminator_model_config)
    results_bt = baseline_predict_all(L, agent_ids, discriminators_bt, use_bt=True)
    
    metrics_bt = compute_metrics(
        predictions=results_bt,
        ground_truth_step=gt_mistake_step,
        ground_truth_agent=gt_mistake_agent,
        agent_ids=agent_ids
    )
    print_metrics(metrics_bt, system_name="BASELINE_BT")
    
    # Test 2: Baseline with Simple Averaging
    print("\n" + "="*60)
    print("Test 2: Baseline with Simple Averaging")
    print("="*60)
    discriminators_avg = baseline_initialize(K, discriminator_model_config)
    results_avg = baseline_predict_all(L, agent_ids, discriminators_avg, use_bt=False)
    
    metrics_avg = compute_metrics(
        predictions=results_avg,
        ground_truth_step=gt_mistake_step,
        ground_truth_agent=gt_mistake_agent,
        agent_ids=agent_ids
    )
    print_metrics(metrics_avg, system_name="BASELINE_AVG")
    
    # Test 3: Supervised with BT consensus
    print("\n" + "="*60)
    print("Test 3: Supervised with BT Consensus")
    print("="*60)
    hyperparams = system_config.get('hyperparams', {})
    discriminators_sup = supervised_train_one_round(
        L, agent_ids, gt_mistake_agent, gt_mistake_step, K, discriminator_model_config, hyperparams
    )
    
    # Set flag for BT consensus (default, but make explicit)
    import supervised_training
    supervised_training.supervised_predict_failure._use_simple_avg = False
    results_sup_bt = supervised_predict_all(L, agent_ids, discriminators_sup, use_ensemble=True)
    
    metrics_sup_bt = compute_metrics(
        predictions=results_sup_bt,
        ground_truth_step=gt_mistake_step,
        ground_truth_agent=gt_mistake_agent,
        agent_ids=agent_ids
    )
    print_metrics(metrics_sup_bt, system_name="SUPERVISED_BT")
    
    # Test 4: Supervised with Simple Averaging
    print("\n" + "="*60)
    print("Test 4: Supervised with Simple Averaging")
    print("="*60)
    # Use same trained discriminators, just change aggregation
    supervised_training.supervised_predict_failure._use_simple_avg = True
    results_sup_avg = supervised_predict_all(L, agent_ids, discriminators_sup, use_ensemble=True)
    
    metrics_sup_avg = compute_metrics(
        predictions=results_sup_avg,
        ground_truth_step=gt_mistake_step,
        ground_truth_agent=gt_mistake_agent,
        agent_ids=agent_ids
    )
    print_metrics(metrics_sup_avg, system_name="SUPERVISED_AVG")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'System':<25} {'Step Acc':<12} {'Agent Acc':<12} {'Exact Match':<12}")
    print("-" * 60)
    print(f"{'Baseline (BT)':<25} {metrics_bt['step_accuracy']:.4f}      {metrics_bt['agent_accuracy']:.4f}      {metrics_bt['exact_match']:.4f}")
    print(f"{'Baseline (Avg)':<25} {metrics_avg['step_accuracy']:.4f}      {metrics_avg['agent_accuracy']:.4f}      {metrics_avg['exact_match']:.4f}")
    print(f"{'Supervised (BT)':<25} {metrics_sup_bt['step_accuracy']:.4f}      {metrics_sup_bt['agent_accuracy']:.4f}      {metrics_sup_bt['exact_match']:.4f}")
    print(f"{'Supervised (Avg)':<25} {metrics_sup_avg['step_accuracy']:.4f}      {metrics_sup_avg['agent_accuracy']:.4f}      {metrics_sup_avg['exact_match']:.4f}")
    print("="*60)
    
    # Calculate differences
    step_diff_baseline = metrics_bt['step_accuracy'] - metrics_avg['step_accuracy']
    agent_diff_baseline = metrics_bt['agent_accuracy'] - metrics_avg['agent_accuracy']
    step_diff_supervised = metrics_sup_bt['step_accuracy'] - metrics_sup_avg['step_accuracy']
    agent_diff_supervised = metrics_sup_bt['agent_accuracy'] - metrics_sup_avg['agent_accuracy']
    
    print(f"\nBT vs Average Differences:")
    print(f"  Baseline - Step: {step_diff_baseline:+.4f}, Agent: {agent_diff_baseline:+.4f}")
    print(f"  Supervised - Step: {step_diff_supervised:+.4f}, Agent: {agent_diff_supervised:+.4f}")
    
    if step_diff_baseline > 0 or step_diff_supervised > 0:
        print("\n✓ BT consensus helps (positive difference)")
    elif step_diff_baseline < 0 or step_diff_supervised < 0:
        print("\n✗ Simple averaging performs better (negative difference)")
    else:
        print("\n≈ BT and averaging perform similarly")

if __name__ == "__main__":
    main()

