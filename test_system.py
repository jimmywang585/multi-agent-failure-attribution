"""
Test script for multi-agent failure attribution system

Tests all three systems:
- System 2: Baseline (no training)
- System 1: Online Consensus Training (semi-supervised)
- System 3: Supervised Training

Usage:
    python test_system.py
"""

import torch
from typing import List, Optional

from dataset import process_dataset
from baseline import build_system, SYSTEM_BASELINE, SYSTEM_CONSENSUS, SYSTEM_SUPERVISED
from consensus_training import consensus_predict_failure, consensus_predict_all
from supervised_training import supervised_predict_failure, supervised_predict_all
from baseline import baseline_predict_failure, baseline_predict_all


def test_dataset_loading():
    """Test Step 1: Dataset loading"""
    print("=" * 60)
    print("Testing Step 1: Dataset Loading")
    print("=" * 60)
    
    try:
        L, agent_ids = process_dataset(split="train")
        print(f"✓ Successfully loaded dataset")
        print(f"  - Number of logs: {len(L)}")
        print(f"  - Number of unique agents: {len(agent_ids)}")
        print(f"  - Sample log length: {len(L[0]) if L else 0} steps")
        print(f"  - Agent IDs: {agent_ids[:5]}..." if len(agent_ids) > 5 else f"  - Agent IDs: {agent_ids}")
        
        # Show sample log structure
        if L and len(L) > 0:
            print(f"\n  Sample log (first 2 steps):")
            for i, step in enumerate(L[0][:2]):
                print(f"    Step {i}: {str(step)[:100]}...")
        
        return L, agent_ids
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        raise


def test_system_baseline(L: List, agent_ids: List[List[str]], K: int = 3):
    """Test System 2: Baseline (no training)"""
    print("\n" + "=" * 60)
    print("Testing System 2: Baseline (No Training)")
    print("=" * 60)
    
    try:
        # Initialize baseline system
        model_config = {"d_model": 64, "vocab_size": 1000}  # Smaller for testing
        discriminators = build_system(
            SYSTEM_BASELINE,
            K=K,
            model_config=model_config
        )
        print(f"✓ Initialized {K} untrained discriminators")
        
        # Test prediction on a single log
        if len(L) > 0:
            test_log = L[0]
            test_agent_ids = agent_ids[0]
            
            t_hat, i_hat = baseline_predict_failure(
                test_log,
                test_agent_ids,
                discriminators
            )
            print(f"✓ Single prediction successful")
            print(f"  - Predicted step: {t_hat}")
            print(f"  - Predicted agent: {i_hat}")
            
            # Test batch prediction
            test_logs = L[:3]  # Test on first 3 logs
            test_agent_ids_list = agent_ids[:3]
            
            results = baseline_predict_all(
                test_logs,
                test_agent_ids_list,
                discriminators
            )
            print(f"✓ Batch prediction successful ({len(results)} predictions)")
            for i, (t, agent) in enumerate(results):
                print(f"  - Log {i}: step={t}, agent={agent}")
        
        return discriminators
    except Exception as e:
        print(f"✗ Error in baseline system: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_system_consensus(L: List, agent_ids: List[List[str]], K: int = 3):
    """Test System 1: Online Consensus Training"""
    print("\n" + "=" * 60)
    print("Testing System 1: Online Consensus Training")
    print("=" * 60)
    
    try:
        # Use a small subset for faster testing
        test_size = min(10, len(L))
        L_test = L[:test_size]
        agent_ids_test = agent_ids[:test_size]
        
        # Optionally provide some labels (semi-supervised)
        # For testing, we'll use None (fully unsupervised)
        mistake_step = [None] * test_size
        mistake_agent = [None] * test_size
        
        # Or provide some labels for semi-supervised testing:
        # mistake_step = [0 if i % 2 == 0 else None for i in range(test_size)]
        # mistake_agent = [agent_ids_test[i][0] if i % 2 == 0 else None for i in range(test_size)]
        
        model_config = {"d_model": 64, "vocab_size": 1000}
        hyperparams = {
            "lr": 0.001,
            "num_epochs": 2,  # Small for testing
            "loss_tolerance": 1e-4,
            "alpha_s": 1.0,
            "alpha_a": 1.0,
            "beta_pair": 1.0,
            "beta_s": 1.0,
            "beta_a": 1.0,
            "gamma_s": 0.01,
            "gamma_a": 0.01,
            "lambda_abs": 0.1,
            "lambda_mix": 0.5
        }
        
        print(f"Training on {test_size} logs with {K} discriminators...")
        discriminators = build_system(
            SYSTEM_CONSENSUS,
            K=K,
            model_config=model_config,
            L=L_test,
            agent_ids=agent_ids_test,
            hyperparams=hyperparams,
            mistake_agent=mistake_agent,
            mistake_step=mistake_step
        )
        print(f"✓ Training completed successfully")
        
        # Test prediction
        if len(L_test) > 0:
            test_log = L_test[0]
            test_agent_ids = agent_ids_test[0]
            
            t_hat, i_hat = consensus_predict_failure(
                test_log,
                test_agent_ids,
                discriminators
            )
            print(f"✓ Prediction successful")
            print(f"  - Predicted step: {t_hat}")
            print(f"  - Predicted agent: {i_hat}")
        
        return discriminators
    except Exception as e:
        print(f"✗ Error in consensus training: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_system_supervised(L: List, agent_ids: List[List[str]], K: int = 3):
    """Test System 3: Supervised Training"""
    print("\n" + "=" * 60)
    print("Testing System 3: Supervised Training")
    print("=" * 60)
    
    try:
        # Use a small subset for faster testing
        test_size = min(10, len(L))
        L_test = L[:test_size]
        agent_ids_test = agent_ids[:test_size]
        
        # Create dummy ground truth labels for testing
        # In real usage, these would come from the dataset
        mistake_step = [0 if i < len(L_test[i]) else None for i in range(test_size)]
        mistake_agent = [agent_ids_test[i][0] if agent_ids_test[i] else None for i in range(test_size)]
        
        model_config = {"d_model": 64, "vocab_size": 1000}
        hyperparams = {
            "lr": 0.001,
            "use_agent_loss": True
        }
        
        print(f"Training on {test_size} logs with {K} discriminators...")
        discriminators = build_system(
            SYSTEM_SUPERVISED,
            K=K,
            model_config=model_config,
            L=L_test,
            agent_ids=agent_ids_test,
            mistake_agent=mistake_agent,
            mistake_step=mistake_step,
            hyperparams=hyperparams
        )
        print(f"✓ Training completed successfully")
        
        # Test prediction
        if len(L_test) > 0:
            test_log = L_test[0]
            test_agent_ids = agent_ids_test[0]
            
            t_hat, i_hat = supervised_predict_failure(
                test_log,
                test_agent_ids,
                discriminators,
                use_ensemble=True
            )
            print(f"✓ Prediction successful")
            print(f"  - Predicted step: {t_hat}")
            print(f"  - Predicted agent: {i_hat}")
        
        return discriminators
    except Exception as e:
        print(f"✗ Error in supervised training: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_components():
    """Test individual components"""
    print("\n" + "=" * 60)
    print("Testing Individual Components")
    print("=" * 60)
    
    # Test BT consensus
    try:
        from bt_consensus import BT_consensus
        import torch
        
        # Create dummy distributions
        p1 = torch.softmax(torch.randn(5), dim=0)
        p2 = torch.softmax(torch.randn(5), dim=0)
        p3 = torch.softmax(torch.randn(5), dim=0)
        
        p_group = BT_consensus([p1, p2, p3])
        print(f"✓ BT_consensus works: output shape {p_group.shape}, sum={p_group.sum().item():.4f}")
    except Exception as e:
        print(f"✗ BT_consensus error: {e}")
    
    # Test consensus loss
    try:
        from losses import consensus_loss
        
        p_k = torch.softmax(torch.randn(5), dim=0)
        p_group = torch.softmax(torch.randn(5), dim=0)
        hyperparams = {"beta_cons": 1.0, "alpha_entropy": 0.01}
        
        loss = consensus_loss(p_k, p_group, hyperparams)
        print(f"✓ consensus_loss works: loss={loss.item():.4f}")
    except Exception as e:
        print(f"✗ consensus_loss error: {e}")
    
    # Test rewards
    try:
        from rewards import (
            compute_agent_distribution,
            supervised_reward,
            unsupervised_reward,
            entropy_and_abstention_reward
        )
        import torch
        
        p_step = torch.softmax(torch.randn(5), dim=0)
        agent_ids = ["agent1", "agent2", "agent1", "agent3", "agent2"]
        
        P_agent = compute_agent_distribution(p_step, agent_ids)
        print(f"✓ compute_agent_distribution works: {len(P_agent)} agents")
        
        hyperparams = {"alpha_s": 1.0, "alpha_a": 1.0}
        R_sup = supervised_reward(p_step, P_agent, gt_step=0, gt_agent="agent1", hyperparams=hyperparams)
        print(f"✓ supervised_reward works: R_sup={R_sup.item():.4f}")
        
        p_group = torch.softmax(torch.randn(5), dim=0)
        P_group = compute_agent_distribution(p_group, agent_ids)
        hyperparams = {"beta_pair": 1.0, "beta_s": 1.0, "beta_a": 1.0}
        R_unsup = unsupervised_reward(p_step, P_agent, p_group, P_group, hyperparams)
        print(f"✓ unsupervised_reward works: R_unsup={R_unsup.item():.4f}")
        
        a = torch.tensor(0.1)
        hyperparams = {"gamma_s": 0.01, "gamma_a": 0.01, "lambda_abs": 0.1}
        R_stab = entropy_and_abstention_reward(p_step, P_agent, a, hyperparams)
        print(f"✓ entropy_and_abstention_reward works: R_stab={R_stab.item():.4f}")
    except Exception as e:
        print(f"✗ Rewards error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("Multi-Agent Failure Attribution System - Test Suite")
    print("=" * 60)
    
    # Test components first (fast)
    test_components()
    
    # Test dataset loading
    try:
        L, agent_ids = test_dataset_loading()
        
        # Test each system
        test_system_baseline(L, agent_ids, K=3)
        test_system_consensus(L, agent_ids, K=3)
        test_system_supervised(L, agent_ids, K=3)
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

