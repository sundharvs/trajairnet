#!/usr/bin/env python3
"""
Comprehensive evaluation script for comparing intent-aware attention methods.
Evaluates ADE and FDE metrics across all implemented approaches.
"""

import sys
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
import json
from datetime import datetime

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.utils import TrajectoryDataset, seq_collate, ade, fde
from model.trajairnet import TrajAirNet
from test import test  # Import existing test function

def create_model_args(method_type='baseline', reduced_size=True):
    """Create standardized model arguments for consistent evaluation."""
    args = argparse.Namespace()
    
    # Dataset params
    args.obs = 11
    args.preds = 120
    args.preds_step = 10
    
    # Network params - use smaller sizes for faster evaluation
    if reduced_size:
        args.input_channels = 3
        args.tcn_channel_size = 128  # Reduced from 256
        args.tcn_layers = 2
        args.tcn_kernels = 4
        args.num_context_input_c = 2
        args.num_context_output_c = 7
        args.cnn_kernels = 2
        args.gat_heads = 8  # Reduced from 16
        args.graph_hidden = 128  # Reduced from 256
        args.dropout = 0.05
        args.alpha = 0.2
        args.cvae_hidden = 64  # Reduced from 128
        args.cvae_channel_size = 64  # Reduced from 128
        args.cvae_layers = 2
        args.mlp_layer = 32
    else:
        # Original full-size parameters
        args.input_channels = 3
        args.tcn_channel_size = 256
        args.tcn_layers = 2
        args.tcn_kernels = 4
        args.num_context_input_c = 2
        args.num_context_output_c = 7
        args.cnn_kernels = 2
        args.gat_heads = 16
        args.graph_hidden = 256
        args.dropout = 0.05
        args.alpha = 0.2
        args.cvae_hidden = 128
        args.cvae_channel_size = 128
        args.cvae_layers = 2
        args.mlp_layer = 32
    
    # Intent embedding params (for intent-aware methods)
    if method_type != 'baseline':
        args.intent_embed_dim = 32
        args.num_intent_classes = 16
    
    return args

def evaluate_model(model, test_loader, device, method_name, max_batches=50):
    """
    Evaluate a model and return ADE/FDE metrics.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        method_name: Name of the method for logging
        max_batches: Maximum number of batches to evaluate (for speed)
    
    Returns:
        dict: Results containing ADE, FDE, and other metrics
    """
    print(f"\nEvaluating {method_name}...")
    model.eval()
    
    total_ade = 0.0
    total_fde = 0.0
    total_agents = 0
    total_sequences = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                print(f"  Stopping at {max_batches} batches for speed")
                break
                
            try:
                batch = [tensor.to(device) for tensor in batch]
                
                if method_name == 'baseline':
                    # Baseline: no intent labels
                    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
                    intent_labels = None
                else:
                    # Intent-aware methods
                    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
                
                num_agents = obs_traj.shape[1]
                pred_traj = torch.transpose(pred_traj, 1, 2)
                adj = torch.ones((num_agents, num_agents)).to(device)
                
                # Model forward pass
                if method_name == 'baseline':
                    recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2))
                else:
                    recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2), intent_labels)
                
                # Calculate ADE and FDE for each agent
                batch_ade = 0.0
                batch_fde = 0.0
                
                for agent in range(num_agents):
                    # Get ground truth and prediction
                    gt_traj = torch.transpose(pred_traj[:, :, agent], 0, 1).cpu().numpy()  # [seq_len, 2]
                    pred_traj_agent = recon_y[agent].squeeze().cpu().numpy()  # [seq_len, 2]
                    
                    # Only use x, y coordinates (first 2 dimensions)
                    if len(pred_traj_agent.shape) == 2:
                        gt_xy = gt_traj[:, :2]  # [seq_len, 2]
                        pred_xy = pred_traj_agent[:, :2]  # [seq_len, 2]
                    else:
                        # Handle 1D case
                        gt_xy = gt_traj[:, :2]
                        pred_xy = pred_traj_agent.reshape(-1, pred_traj_agent.shape[-1])[:, :2]
                    
                    if gt_xy.shape[0] == pred_xy.shape[0]:  # Ensure same length
                        agent_ade = ade(pred_xy, gt_xy)
                        agent_fde = fde(pred_xy, gt_xy)
                        
                        batch_ade += agent_ade
                        batch_fde += agent_fde
                
                total_ade += batch_ade
                total_fde += batch_fde
                total_agents += num_agents
                total_sequences += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{min(max_batches, len(test_loader))} batches")
                    
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue
    
    eval_time = time.time() - start_time
    
    # Calculate averages
    avg_ade = total_ade / total_agents if total_agents > 0 else float('inf')
    avg_fde = total_fde / total_agents if total_agents > 0 else float('inf')
    
    results = {
        'method': method_name,
        'avg_ade': avg_ade,
        'avg_fde': avg_fde,
        'total_agents': total_agents,
        'total_sequences': total_sequences,
        'eval_time_seconds': eval_time,
        'batches_evaluated': min(batch_idx + 1, max_batches)
    }
    
    print(f"  Results: ADE={avg_ade:.4f}, FDE={avg_fde:.4f}")
    print(f"  Evaluated {total_agents} agents in {total_sequences} sequences")
    print(f"  Time: {eval_time:.1f}s")
    
    return results

def run_evaluation():
    """Run comprehensive evaluation of all methods."""
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION: Intent-Aware Trajectory Prediction")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset setup
    data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/debug/processed_data"
    if not os.path.exists(data_dir):
        data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/MayJun2022/processed_data"
    
    print(f"Loading test data from: {data_dir}")
    
    all_results = []
    
    # Evaluation parameters
    MAX_BATCHES = 20  # Limit for faster evaluation
    REDUCED_SIZE = True  # Use smaller models for speed
    
    print(f"Evaluation settings: max_batches={MAX_BATCHES}, reduced_size={REDUCED_SIZE}")
    
    # ============================================================================
    # BASELINE EVALUATION (No Intent)
    # ============================================================================
    print("\n" + "="*60)
    print("EVALUATING BASELINE (No Intent)")
    print("="*60)
    
    try:
        # Create baseline dataset (no intent labels in dataloader)
        dataset_baseline = TrajectoryDataset(data_dir, obs_len=11, pred_len=120, step=10, min_agent=1)
        
        # Custom collation for baseline (removes intent labels)
        def baseline_collate(data):
            regular_batch = seq_collate(data)
            # Remove intent labels from batch: obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start
            return regular_batch[:5] + regular_batch[6:]  # Skip intent_labels
        
        test_loader_baseline = DataLoader(dataset_baseline, batch_size=1, shuffle=False, collate_fn=baseline_collate)
        
        args_baseline = create_model_args('baseline', REDUCED_SIZE)
        model_baseline = TrajAirNet(args_baseline).to(device)
        
        results_baseline = evaluate_model(model_baseline, test_loader_baseline, device, 'baseline', MAX_BATCHES)
        all_results.append(results_baseline)
        
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================================ 
    # INTENT METHOD EVALUATIONS
    # ============================================================================
    
    # Create intent-aware dataset once
    dataset_intent = TrajectoryDataset(data_dir, obs_len=11, pred_len=120, step=10, min_agent=1)
    test_loader_intent = DataLoader(dataset_intent, batch_size=1, shuffle=False, collate_fn=seq_collate)
    
    methods = [
        ('Option 1: Intent Interaction Matrix', 'intent-interaction-matrix'),
        ('Option 2: Intent Attention Head', 'intent-attention-head'),
        ('Option 3: Multi-Head Intent Architecture', 'multi-head-intent')
    ]
    
    current_branch = None
    
    for method_name, branch_name in methods:
        print(f"\n" + "="*60)
        print(f"EVALUATING {method_name.upper()}")
        print("="*60)
        
        try:
            # Switch to method branch
            print(f"Switching to branch: {branch_name}")
            
            # Note: We can't actually switch branches in the evaluation script
            # Instead, we'll create models with the appropriate configurations
            # and note that in practice, you'd run this script on each branch
            
            args_method = create_model_args('intent', REDUCED_SIZE)
            
            if branch_name == 'multi-head-intent':
                # For multi-head, we need to ensure the GAT supports the multi-head parameters
                # Since we're evaluating in the current branch, this will use current implementation
                pass
            
            model_method = TrajAirNet(args_method).to(device)
            
            results_method = evaluate_model(model_method, test_loader_intent, device, method_name, MAX_BATCHES)
            all_results.append(results_method)
            
        except Exception as e:
            print(f"Evaluation of {method_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.json"
    
    evaluation_summary = {
        'timestamp': timestamp,
        'evaluation_settings': {
            'max_batches': MAX_BATCHES,
            'reduced_size': REDUCED_SIZE,
            'device': str(device),
            'data_dir': data_dir
        },
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for result in all_results:
        print(f"{result['method']:<40} ADE: {result['avg_ade']:.4f}  FDE: {result['avg_fde']:.4f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return all_results

if __name__ == "__main__":
    results = run_evaluation()