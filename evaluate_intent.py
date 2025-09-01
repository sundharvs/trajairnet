#!/usr/bin/env python3
"""
Intent-aware evaluation script - evaluates TrajAirNet with intent labels.
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

def evaluate_intent_model(method_name, epoch=5):
    """Evaluate intent-aware TrajAirNet model."""
    print("=" * 60)
    print(f"INTENT EVALUATION: {method_name}")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset setup - match training script paths
    dataset_name = "7days1"  # Match training script
    dataset_folder = "/dataset/"
    datapath = os.getcwd() + dataset_folder + dataset_name + "/processed_data/"
    test_path = datapath + "test"
    
    # Fallback paths if main dataset not found
    if not os.path.exists(test_path):
        data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/debug/processed_data"
        if not os.path.exists(data_dir):
            data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/MayJun2022/processed_data"
        test_path = data_dir
        
    print(f"Loading data from: {test_path}")
    
    # Create dataset with intent labels
    dataset = TrajectoryDataset(test_path, obs_len=11, pred_len=120, step=10)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=seq_collate)
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    # Model setup - match training script parameters
    args = argparse.Namespace()
    args.input_channels = 3
    args.preds = 120
    args.preds_step = 10
    args.tcn_channel_size = 256  # Full size to match training
    args.tcn_layers = 2
    args.tcn_kernels = 4
    args.num_context_input_c = 2
    args.num_context_output_c = 7
    args.cnn_kernels = 2
    args.gat_heads = 16  # Full size to match training
    args.graph_hidden = 256  # Full size to match training
    args.dropout = 0.05
    args.alpha = 0.2
    args.cvae_hidden = 128  # Full size to match training
    args.cvae_channel_size = 128  # Full size to match training
    args.cvae_layers = 2
    args.mlp_layer = 32
    args.obs = 11
    args.intent_embed_dim = 32
    args.num_intent_classes = 16
    
    model = TrajAirNet(args).to(device)
    
    # Using random initialization for quick comparison
    print("Using random initialization for intent model (no training performed)")
    
    print(f"{method_name} model loaded")
    
    # Evaluation
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    total_agents = 0
    total_sequences = 0
    max_batches = 50  # Match baseline for fair comparison
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
                
            try:
                batch = [tensor.to(device) for tensor in batch]
                # Intent-aware expects: obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
                
                num_agents = obs_traj.shape[1]
                pred_traj_transposed = torch.transpose(pred_traj, 1, 2)
                adj = torch.ones((num_agents, num_agents)).to(device)
                
                # Intent-aware forward pass
                recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj_transposed, adj[0], torch.transpose(context, 1, 2), intent_labels)
                
                # Calculate metrics
                batch_ade = 0.0
                batch_fde = 0.0
                
                for agent in range(num_agents):
                    gt_traj = torch.transpose(pred_traj_transposed[:, :, agent], 0, 1).cpu().numpy()
                    pred_traj_agent = recon_y[agent].squeeze().cpu().numpy()
                    
                    if len(pred_traj_agent.shape) == 1:
                        pred_traj_agent = pred_traj_agent.reshape(-1, 1)
                    if len(gt_traj.shape) == 1:
                        gt_traj = gt_traj.reshape(-1, 1)
                    
                    # Use only x, y coordinates
                    min_len = min(gt_traj.shape[0], pred_traj_agent.shape[0])
                    gt_xy = gt_traj[:min_len, :2] if gt_traj.shape[1] >= 2 else gt_traj[:min_len, :1]
                    pred_xy = pred_traj_agent[:min_len, :2] if pred_traj_agent.shape[1] >= 2 else pred_traj_agent[:min_len, :1]
                    
                    if gt_xy.shape == pred_xy.shape and gt_xy.shape[0] > 0:
                        agent_ade = ade(pred_xy, gt_xy)
                        agent_fde = fde(pred_xy, gt_xy)
                        
                        if not (np.isnan(agent_ade) or np.isnan(agent_fde)):
                            batch_ade += agent_ade
                            batch_fde += agent_fde
                
                total_ade += batch_ade
                total_fde += batch_fde
                total_agents += num_agents
                total_sequences += 1
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"Processed {batch_idx + 1}/{max_batches} batches")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    eval_time = time.time() - start_time
    
    # Results
    avg_ade = total_ade / total_agents if total_agents > 0 else float('inf')
    avg_fde = total_fde / total_agents if total_agents > 0 else float('inf')
    
    # Get current branch name
    try:
        with os.popen('git rev-parse --abbrev-ref HEAD') as f:
            branch_name = f.read().strip()
    except:
        branch_name = 'unknown'
    
    results = {
        'method': method_name,
        'branch': branch_name,
        'avg_ade': float(avg_ade),
        'avg_fde': float(avg_fde),
        'total_agents': int(total_agents),
        'total_sequences': int(total_sequences),
        'eval_time_seconds': float(eval_time),
        'batches_evaluated': int(min(batch_idx + 1, max_batches)),
        'timestamp': datetime.now().isoformat(),
        'model_params': {
            'tcn_channel_size': int(args.tcn_channel_size),
            'gat_heads': int(args.gat_heads),
            'graph_hidden': int(args.graph_hidden),
            'cvae_hidden': int(args.cvae_hidden),
            'intent_embed_dim': int(args.intent_embed_dim),
            'num_intent_classes': int(args.num_intent_classes)
        }
    }
    
    print(f"\n{method_name} RESULTS:")
    print(f"ADE: {avg_ade:.6f}")
    print(f"FDE: {avg_fde:.6f}")
    print(f"Agents: {total_agents}")
    print(f"Sequences: {total_sequences}")
    print(f"Time: {eval_time:.1f}s")
    
    # Save results with branch-specific filename
    filename = f'intent_results_{branch_name}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filename}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate intent-aware TrajAirNet model')
    parser.add_argument('--epoch', type=int, default=5, help='Epoch of model to evaluate')
    
    args = parser.parse_args()
    
    # Determine method name from branch
    try:
        with os.popen('git rev-parse --abbrev-ref HEAD') as f:
            branch_name = f.read().strip()
        
        if 'interaction-matrix' in branch_name:
            method_name = "Option 1: Intent Interaction Matrix"
        elif 'attention-head' in branch_name:
            method_name = "Option 2: Intent Attention Head"
        elif 'multi-head' in branch_name:
            method_name = "Option 3: Multi-Head Intent"
        else:
            method_name = f"Intent-Aware ({branch_name})"
    except:
        method_name = "Intent-Aware (Unknown)"
    
    evaluate_intent_model(method_name, args.epoch)