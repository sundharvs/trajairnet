#!/usr/bin/env python3
"""
Baseline evaluation script - evaluates original TrajAirNet without intent labels.
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

def evaluate_baseline():
    """Evaluate baseline TrajAirNet model without intent labels."""
    print("=" * 60)
    print("BASELINE EVALUATION: Original TrajAirNet")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset setup - USE SIMPLER DATASET FOR QUICK COMPARISON
    data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/debug/processed_data"
    print(f"Loading data from: {data_dir}")
    
    # Create dataset
    dataset = TrajectoryDataset(data_dir, obs_len=11, pred_len=120, step=10, min_agent=1)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=seq_collate)
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    # Model setup
    args = argparse.Namespace()
    args.input_channels = 3
    args.preds = 120
    args.preds_step = 10
    args.tcn_channel_size = 128  # Reduced for speed
    args.tcn_layers = 2
    args.tcn_kernels = 4
    args.num_context_input_c = 2
    args.num_context_output_c = 7
    args.cnn_kernels = 2
    args.gat_heads = 8  # Reduced for speed
    args.graph_hidden = 128
    args.dropout = 0.05
    args.alpha = 0.2
    args.cvae_hidden = 64
    args.cvae_channel_size = 64
    args.cvae_layers = 2
    args.mlp_layer = 32
    args.obs = 11
    
    model = TrajAirNet(args).to(device)
    
    # LOAD TRAINED BASELINE MODEL FOR FAIR COMPARISON
    dataset_name = "MayJun2022"
    epoch = 5  # Default epoch to load
    model_path = os.getcwd() + "/saved_models/" + f"model_{dataset_name}_{epoch}.pt"
    
    if os.path.exists(model_path):
        print(f"Loading trained baseline model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded baseline model from epoch {checkpoint['epoch']}")
    else:
        print(f"WARNING: No trained baseline model found at {model_path}")
        print("Using random initialization for quick comparison!")
        # List available models
        model_dir = os.getcwd() + "/saved_models/"
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.startswith('model_')]
            if models:
                print(f"Available models: {models}")
            else:
                print("No trained models found")
    
    print("Baseline model ready for evaluation")
    
    # Evaluation
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    total_agents = 0
    total_sequences = 0
    max_batches = 50  # Reduced for quick comparison
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
                
            try:
                batch = [tensor.to(device) for tensor in batch]
                # Handle both old and new batch formats
                if len(batch) == 7:
                    # New format with intent labels: obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start
                    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
                else:
                    # Old format: obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, timestamp, tail, seq_start
                    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, timestamp, tail, seq_start = batch
                
                num_agents = obs_traj.shape[1]
                pred_traj_transposed = torch.transpose(pred_traj, 1, 2)
                adj = torch.ones((num_agents, num_agents)).to(device)
                
                # Baseline forward pass (no intent labels)
                recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj_transposed, adj[0], torch.transpose(context, 1, 2))
                
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
    
    results = {
        'method': 'Baseline (No Intent)',
        'branch': 'main',
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
            'cvae_hidden': int(args.cvae_hidden)
        }
    }
    
    print(f"\nBASELINE RESULTS:")
    print(f"ADE: {avg_ade:.6f}")
    print(f"FDE: {avg_fde:.6f}")
    print(f"Agents: {total_agents}")
    print(f"Sequences: {total_sequences}")
    print(f"Time: {eval_time:.1f}s")
    
    # Save results
    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: baseline_results.json")
    return results

if __name__ == "__main__":
    evaluate_baseline()