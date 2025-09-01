#!/usr/bin/env python3
"""
Proper evaluation script that trains models if needed and evaluates with checkpoints.
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
import subprocess

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.utils import TrajectoryDataset, seq_collate, ade, fde
from model.trajairnet import TrajAirNet

def run_training(branch_name, epochs=10):
    """Run training for a specific branch."""
    print(f"Training model on branch {branch_name} for {epochs} epochs...")
    
    # Training command matching the original train.py parameters
    cmd = [
        'python', 'train.py',
        '--dataset_name', '7days1',
        '--total_epochs', str(epochs),
        '--lr', '0.001',
        '--save_model', 'True',
        '--evaluate', 'True'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"Training completed successfully for {branch_name}")
            return True
        else:
            print(f"Training failed for {branch_name}:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"Error running training: {e}")
        return False

def evaluate_trained_model(method_name, branch_name, epoch=5, max_batches=50):
    """
    Evaluate a trained model with proper checkpoint loading.
    """
    print(f"\nEvaluating {method_name} (epoch {epoch})...")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset setup - match training exactly
    dataset_name = "7days1"
    dataset_folder = "/dataset/"
    datapath = os.getcwd() + dataset_folder + dataset_name + "/processed_data/"
    test_path = datapath + "test"
    
    # Fallback paths
    if not os.path.exists(test_path):
        fallback_paths = [
            "/home/ssangeetha3/git/ctaf-intent-inference/dataset/debug/processed_data",
            "/home/ssangeetha3/git/ctaf-intent-inference/dataset/MayJun2022/processed_data"
        ]
        
        for path in fallback_paths:
            if os.path.exists(path):
                test_path = path
                break
    
    print(f"Loading test data from: {test_path}")
    
    # Create dataset
    dataset = TrajectoryDataset(test_path, obs_len=11, pred_len=120, step=10)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=seq_collate)
    
    # Model setup - FULL SIZE to match training
    args = argparse.Namespace()
    args.input_channels = 3
    args.preds = 120
    args.preds_step = 10
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
    args.obs = 11
    
    # Intent parameters for intent-aware methods
    if 'baseline' not in method_name.lower():
        args.intent_embed_dim = 32
        args.num_intent_classes = 16
    
    model = TrajAirNet(args).to(device)
    
    # Load checkpoint
    model_path = os.getcwd() + "/saved_models/" + f"model_{dataset_name}_{epoch}.pt"
    
    if os.path.exists(model_path):
        print(f"Loading trained model checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model from epoch {epoch}")
    else:
        print(f"ERROR: No trained model found at {model_path}")
        print("Available models:")
        model_dir = os.getcwd() + "/saved_models/"
        if os.path.exists(model_dir):
            models = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
            for m in models:
                print(f"  {m}")
        return None
    
    # Evaluation
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    total_agents = 0
    total_sequences = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
                
            try:
                batch = [tensor.to(device) for tensor in batch]
                
                # Handle different batch formats
                if 'baseline' in method_name.lower():
                    # For baseline: might not have intent labels
                    if len(batch) == 7:
                        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
                        intent_labels = None  # Ignore for baseline
                    else:
                        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
                        intent_labels = None
                else:
                    # Intent-aware methods
                    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
                
                num_agents = obs_traj.shape[1]
                pred_traj_transposed = torch.transpose(pred_traj, 1, 2)
                adj = torch.ones((num_agents, num_agents)).to(device)
                
                # Model forward pass
                if intent_labels is not None and 'baseline' not in method_name.lower():
                    recon_y, m, var = model(
                        torch.transpose(obs_traj, 1, 2), 
                        pred_traj_transposed, 
                        adj[0], 
                        torch.transpose(context, 1, 2), 
                        intent_labels
                    )
                else:
                    # Baseline or fallback without intent labels
                    recon_y, m, var = model(
                        torch.transpose(obs_traj, 1, 2), 
                        pred_traj_transposed, 
                        adj[0], 
                        torch.transpose(context, 1, 2)
                    )
                
                # Calculate metrics
                batch_ade = 0.0
                batch_fde = 0.0
                
                for agent in range(num_agents):
                    gt_traj = torch.transpose(pred_traj_transposed[:, :, agent], 0, 1).cpu().numpy()
                    pred_traj_agent = recon_y[agent].squeeze().cpu().numpy()
                    
                    # Ensure proper shape
                    if len(pred_traj_agent.shape) == 1:
                        pred_traj_agent = pred_traj_agent.reshape(-1, 1)
                    if len(gt_traj.shape) == 1:
                        gt_traj = gt_traj.reshape(-1, 1)
                    
                    # Use x,y coordinates only
                    min_len = min(gt_traj.shape[0], pred_traj_agent.shape[0])
                    if gt_traj.shape[1] >= 2:
                        gt_xy = gt_traj[:min_len, :2]
                    else:
                        gt_xy = gt_traj[:min_len, :1]
                        
                    if pred_traj_agent.shape[1] >= 2:
                        pred_xy = pred_traj_agent[:min_len, :2]
                    else:
                        pred_xy = pred_traj_agent[:min_len, :1]
                    
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
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{max_batches} batches")
                    
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    eval_time = time.time() - start_time
    
    # Results
    avg_ade = total_ade / total_agents if total_agents > 0 else float('inf')
    avg_fde = total_fde / total_agents if total_agents > 0 else float('inf')
    
    results = {
        'method': method_name,
        'branch': branch_name,
        'epoch': epoch,
        'avg_ade': float(avg_ade),
        'avg_fde': float(avg_fde),
        'total_agents': int(total_agents),
        'total_sequences': int(total_sequences),
        'eval_time_seconds': float(eval_time),
        'batches_evaluated': int(min(batch_idx + 1, max_batches)),
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path
    }
    
    print(f"\n{method_name} RESULTS (Epoch {epoch}):")
    print(f"  ADE: {avg_ade:.6f}")
    print(f"  FDE: {avg_fde:.6f}")
    print(f"  Agents: {total_agents}")
    print(f"  Sequences: {total_sequences}")
    print(f"  Time: {eval_time:.1f}s")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Proper training and evaluation of intent methods')
    parser.add_argument('--train', action='store_true', help='Train models before evaluation')
    parser.add_argument('--epoch', type=int, default=5, help='Epoch to evaluate')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--max_batches', type=int, default=50, help='Max batches for evaluation')
    parser.add_argument('--evaluate_all', action='store_true', help='Evaluate all methods on current branch')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PROPER INTENT METHODS EVALUATION")
    print("=" * 80)
    
    # Get current branch
    try:
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        current_branch = result.stdout.strip()
    except:
        current_branch = 'unknown'
    
    print(f"Current branch: {current_branch}")
    
    # Determine method name
    if 'interaction-matrix' in current_branch:
        method_name = "Option 1: Intent Interaction Matrix"
    elif 'attention-head' in current_branch:
        method_name = "Option 2: Intent Attention Head"  
    elif 'multi-head' in current_branch:
        method_name = "Option 3: Multi-Head Intent"
    elif 'baseline' in current_branch or current_branch == 'main':
        method_name = "Baseline (No Intent)"
    else:
        method_name = f"Intent Method ({current_branch})"
    
    all_results = []
    
    # Train if requested
    if args.train:
        print(f"\nTraining {method_name} for {args.train_epochs} epochs...")
        success = run_training(current_branch, args.train_epochs)
        if not success:
            print("Training failed! Evaluation may not work correctly.")
    
    # Evaluate current method
    print(f"\nEvaluating {method_name}...")
    result = evaluate_trained_model(method_name, current_branch, args.epoch, args.max_batches)
    
    if result:
        all_results.append(result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'proper_evaluation_{current_branch}_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    else:
        print("Evaluation failed!")
    
    if args.evaluate_all:
        print("\nNOTE: To evaluate all methods, run this script on each branch:")
        branches = ['intent-interaction-matrix', 'intent-attention-head', 'multi-head-intent']
        for branch in branches:
            print(f"  git checkout {branch} && python proper_evaluation.py --epoch {args.epoch}")

if __name__ == "__main__":
    main()