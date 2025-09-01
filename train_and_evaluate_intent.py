#!/usr/bin/env python3
"""
Complete training and evaluation pipeline for intent-aware methods.
This script ACTUALLY trains models on the MayJun2022 dataset (which has intent labels)
and then evaluates them properly.
"""

import sys
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
import time
import json
from datetime import datetime
import subprocess
from tqdm import tqdm

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.utils import TrajectoryDataset, seq_collate, ade, fde, loss_func
from model.trajairnet import TrajAirNet

def train_intent_model(method_name, branch_name, epochs=10, save_every=2):
    """
    Train an intent-aware model on the MayJun2022 dataset (which has intent labels).
    """
    print("=" * 80)
    print(f"TRAINING INTENT-AWARE MODEL: {method_name}")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset setup - USE MAYJUN2022 DATASET (which has intent labels)
    dataset_name = "MayJun2022"
    dataset_folder = "/dataset/"
    datapath = os.getcwd() + dataset_folder + dataset_name + "/processed_data/"
    
    # Check if data exists
    train_path = datapath + "train" if os.path.exists(datapath + "train") else datapath
    test_path = datapath + "test" if os.path.exists(datapath + "test") else datapath
    
    print(f"Training data path: {train_path}")
    print(f"Test data path: {test_path}")
    
    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}")
        return None
    
    # Create datasets - these will automatically load intent labels
    print("Loading training dataset with intent labels...")
    dataset_train = TrajectoryDataset(train_path, obs_len=11, pred_len=120, step=10)
    print(f"Training dataset loaded: {len(dataset_train)} sequences")
    
    if os.path.exists(test_path) and test_path != train_path:
        print("Loading test dataset with intent labels...")
        dataset_test = TrajectoryDataset(test_path, obs_len=11, pred_len=120, step=10)
        print(f"Test dataset loaded: {len(dataset_test)} sequences")
    else:
        # Use same data for train/test if no separate test set
        dataset_test = dataset_train
        print("Using training data for testing (no separate test set)")
    
    # Create data loaders
    loader_train = DataLoader(dataset_train, batch_size=1, num_workers=4, shuffle=True, collate_fn=seq_collate)
    loader_test = DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=False, collate_fn=seq_collate)
    
    # Model setup - FULL SIZE
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
    
    # Intent parameters (CRITICAL FOR INTENT-AWARE METHODS)
    args.intent_embed_dim = 32
    args.num_intent_classes = 16
    
    model = TrajAirNet(args).to(device)
    print(f"Model created: {method_name}")
    print(f"Intent embedding dimension: {args.intent_embed_dim}")
    print(f"Number of intent classes: {args.num_intent_classes}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting training for {epochs} epochs...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch}/{epochs}")
        
        for batch_idx, batch in enumerate(tqdm(loader_train, desc=f"Epoch {epoch}")):
            try:
                batch = [tensor.to(device) for tensor in batch]
                
                # CRITICAL: Unpack batch with intent labels
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
                
                num_agents = obs_traj.shape[1]
                pred_traj = torch.transpose(pred_traj, 1, 2)
                adj = torch.ones((num_agents, num_agents)).to(device)
                
                optimizer.zero_grad()
                
                # CRITICAL: Forward pass with intent labels
                recon_y, m, var = model(
                    torch.transpose(obs_traj, 1, 2), 
                    pred_traj, 
                    adj[0], 
                    torch.transpose(context, 1, 2), 
                    intent_labels  # INTENT LABELS PASSED TO MODEL
                )
                
                # Calculate loss
                loss = 0
                for agent in range(num_agents):
                    loss += loss_func(
                        recon_y[agent], 
                        torch.transpose(pred_traj[:, :, agent], 0, 1).unsqueeze(0), 
                        m[agent], 
                        var[agent]
                    )
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.6f}")
        
        # Save model
        if epoch % save_every == 0 or epoch == epochs:
            model_dir = os.getcwd() + "/saved_models/"
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = model_dir + f"intent_model_{branch_name}_{dataset_name}_{epoch}.pt"
            print(f"Saving model: {model_path}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'method_name': method_name,
                'branch_name': branch_name,
                'dataset_name': dataset_name,
                'args': args
            }, model_path)
        
        # Quick evaluation every few epochs
        if epoch % save_every == 0:
            print(f"Quick evaluation at epoch {epoch}...")
            model.eval()
            total_ade = 0.0
            total_fde = 0.0
            total_agents = 0
            eval_batches = 0
            max_eval_batches = 20  # Quick evaluation
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader_test):
                    if batch_idx >= max_eval_batches:
                        break
                    
                    try:
                        batch = [tensor.to(device) for tensor in batch]
                        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
                        
                        num_agents = obs_traj.shape[1]
                        pred_traj_t = torch.transpose(pred_traj, 1, 2)
                        adj = torch.ones((num_agents, num_agents)).to(device)
                        
                        # Forward pass with intent labels
                        recon_y, m, var = model(
                            torch.transpose(obs_traj, 1, 2), 
                            pred_traj_t, 
                            adj[0], 
                            torch.transpose(context, 1, 2), 
                            intent_labels
                        )
                        
                        # Calculate metrics
                        for agent in range(num_agents):
                            gt_traj = torch.transpose(pred_traj_t[:, :, agent], 0, 1).cpu().numpy()
                            pred_traj_agent = recon_y[agent].squeeze().cpu().numpy()
                            
                            if len(pred_traj_agent.shape) == 1:
                                pred_traj_agent = pred_traj_agent.reshape(-1, 1)
                            if len(gt_traj.shape) == 1:
                                gt_traj = gt_traj.reshape(-1, 1)
                            
                            # Use x,y coordinates
                            min_len = min(gt_traj.shape[0], pred_traj_agent.shape[0])
                            gt_xy = gt_traj[:min_len, :2] if gt_traj.shape[1] >= 2 else gt_traj[:min_len, :1]
                            pred_xy = pred_traj_agent[:min_len, :2] if pred_traj_agent.shape[1] >= 2 else pred_traj_agent[:min_len, :1]
                            
                            if gt_xy.shape == pred_xy.shape and gt_xy.shape[0] > 0:
                                agent_ade = ade(pred_xy, gt_xy)
                                agent_fde = fde(pred_xy, gt_xy)
                                
                                if not (np.isnan(agent_ade) or np.isnan(agent_fde)):
                                    total_ade += agent_ade
                                    total_fde += agent_fde
                        
                        total_agents += num_agents
                        eval_batches += 1
                        
                    except Exception as e:
                        continue
            
            if total_agents > 0:
                avg_ade = total_ade / total_agents
                avg_fde = total_fde / total_agents
                print(f"  Quick eval ADE: {avg_ade:.6f}, FDE: {avg_fde:.6f}")
            
            model.train()  # Back to training mode
    
    print(f"\nTraining completed for {method_name}!")
    return model_path

def evaluate_intent_model(method_name, branch_name, dataset_name="MayJun2022", epoch=10, max_batches=100):
    """
    Evaluate a trained intent-aware model.
    """
    print("=" * 80)
    print(f"EVALUATING INTENT-AWARE MODEL: {method_name}")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_dir = os.getcwd() + "/saved_models/"
    model_path = model_dir + f"intent_model_{branch_name}_{dataset_name}_{epoch}.pt"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model not found at {model_path}")
        print("Available intent models:")
        if os.path.exists(model_dir):
            models = sorted([f for f in os.listdir(model_dir) if f.startswith('intent_model_')])
            for m in models:
                print(f"  {m}")
        return None
    
    print(f"Loading trained model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model with same args
    saved_args = checkpoint['args']
    model = TrajAirNet(saved_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Dataset setup
    dataset_folder = "/dataset/"
    datapath = os.getcwd() + dataset_folder + dataset_name + "/processed_data/"
    test_path = datapath + "test" if os.path.exists(datapath + "test") else datapath
    
    print(f"Loading test data: {test_path}")
    dataset_test = TrajectoryDataset(test_path, obs_len=11, pred_len=120, step=10)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=seq_collate)
    
    # Evaluation
    total_ade = 0.0
    total_fde = 0.0
    total_agents = 0
    total_sequences = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if batch_idx >= max_batches:
                break
                
            try:
                batch = [tensor.to(device) for tensor in batch]
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
                
                num_agents = obs_traj.shape[1]
                pred_traj_t = torch.transpose(pred_traj, 1, 2)
                adj = torch.ones((num_agents, num_agents)).to(device)
                
                # Forward pass with intent labels
                recon_y, m, var = model(
                    torch.transpose(obs_traj, 1, 2), 
                    pred_traj_t, 
                    adj[0], 
                    torch.transpose(context, 1, 2), 
                    intent_labels
                )
                
                # Calculate metrics
                batch_ade = 0.0
                batch_fde = 0.0
                
                for agent in range(num_agents):
                    gt_traj = torch.transpose(pred_traj_t[:, :, agent], 0, 1).cpu().numpy()
                    pred_traj_agent = recon_y[agent].squeeze().cpu().numpy()
                    
                    if len(pred_traj_agent.shape) == 1:
                        pred_traj_agent = pred_traj_agent.reshape(-1, 1)
                    if len(gt_traj.shape) == 1:
                        gt_traj = gt_traj.reshape(-1, 1)
                    
                    # Use x,y coordinates
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
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    eval_time = time.time() - start_time
    
    # Results
    avg_ade = total_ade / total_agents if total_agents > 0 else float('inf')
    avg_fde = total_fde / total_agents if total_agents > 0 else float('inf')
    
    results = {
        'method': method_name,
        'branch': branch_name,
        'dataset': dataset_name,
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
    
    print(f"\n{method_name} EVALUATION RESULTS:")
    print(f"  ADE: {avg_ade:.6f}")
    print(f"  FDE: {avg_fde:.6f}")
    print(f"  Agents evaluated: {total_agents}")
    print(f"  Sequences evaluated: {total_sequences}")
    print(f"  Evaluation time: {eval_time:.1f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'intent_trained_results_{branch_name}_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filename}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate intent-aware TrajAirNet')
    parser.add_argument('--train', action='store_true', help='Train model before evaluation')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--eval_epoch', type=int, default=10, help='Which epoch to evaluate')
    parser.add_argument('--max_eval_batches', type=int, default=100, help='Max batches for evaluation')
    parser.add_argument('--dataset', type=str, default='MayJun2022', help='Dataset name')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("INTENT-AWARE TRAJAIRNET: PROPER TRAINING & EVALUATION")
    print("=" * 80)
    
    # Get current branch and method name
    try:
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        current_branch = result.stdout.strip()
    except:
        current_branch = 'unknown'
    
    if 'interaction-matrix' in current_branch:
        method_name = "Option 1: Intent Interaction Matrix"
    elif 'attention-head' in current_branch:
        method_name = "Option 2: Intent Attention Head"
    elif 'multi-head' in current_branch:
        method_name = "Option 3: Multi-Head Intent"
    else:
        method_name = f"Intent Method ({current_branch})"
    
    print(f"Current branch: {current_branch}")
    print(f"Method: {method_name}")
    print(f"Dataset: {args.dataset} (with intent labels)")
    
    # Train if requested
    if args.train:
        print("\n" + "="*60)
        print("TRAINING PHASE")
        print("="*60)
        
        model_path = train_intent_model(method_name, current_branch, args.train_epochs)
        
        if model_path:
            print(f"Training completed! Model saved to: {model_path}")
        else:
            print("Training failed!")
            return
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    results = evaluate_intent_model(
        method_name, 
        current_branch, 
        args.dataset, 
        args.eval_epoch, 
        args.max_eval_batches
    )
    
    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"Final ADE: {results['avg_ade']:.6f}")
        print(f"Final FDE: {results['avg_fde']:.6f}")
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main()