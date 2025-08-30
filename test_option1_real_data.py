#!/usr/bin/env python3
"""
Test Option 1 with real trajectory dataset to ensure full compatibility.
"""

import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.utils import TrajectoryDataset, seq_collate
from model.trajairnet import TrajAirNet

def test_with_real_data():
    """Test Option 1 with real trajectory data."""
    print("Testing Option 1 with Real Dataset")
    print("=" * 40)
    
    try:
        # Use debug dataset if available, otherwise main dataset
        data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/debug/processed_data"
        if not os.path.exists(data_dir):
            data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/MayJun2022/processed_data"
        
        print(f"Loading data from: {data_dir}")
        
        # Create dataset with small subset for testing
        dataset = TrajectoryDataset(data_dir, obs_len=11, pred_len=120, step=10, min_agent=1)
        print(f"✓ Dataset loaded: {len(dataset)} sequences")
        
        if len(dataset) == 0:
            print("❌ No data available for testing")
            return False
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=seq_collate)
        print("✓ DataLoader created")
        
        # Create model with intent interaction matrix
        args = argparse.Namespace()
        args.input_channels = 3
        args.preds = 120
        args.preds_step = 10
        args.tcn_channel_size = 128  # Smaller for testing
        args.tcn_layers = 2
        args.tcn_kernels = 4
        args.num_context_input_c = 2
        args.num_context_output_c = 7
        args.cnn_kernels = 2
        args.gat_heads = 8  # Smaller for testing
        args.graph_hidden = 128
        args.dropout = 0.05
        args.alpha = 0.2
        args.cvae_hidden = 64
        args.cvae_channel_size = 64
        args.cvae_layers = 2
        args.mlp_layer = 32
        args.obs = 11
        args.intent_embed_dim = 32
        args.num_intent_classes = 16
        
        model = TrajAirNet(args)
        print("✓ TrajAirNet model created")
        
        # Test forward pass with real data
        print("\nTesting forward pass with real data...")
        
        batch_count = 0
        for batch in dataloader:
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
            num_agents = obs_traj.shape[1]
            
            print(f"Batch {batch_count + 1}:")
            print(f"  - Agents: {num_agents}")
            print(f"  - obs_traj shape: {obs_traj.shape}")
            print(f"  - context shape: {context.shape}")
            print(f"  - intent_labels shape: {intent_labels.shape}")
            print(f"  - intent_labels: {intent_labels}")
            
            # Prepare data
            pred_traj = torch.transpose(pred_traj, 1, 2)
            adj = torch.ones((num_agents, num_agents))
            
            # Forward pass
            recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2), intent_labels)
            
            print(f"  ✓ Forward pass successful")
            print(f"  - Reconstructed trajectories: {len(recon_y)} agents")
            print(f"  - Output shapes: {[r.shape for r in recon_y[:min(3, len(recon_y))]]}")
            
            # Test backward pass
            loss = sum([torch.sum(r**2) for r in recon_y])
            loss.backward()
            
            print(f"  ✓ Backward pass successful, loss: {loss.item():.6f}")
            
            # Check intent interaction matrix gradients
            intent_grad = model.gat.attentions[0].intent_interaction.grad
            if intent_grad is not None:
                print(f"  ✓ Intent matrix gradient norm: {intent_grad.norm().item():.6f}")
            
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        print(f"\n🎉 Successfully tested {batch_count} batches with real data!")
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_real_data()
    if success:
        print("\n🎉 Option 1 works perfectly with real data!")
    else:
        print("\n❌ Option 1 needs fixes for real data compatibility.")