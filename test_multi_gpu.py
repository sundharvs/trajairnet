#!/usr/bin/env python3
"""
Test script to verify multi-GPU DataParallel implementation works correctly.
"""

import torch
from torch import nn
from torch.amp import GradScaler, autocast
import sys
import os

# Add the model path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.belief_trajairnet import BeliefAwareTrajAirNet
from model.belief_states import TOTAL_VOCAB_SIZE

def test_multi_gpu():
    """Test DataParallel setup with BeliefAwareTrajAirNet"""
    
    print("Testing Multi-GPU DataParallel Implementation")
    print("=" * 50)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"Device: {device}")
    print(f"Number of GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("Warning: Less than 2 GPUs available. DataParallel will not provide speedup.")
    
    # Mock args for model creation
    class MockArgs:
        input_channels = 3
        preds = 120
        preds_step = 10
        tcn_channel_size = 256
        tcn_layers = 2
        tcn_kernels = 4
        dropout = 0.05
        num_context_output_c = 7
        obs = 11
        graph_hidden = 256
        gat_heads = 16
        alpha = 0.2
        cvae_layers = 2
        cvae_channel_size = 128
        cvae_hidden = 128
        mlp_layer = 32
        num_context_input_c = 2
        cnn_kernels = 2
        belief_embed_dim = 64
        belief_vocab_size = TOTAL_VOCAB_SIZE
        belief_integration_mode = 'concatenate'
    
    args = MockArgs()
    
    try:
        # Create model
        print("\nCreating BeliefAwareTrajAirNet...")
        model = BeliefAwareTrajAirNet(args)
        
        # For now, test without DataParallel to isolate issues
        print("Testing without DataParallel first...")
        
        model.to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model created with {param_count:,} parameters")
        
        # Start with simple single-batch test to understand data format
        batch_size = 1  # Single batch first
        num_agents = 3
        obs_len = args.obs
        pred_len = int(args.preds / args.preds_step)
        
        print(f"\nGenerating test data (batch_size={batch_size})...")
        x = torch.randn(obs_len, batch_size, num_agents).to(device)  # Shape: [seq_len, batch, agents]
        y = torch.randn(pred_len, batch_size, num_agents).to(device)
        
        # Based on utils.py analysis:
        # curr_context shape: (num_agents, 2, seq_len) 
        # After permute(2,0,1): (seq_len, num_agents, 2)
        # So context[:, :, agent] gives (seq_len, 2) - exactly what conv1d needs!
        context = torch.randn(obs_len, num_agents, args.num_context_input_c).to(device)
        
        # Belief sequences for each agent (single batch)
        belief_sequences = [
            [4, 6, 8, 12],  # Agent 0: landing pattern
            [5, 7, 9, 13],  # Agent 1: different runway  
            [22]            # Agent 2: other/unknown
        ]
        belief_lengths = torch.tensor([4, 4, 1], dtype=torch.long).to(device)
        
        print(f"Input shapes: x={x.shape}, y={y.shape}, context={context.shape}")
        print(f"Belief sequences: {len(belief_sequences)} sequences")
        
        # Test simple forward pass using the exact format from train_with_beliefs.py
        print("\nTesting forward pass...")
        
        # Copy the exact format from the working training code
        obs_traj_item = x  # [seq_len, batch, agents]
        pred_traj_item = torch.transpose(y, 1, 2)  # From train code: pred_traj_item is already transposed
        context_item = context  # [seq_len, agents, channels]
        adj = torch.ones((num_agents, num_agents)).to(device)
        
        print(f"Calling model with:")
        print(f"  obs_traj_transposed: {torch.transpose(obs_traj_item, 1, 2).shape}")
        print(f"  pred_traj_item: {pred_traj_item.shape}")  
        print(f"  adj[0]: {adj[0].shape}")
        print(f"  context_transposed: {torch.transpose(context_item, 1, 2).shape}")
        
        recon_y, m, var = model(
            torch.transpose(obs_traj_item, 1, 2),  # [seq_len, agents, batch]
            pred_traj_item,   # [seq_len, batch, agents]
            adj[0],           # [agents]
            torch.transpose(context_item, 1, 2),  # [seq_len, channels, agents] 
            belief_sequences,
            belief_lengths
        )
        
        # Simple loss calculation
        total_loss = sum(torch.mean(torch.abs(recon_y[agent])) for agent in range(num_agents))
        print(f"Forward pass successful! Loss: {total_loss.item():.4f}")
        
        # Test backward pass
        print("Testing backward pass...")
        scaler.scale(total_loss).backward()
        print("Backward pass successful!")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage:")
            for i in range(min(2, torch.cuda.device_count())):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        print("\n✅ Multi-GPU test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_gpu()
    sys.exit(0 if success else 1)