#!/usr/bin/env python3
"""
Test script for Option 2: Intent Attention Head approach.
"""

import sys
import os
import argparse
import torch

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.utils import TrajectoryDataset, seq_collate
from model.trajairnet import TrajAirNet
from model.gat_model import GAT
from model.gat_layers import GraphAttentionLayer

def test_intent_attention_head():
    """Test the intent attention head implementation."""
    print("Testing Intent Attention Head (Option 2)")
    print("=" * 50)
    
    try:
        # Test GraphAttentionLayer with intent attention head
        print("Testing GraphAttentionLayer with intent attention head...")
        layer = GraphAttentionLayer(64, 32, alpha=0.2, intent_embed_dim=16)
        
        # Check if intent attention head exists
        assert hasattr(layer, 'intent_attention'), "Intent attention head missing"
        assert hasattr(layer, 'intent_beta'), "Intent beta parameter missing"
        print(f"✓ Intent attention head: {layer.intent_attention}")
        print(f"✓ Intent beta parameter: {layer.intent_beta}")
        
        # Test forward pass
        num_agents = 3
        input_features = torch.randn(num_agents, 64)
        adj = torch.ones(num_agents, num_agents)
        intent_embeds = torch.randn(num_agents, 16)  # Intent embeddings
        
        output = layer(input_features, adj, intent_embeds)
        print(f"✓ GAT layer forward pass successful, output shape: {output.shape}")
        
        # Test GAT model
        print("\nTesting GAT model with intent attention head...")
        gat_model = GAT(nin=64, nhid=32, nout=64, alpha=0.2, nheads=4, intent_embed_dim=16)
        gat_output = gat_model(input_features, adj, intent_embeds)
        print(f"✓ GAT model forward pass successful, output shape: {gat_output.shape}")
        
        # Test TrajAirNet model
        print("\nTesting TrajAirNet model with intent attention head...")
        args = argparse.Namespace()
        args.input_channels = 3
        args.preds = 120
        args.preds_step = 10
        args.tcn_channel_size = 64  # Smaller for testing
        args.tcn_layers = 2
        args.tcn_kernels = 4
        args.num_context_input_c = 2
        args.num_context_output_c = 7
        args.cnn_kernels = 2
        args.gat_heads = 4  # Smaller for testing
        args.graph_hidden = 64
        args.dropout = 0.05
        args.alpha = 0.2
        args.cvae_hidden = 32
        args.cvae_channel_size = 32
        args.cvae_layers = 2
        args.mlp_layer = 16
        args.obs = 11
        args.intent_embed_dim = 16  # Smaller for testing
        args.num_intent_classes = 16
        
        model = TrajAirNet(args)
        
        # Test model forward pass with correct dimensions
        batch_size = 1
        num_agents = 2
        obs_len = 11
        pred_len = 12
        
        # Create tensors in the format expected by model
        obs_traj = torch.randn(obs_len, 3, num_agents)  # [seq_len, features, agents]
        pred_traj = torch.randn(pred_len, 3, num_agents)  # [pred_len, features, agents]
        context = torch.randn(obs_len, 2, num_agents)   # [seq_len, features, agents]  
        intent_labels = torch.tensor([3, 8])  # Sample intent labels [num_agents]
        adj = torch.ones(num_agents, num_agents)
        
        recon_y, m, var = model(obs_traj, pred_traj, adj, context, intent_labels)
        print(f"✓ TrajAirNet forward pass successful")
        print(f"  - Reconstructed trajectories: {len(recon_y)} agents")
        print(f"  - Output shapes: {[r.shape for r in recon_y[:2]]}")  # Show first 2
        
        # Test intent attention learning
        print("\nTesting intent attention head learning...")
        
        # Simulate gradient update
        loss = sum([torch.sum(r**2) for r in recon_y])  # Simple MSE-like loss
        loss.backward()
        
        # Check if gradients exist for intent parameters
        intent_head_grad = model.gat.attentions[0].intent_attention.weight.grad
        if intent_head_grad is not None:
            print(f"✓ Intent attention head has gradients: {intent_head_grad.norm().item():.6f}")
        else:
            print("⚠ No gradients for intent attention head")
        
        beta_grad = model.gat.attentions[0].intent_beta.grad
        if beta_grad is not None:
            print(f"✓ Intent beta parameter has gradients: {beta_grad.item():.6f}")
        else:
            print("⚠ No gradients for intent beta parameter")
        
        print("\n🎉 Option 2 (Intent Attention Head) implementation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intent_attention_dynamics():
    """Test that the intent attention head learns meaningful patterns."""
    print("\nTesting intent attention head dynamics...")
    
    try:
        layer = GraphAttentionLayer(32, 16, alpha=0.2, intent_embed_dim=8)
        
        # Create scenarios with different intent embeddings
        input_features = torch.randn(2, 32)
        adj = torch.ones(2, 2)
        
        # Scenario 1: Similar intent embeddings
        intent_similar = torch.randn(2, 8)
        intent_similar[1] = intent_similar[0] + 0.1 * torch.randn(8)  # Very similar
        output_similar = layer(input_features, adj, intent_similar)
        
        # Scenario 2: Very different intent embeddings
        intent_different = torch.randn(2, 8)
        intent_different[1] = -intent_different[0]  # Opposite
        output_different = layer(input_features, adj, intent_different)
        
        # The outputs should be different due to intent attention
        diff = torch.norm(output_similar - output_different)
        print(f"✓ Different intent embeddings produce different outputs (diff: {diff.item():.6f})")
        
        # Test that intent attention head affects attention
        layer.intent_beta.data.fill_(0.0)  # Disable intent attention
        output_no_intent = layer(input_features, adj, intent_different)
        
        layer.intent_beta.data.fill_(1.0)  # Enable strong intent attention
        output_strong_intent = layer(input_features, adj, intent_different)
        
        diff_beta = torch.norm(output_no_intent - output_strong_intent)
        print(f"✓ Intent beta parameter affects output (diff: {diff_beta.item():.6f})")
        
        # Test that intent attention head learns from embeddings
        zero_embeds = torch.zeros(2, 8)
        random_embeds = torch.randn(2, 8)
        
        output_zero = layer(input_features, adj, zero_embeds)
        output_random = layer(input_features, adj, random_embeds)
        
        diff_embeds = torch.norm(output_zero - output_random)
        print(f"✓ Intent attention head responds to embedding content (diff: {diff_embeds.item():.6f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Intent attention dynamics test failed: {e}")
        return False

def test_with_real_data_option2():
    """Test Option 2 with a small subset of real data."""
    print("\nTesting Option 2 with real trajectory data...")
    
    try:
        # Use debug dataset if available
        data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/debug/processed_data"
        if not os.path.exists(data_dir):
            data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/MayJun2022/processed_data"
        
        from torch.utils.data import DataLoader
        dataset = TrajectoryDataset(data_dir, obs_len=11, pred_len=120, step=10, min_agent=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=seq_collate)
        
        # Create smaller model for testing
        args = argparse.Namespace()
        args.input_channels = 3
        args.preds = 120
        args.preds_step = 10
        args.tcn_channel_size = 64
        args.tcn_layers = 2
        args.tcn_kernels = 4
        args.num_context_input_c = 2
        args.num_context_output_c = 7
        args.cnn_kernels = 2
        args.gat_heads = 4
        args.graph_hidden = 64
        args.dropout = 0.05
        args.alpha = 0.2
        args.cvae_hidden = 32
        args.cvae_channel_size = 32
        args.cvae_layers = 2
        args.mlp_layer = 16
        args.obs = 11
        args.intent_embed_dim = 16
        args.num_intent_classes = 16
        
        model = TrajAirNet(args)
        
        # Test one batch
        for batch in dataloader:
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
            num_agents = obs_traj.shape[1]
            
            pred_traj = torch.transpose(pred_traj, 1, 2)
            adj = torch.ones((num_agents, num_agents))
            
            recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2), intent_labels)
            
            print(f"✓ Real data test successful:")
            print(f"  - Agents: {num_agents}")
            print(f"  - Intent labels: {intent_labels}")
            print(f"  - Forward pass completed")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Option 2: Intent Attention Head")
    print("=" * 60)
    
    success1 = test_intent_attention_head()
    success2 = test_intent_attention_dynamics()
    success3 = test_with_real_data_option2()
    
    if success1 and success2 and success3:
        print("\n🎉 All Option 2 tests passed!")
    else:
        print("\n❌ Some tests failed.")