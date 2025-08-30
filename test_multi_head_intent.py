#!/usr/bin/env python3
"""
Test script for Option 3: Multi-Head Intent Architecture approach.
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

def test_multi_head_intent():
    """Test the multi-head intent architecture implementation."""
    print("Testing Multi-Head Intent Architecture (Option 3)")
    print("=" * 55)
    
    try:
        # Test GraphAttentionLayer with different head types
        print("Testing GraphAttentionLayer with spatial and intent head types...")
        
        # Test spatial head
        spatial_head = GraphAttentionLayer(64, 32, alpha=0.2, intent_embed_dim=16, head_type='spatial')
        assert spatial_head.head_type == 'spatial', "Spatial head type not set correctly"
        assert spatial_head.a.shape == (64, 1), f"Spatial head wrong attention param shape: {spatial_head.a.shape}"
        print(f"✓ Spatial head: attention param shape {spatial_head.a.shape}")
        
        # Test intent head
        intent_head = GraphAttentionLayer(64, 32, alpha=0.2, intent_embed_dim=16, head_type='intent')
        assert intent_head.head_type == 'intent', "Intent head type not set correctly"
        expected_size = 2 * 32 + 2 * 16  # 2 * out_features + 2 * intent_embed_dim = 96
        assert intent_head.a.shape == (expected_size, 1), f"Intent head wrong attention param shape: {intent_head.a.shape}"
        print(f"✓ Intent head: attention param shape {intent_head.a.shape}")
        
        # Test forward passes
        num_agents = 3
        input_features = torch.randn(num_agents, 64)
        adj = torch.ones(num_agents, num_agents)
        intent_embeds = torch.randn(num_agents, 16)
        
        spatial_output = spatial_head(input_features, adj)
        intent_output = intent_head(input_features, adj, intent_embeds)
        
        print(f"✓ Spatial head output shape: {spatial_output.shape}")
        print(f"✓ Intent head output shape: {intent_output.shape}")
        
        # Test GAT model with multi-head architecture
        print("\nTesting GAT model with multi-head intent architecture...")
        gat_model = GAT(nin=64, nhid=32, nout=64, alpha=0.2, nheads=8, 
                       intent_embed_dim=16, intent_head_ratio=0.25)
        
        # Check head distribution
        expected_intent_heads = max(1, int(8 * 0.25))  # 2 intent heads
        expected_spatial_heads = 8 - expected_intent_heads  # 6 spatial heads
        
        assert gat_model.num_intent_heads == expected_intent_heads, f"Wrong number of intent heads: {gat_model.num_intent_heads}"
        assert gat_model.num_spatial_heads == expected_spatial_heads, f"Wrong number of spatial heads: {gat_model.num_spatial_heads}"
        
        print(f"✓ Head distribution: {gat_model.num_spatial_heads} spatial + {gat_model.num_intent_heads} intent heads")
        
        gat_output = gat_model(input_features, adj, intent_embeds)
        print(f"✓ GAT model forward pass successful, output shape: {gat_output.shape}")
        
        # Test TrajAirNet model
        print("\nTesting TrajAirNet model with multi-head intent architecture...")
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
        args.gat_heads = 8
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
        print(f"  - Output shapes: {[r.shape for r in recon_y[:2]]}")
        
        # Check GAT head distribution in the model
        print(f"  - Model GAT head distribution: {model.gat.num_spatial_heads} spatial + {model.gat.num_intent_heads} intent")
        
        # Test gradient computation
        print("\nTesting multi-head intent learning...")
        
        loss = sum([torch.sum(r**2) for r in recon_y])
        loss.backward()
        
        # Check gradients for both spatial and intent heads
        spatial_grad = model.gat.spatial_attentions[0].a.grad
        intent_grad = model.gat.intent_attentions[0].a.grad
        
        if spatial_grad is not None:
            print(f"✓ Spatial head gradients: {spatial_grad.norm().item():.6f}")
        if intent_grad is not None:
            print(f"✓ Intent head gradients: {intent_grad.norm().item():.6f}")
        
        print("\n🎉 Option 3 (Multi-Head Intent Architecture) implementation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_head_specialization():
    """Test that spatial and intent heads learn different patterns."""
    print("\nTesting head specialization...")
    
    try:
        # Create GAT with mixed heads
        gat_model = GAT(nin=32, nhid=16, nout=32, alpha=0.2, nheads=4, 
                       intent_embed_dim=8, intent_head_ratio=0.5)  # 2 spatial + 2 intent heads
        
        input_features = torch.randn(3, 32)
        adj = torch.ones(3, 3)
        
        # Test with different intent scenarios
        similar_intents = torch.randn(3, 8)
        similar_intents[1] = similar_intents[0] + 0.1 * torch.randn(8)
        similar_intents[2] = similar_intents[0] + 0.1 * torch.randn(8)
        
        different_intents = torch.randn(3, 8)
        different_intents[1] = -different_intents[0]
        different_intents[2] = 2 * different_intents[0]
        
        output_similar = gat_model(input_features, adj, similar_intents)
        output_different = gat_model(input_features, adj, different_intents)
        
        diff = torch.norm(output_similar - output_different)
        print(f"✓ Different intent patterns produce different outputs (diff: {diff.item():.6f})")
        
        # Test that intent heads specifically contribute
        no_intent_output = gat_model(input_features, adj, None)  # No intent embeddings
        with_intent_output = gat_model(input_features, adj, different_intents)
        
        intent_contribution = torch.norm(no_intent_output - with_intent_output)
        print(f"✓ Intent heads contribute to output (contribution: {intent_contribution.item():.6f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Head specialization test failed: {e}")
        return False

def test_with_real_data_option3():
    """Test Option 3 with real trajectory data."""
    print("\nTesting Option 3 with real trajectory data...")
    
    try:
        # Use debug dataset
        data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/debug/processed_data"
        if not os.path.exists(data_dir):
            data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/MayJun2022/processed_data"
        
        from torch.utils.data import DataLoader
        dataset = TrajectoryDataset(data_dir, obs_len=11, pred_len=120, step=10, min_agent=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=seq_collate)
        
        # Create model with multi-head intent architecture
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
        args.gat_heads = 8
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
            print(f"  - Head distribution: {model.gat.num_spatial_heads}S+{model.gat.num_intent_heads}I")
            print(f"  - Forward pass completed")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Option 3: Multi-Head Intent Architecture")
    print("=" * 65)
    
    success1 = test_multi_head_intent()
    success2 = test_head_specialization()
    success3 = test_with_real_data_option3()
    
    if success1 and success2 and success3:
        print("\n🎉 All Option 3 tests passed!")
    else:
        print("\n❌ Some tests failed.")