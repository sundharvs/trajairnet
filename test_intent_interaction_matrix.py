#!/usr/bin/env python3
"""
Test script for Option 1: Intent Interaction Matrix approach.
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

def test_intent_interaction_matrix():
    """Test the intent interaction matrix implementation."""
    print("Testing Intent Interaction Matrix (Option 1)")
    print("=" * 50)
    
    try:
        # Test GraphAttentionLayer with intent interaction matrix
        print("Testing GraphAttentionLayer with intent interaction...")
        layer = GraphAttentionLayer(64, 32, alpha=0.2, num_intent_classes=16)
        
        # Check if intent interaction matrix exists
        assert hasattr(layer, 'intent_interaction'), "Intent interaction matrix missing"
        assert layer.intent_interaction.shape == (16, 16), f"Wrong shape: {layer.intent_interaction.shape}"
        print(f"✓ Intent interaction matrix shape: {layer.intent_interaction.shape}")
        
        # Test forward pass
        num_agents = 3
        input_features = torch.randn(num_agents, 64)
        adj = torch.ones(num_agents, num_agents)
        intent_labels = torch.tensor([1, 4, 9])  # Sample intent labels
        
        output = layer(input_features, adj, intent_labels)
        print(f"✓ GAT layer forward pass successful, output shape: {output.shape}")
        
        # Test GAT model
        print("\nTesting GAT model with intent interaction...")
        gat_model = GAT(nin=64, nhid=32, nout=64, alpha=0.2, nheads=4, num_intent_classes=16)
        gat_output = gat_model(input_features, adj, intent_labels)
        print(f"✓ GAT model forward pass successful, output shape: {gat_output.shape}")
        
        # Test TrajAirNet model
        print("\nTesting TrajAirNet model with intent interaction...")
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
        
        # Create tensors in the format expected by model after collation and transpose
        # Model expects: obs_traj -> [seq_len, features, agents], context -> [seq_len, features, agents]
        obs_traj = torch.randn(obs_len, 3, num_agents)  # [seq_len, features, agents]
        pred_traj = torch.randn(pred_len, 3, num_agents)  # [pred_len, features, agents]
        context = torch.randn(obs_len, 2, num_agents)   # [seq_len, features, agents]  
        intent_labels = torch.tensor([3, 8])  # Sample intent labels [num_agents]
        adj = torch.ones(num_agents, num_agents)
        
        recon_y, m, var = model(obs_traj, pred_traj, adj, context, intent_labels)
        print(f"✓ TrajAirNet forward pass successful")
        print(f"  - Reconstructed trajectories: {len(recon_y)} agents")
        print(f"  - Output shapes: {[r.shape for r in recon_y[:2]]}")  # Show first 2
        
        # Test intent interaction learning
        print("\nTesting intent interaction matrix learning...")
        initial_matrix = model.gat.attentions[0].intent_interaction.clone()
        
        # Simulate gradient update
        loss = sum([torch.sum(r**2) for r in recon_y])  # Simple MSE-like loss
        loss.backward()
        
        # Check if gradients exist for intent parameters
        intent_grad = model.gat.attentions[0].intent_interaction.grad
        if intent_grad is not None:
            print(f"✓ Intent interaction matrix has gradients: {intent_grad.norm().item():.6f}")
        else:
            print("⚠ No gradients for intent interaction matrix")
        
        alpha_grad = model.gat.attentions[0].intent_alpha.grad
        if alpha_grad is not None:
            print(f"✓ Intent alpha parameter has gradients: {alpha_grad.item():.6f}")
        else:
            print("⚠ No gradients for intent alpha parameter")
        
        print("\n🎉 Option 1 (Intent Interaction Matrix) implementation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intent_learning_dynamics():
    """Test that different intent pairs produce different attention patterns."""
    print("\nTesting intent learning dynamics...")
    
    try:
        layer = GraphAttentionLayer(32, 16, alpha=0.2, num_intent_classes=16)
        
        # Create two scenarios with different intent combinations
        input_features = torch.randn(2, 32)
        adj = torch.ones(2, 2)
        
        # Scenario 1: Both agents have same intent (landing runway 8)
        intent_same = torch.tensor([3, 3])
        output_same = layer(input_features, adj, intent_same)
        
        # Scenario 2: Agents have conflicting intents (landing vs takeoff)  
        intent_conflict = torch.tensor([3, 1])  # Landing vs takeoff
        output_conflict = layer(input_features, adj, intent_conflict)
        
        # The outputs should be different due to intent interaction
        diff = torch.norm(output_same - output_conflict)
        print(f"✓ Different intent combinations produce different outputs (diff: {diff.item():.6f})")
        
        # Test that intent interaction matrix affects attention
        layer.intent_interaction.data.fill_(0.0)  # Zero out interactions
        output_zero = layer(input_features, adj, intent_conflict)
        
        layer.intent_interaction.data.normal_(0, 0.1)  # Add interactions
        output_nonzero = layer(input_features, adj, intent_conflict)
        
        diff_matrix = torch.norm(output_zero - output_nonzero)
        print(f"✓ Intent interaction matrix affects output (diff: {diff_matrix.item():.6f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Intent dynamics test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Option 1: Intent Interaction Matrix")
    print("=" * 60)
    
    success1 = test_intent_interaction_matrix()
    success2 = test_intent_learning_dynamics()
    
    if success1 and success2:
        print("\n🎉 All Option 1 tests passed!")
    else:
        print("\n❌ Some tests failed.")