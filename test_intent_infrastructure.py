#!/usr/bin/env python3
"""
Test script for intent infrastructure in base branch.
Tests that all components load and work together.
"""

import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.utils import TrajectoryDataset, seq_collate, IntentLookup
from model.trajairnet import TrajAirNet

def test_intent_lookup():
    """Test intent lookup functionality."""
    print("Testing IntentLookup class...")
    
    try:
        # Initialize intent lookup
        intent_lookup = IntentLookup()
        print("✓ IntentLookup initialization successful")
        
        # Test intent lookup for a sample tail and timestamp
        test_intent = intent_lookup.get_most_recent_intent("N651TW", 1651750000.0)
        print(f"✓ Sample intent lookup returned: {test_intent}")
        
        return True
    except Exception as e:
        print(f"✗ IntentLookup test failed: {e}")
        return False

def test_trajectory_dataset():
    """Test trajectory dataset with intent labels."""
    print("Testing TrajectoryDataset with intent labels...")
    
    try:
        # Test with debug dataset if available
        data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/debug/processed_data"
        if not os.path.exists(data_dir):
            data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/MayJun2022/processed_data"
        
        dataset = TrajectoryDataset(data_dir, obs_len=11, pred_len=120, step=10, min_agent=0)
        print(f"✓ TrajectoryDataset loaded with {len(dataset)} sequences")
        
        # Test data loading
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample data shape: {[x.shape if hasattr(x, 'shape') else len(x) for x in sample]}")
            print(f"✓ Intent labels in sample: {sample[5]}")  # Intent labels at index 5
        
        return True
    except Exception as e:
        print(f"✗ TrajectoryDataset test failed: {e}")
        return False

def test_model_creation():
    """Test TrajAirNet model creation with intent embedding."""
    print("Testing TrajAirNet model creation...")
    
    try:
        # Create mock arguments
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
        args.intent_embed_dim = 32
        args.num_intent_classes = 16
        
        model = TrajAirNet(args)
        print("✓ TrajAirNet model creation successful")
        
        # Check if intent embedding layer exists
        if hasattr(model, 'intent_embedding'):
            print(f"✓ Intent embedding layer: {model.intent_embedding}")
        else:
            print("✗ Intent embedding layer missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ TrajAirNet model test failed: {e}")
        return False

def test_data_collation():
    """Test data collation with intent labels."""
    print("Testing data collation with intent labels...")
    
    try:
        # Create mock data
        batch_size = 2
        num_agents = 3
        obs_len = 11
        pred_len = 12  # 120/10
        
        # Mock trajectory data
        obs_traj = torch.randn(num_agents, 3, obs_len)
        pred_traj = torch.randn(num_agents, 3, pred_len)
        context = torch.randn(num_agents, 2, obs_len)
        intent_labels = torch.randint(1, 17, (num_agents,))  # Random intent labels 1-16
        timestamp = torch.randn(num_agents, 1, obs_len)
        tail = torch.randn(num_agents, 1, obs_len)
        
        # Create batch data
        batch_data = [
            (obs_traj, pred_traj, obs_traj, pred_traj, context, intent_labels, timestamp, tail),
            (obs_traj, pred_traj, obs_traj, pred_traj, context, intent_labels, timestamp, tail)
        ]
        
        collated = seq_collate(batch_data)
        print(f"✓ Collation successful, output shapes: {[x.shape for x in collated]}")
        
        return True
    except Exception as e:
        print(f"✗ Data collation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Intent Infrastructure Base Implementation")
    print("=" * 60)
    
    tests = [
        test_intent_lookup,
        test_trajectory_dataset, 
        test_model_creation,
        test_data_collation
    ]
    
    passed = 0
    for test_func in tests:
        print(f"\n{'-' * 40}")
        if test_func():
            passed += 1
        print("")
    
    print("=" * 60)
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    if passed == len(tests):
        print("🎉 All tests passed! Base infrastructure is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    main()