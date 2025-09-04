"""
End-to-End Test for Belief State Pipeline

This script tests the complete belief-aware trajectory prediction pipeline
from radio call processing to trajectory generation.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.belief_states import BeliefState, BeliefEncoder, test_belief_encoder, VOCAB_SIZE, INTENT_VOCABULARY
from model.belief_manager import BeliefManager, RadioCall, test_belief_manager
from model.llm_belief_updater import LLMBeliefUpdater, test_llm_belief_updater
from model.belief_aware_gat import BeliefAwareGAT, test_belief_aware_gat
from model.belief_trajairnet import BeliefAwareTrajAirNet, test_belief_trajairnet


def test_radio_call_to_belief_pipeline():
    """Test the radio call processing pipeline."""
    print("=" * 60)
    print("TESTING RADIO CALL TO BELIEF PIPELINE")
    print("=" * 60)
    
    # Create components
    belief_manager = BeliefManager("test_pipeline")
    llm_updater = LLMBeliefUpdater()  # Will use fallback mode
    
    # Sample radio call sequence for one aircraft
    radio_calls = [
        RadioCall(1000.0, "N135PL", "Butler county traffic Cherokee 135PL entering left downwind runway 8"),
        RadioCall(1030.0, "N135PL", "Cherokee 135PL turning left base runway 8"),
        RadioCall(1060.0, "N135PL", "Cherokee 135PL final runway 8 touch and go"),
        RadioCall(1090.0, "N135PL", "Cherokee 135PL going around runway 8"),
        RadioCall(1120.0, "N135PL", "Cherokee 135PL departing to the north Butler county"),
    ]
    
    print(f"Processing {len(radio_calls)} radio calls...")
    
    # Process each radio call
    for i, call in enumerate(radio_calls):
        print(f"\nCall {i+1}: {call.transcript}")
        
        # Get previous belief
        prev_belief = belief_manager.get_most_recent_belief(call.tail_number, call.timestamp)
        print(f"Previous belief: {prev_belief.intent_sequence if prev_belief else None}")
        
        # Update belief
        new_belief = llm_updater.update_belief(call, prev_belief)
        belief_manager.aircraft_beliefs[call.tail_number] = belief_manager.aircraft_beliefs.get(call.tail_number, []) + [new_belief]
        
        print(f"Updated belief:  {new_belief.intent_sequence}")
    
    # Test belief retrieval at different times
    print(f"\n--- Belief Retrieval Test ---")
    test_times = [1020.0, 1050.0, 1080.0, 1110.0, 1150.0]
    
    for test_time in test_times:
        belief = belief_manager.get_most_recent_belief("N135PL", test_time)
        print(f"Belief at t={test_time}: {belief.intent_sequence if belief else 'None'}")
    
    print(f"\nRadio call to belief pipeline test: ✓ PASSED")
    return belief_manager


def test_belief_to_trajectory_pipeline():
    """Test belief integration with trajectory prediction."""
    print("\n" + "=" * 60)
    print("TESTING BELIEF TO TRAJECTORY PIPELINE")  
    print("=" * 60)
    
    # Mock args for model creation
    class MockArgs:
        input_channels = 3
        preds = 120
        preds_step = 10
        tcn_channel_size = 64
        tcn_layers = 2
        tcn_kernels = 4
        dropout = 0.05
        num_context_output_c = 7
        obs = 11
        graph_hidden = 64
        gat_heads = 4
        alpha = 0.2
        cvae_layers = 2
        cvae_channel_size = 32
        cvae_hidden = 32
        mlp_layer = 16
        num_context_input_c = 2
        cnn_kernels = 2
        belief_embed_dim = 32
        belief_vocab_size = VOCAB_SIZE
        belief_integration_mode = 'concatenate'
    
    args = MockArgs()
    
    # Create model
    print("Creating BeliefAwareTrajAirNet...")
    model = BeliefAwareTrajAirNet(args)
    
    # Create sample trajectory data
    batch_size = 1
    num_agents = 3
    obs_len = args.obs
    pred_len = int(args.preds / args.preds_step)
    
    x = torch.randn(obs_len, batch_size, num_agents)
    y = torch.randn(pred_len, batch_size, num_agents) 
    context = torch.randn(obs_len, batch_size, num_agents)
    adj = torch.ones(num_agents, num_agents)
    
    # Create belief sequences for each agent
    belief_sequences = [
        [4, 6, 8, 12],     # Agent 1: downwind_8 → base_8 → final_8 → land_8
        [5, 7, 9, 13, 11], # Agent 2: downwind_26 → base_26 → final_26 → land_26 → takeoff_26
        [INTENT_VOCABULARY['unknown']]               # Agent 3: unknown
    ]
    belief_lengths = torch.tensor([4, 5, 1], dtype=torch.long)
    
    print(f"Input shapes:")
    print(f"  Trajectories: x={x.shape}, y={y.shape}")
    print(f"  Context: {context.shape}")
    print(f"  Belief sequences: {[len(seq) for seq in belief_sequences]}")
    
    # Test forward pass with beliefs
    print(f"\nTesting forward pass with beliefs...")
    recon_y, means, variances = model(x, y, adj, context, belief_sequences, belief_lengths)
    
    print(f"✓ Forward pass successful")
    print(f"  Output trajectories: {len(recon_y)}")
    print(f"  Trajectory shape: {recon_y[0].shape}")
    
    # Test forward pass without beliefs
    print(f"\nTesting forward pass without beliefs...")
    recon_y_no_belief, _, _ = model(x, y, adj, context)
    
    print(f"✓ Forward pass without beliefs successful")
    
    # Compare outputs
    diff = torch.mean(torch.abs(recon_y[0] - recon_y_no_belief[0])).item()
    print(f"  Difference with/without beliefs: {diff:.6f}")
    
    if diff > 1e-6:
        print(f"✓ Belief information affects model output")
    else:
        print(f"⚠ Belief information has minimal effect")
    
    # Test inference
    print(f"\nTesting inference...")
    z = torch.randn(1, 1, args.cvae_hidden)
    pred_y = model.inference(x, z, adj, context, belief_sequences, belief_lengths)
    
    print(f"✓ Inference successful")
    print(f"  Prediction shape: {pred_y[0].shape}")
    
    print(f"\nBelief to trajectory pipeline test: ✓ PASSED")
    return model


def test_full_integration():
    """Test the complete integrated pipeline."""
    print("\n" + "=" * 60)
    print("TESTING FULL INTEGRATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Create belief manager with radio calls
    belief_manager = test_radio_call_to_belief_pipeline()
    
    # Step 2: Test trajectory prediction
    model = test_belief_to_trajectory_pipeline()
    
    # Step 3: Test realistic scenario
    print(f"\n--- Realistic Scenario Test ---")
    
    # Simulate a trajectory sequence with multiple aircraft
    sequence_timestamp = 1100.0  # Between some of our radio calls
    tail_numbers = ["N135PL", "N456AB", "N789CD"]  # Mix of known and unknown aircraft
    
    # Get beliefs for all aircraft at sequence time
    beliefs = belief_manager.get_belief_at_sequence_time(tail_numbers, sequence_timestamp)
    print(f"Aircraft beliefs at t={sequence_timestamp}:")
    
    belief_sequences = []
    belief_lengths = []
    
    for i, (tail, belief) in enumerate(zip(tail_numbers, beliefs)):
        if belief:
            indices = belief.to_indices()
            print(f"  {tail}: {belief.intent_sequence} → {indices}")
            belief_sequences.append(indices)
            belief_lengths.append(len(indices))
        else:
            print(f"  {tail}: No belief available → [{INTENT_VOCABULARY['unknown']}] (unknown)")
            belief_sequences.append([INTENT_VOCABULARY['unknown']])
            belief_lengths.append(1)
    
    belief_lengths = torch.tensor(belief_lengths, dtype=torch.long)
    
    # Create mock trajectory data for these aircraft
    num_agents = len(tail_numbers)
    x = torch.randn(11, 1, num_agents)
    context = torch.randn(11, 1, num_agents)
    adj = torch.ones(num_agents, num_agents)
    z = torch.randn(1, 1, 32)
    
    # Generate predictions with beliefs
    predictions = model.inference(x, z, adj, context, belief_sequences, belief_lengths)
    
    print(f"\n✓ Generated trajectory predictions for {num_agents} aircraft")
    print(f"  Prediction shapes: {[pred.shape for pred in predictions]}")
    
    print(f"\nFull integration test: ✓ PASSED")


def main():
    """Run all pipeline tests."""
    print("BELIEF STATE PIPELINE TESTING")
    print("=" * 60)
    print("Testing all components of the belief-aware trajectory prediction system\n")
    
    try:
        # Test individual components first
        print("1. Testing BeliefEncoder...")
        test_belief_encoder()
        print("   ✓ BeliefEncoder working\n")
        
        print("2. Testing BeliefManager...")
        test_belief_manager()
        print("   ✓ BeliefManager working\n")
        
        print("3. Testing LLMBeliefUpdater...")
        test_llm_belief_updater()
        print("   ✓ LLMBeliefUpdater working\n")
        
        print("4. Testing BeliefAwareGAT...")
        test_belief_aware_gat()
        print("   ✓ BeliefAwareGAT working\n")
        
        print("5. Testing BeliefAwareTrajAirNet...")
        test_belief_trajairnet()
        print("   ✓ BeliefAwareTrajAirNet working\n")
        
        # Test full integration
        print("6. Testing Full Integration...")
        test_full_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("The belief-aware trajectory prediction pipeline is ready for training.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)