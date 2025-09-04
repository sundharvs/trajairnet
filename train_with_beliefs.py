"""
Training Script for Belief-Aware TrajAirNet

This script extends the original training pipeline to incorporate belief states
from radio communications for enhanced trajectory prediction.
"""

import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
from typing import List, Optional, Tuple
import numpy as np

# Import belief-aware components
from model.belief_trajairnet import BeliefAwareTrajAirNet
from model.utils import TrajectoryDataset, seq_collate, loss_func
from model.belief_manager import BeliefManager, RadioCall
from model.llm_belief_updater import LLMBeliefUpdater
from model.belief_states import pad_belief_sequences, VOCAB_SIZE, INTENT_VOCABULARY

# Import test function
from test_with_beliefs import test


class BeliefTrajectoryDataset:
    """
    Enhanced dataset that combines trajectory data with belief states.
    
    This class integrates the TrajectoryDataset with BeliefManager to provide
    belief information alongside trajectory sequences for training.
    """
    
    def __init__(self, trajectory_dataset: TrajectoryDataset, belief_manager: BeliefManager):
        """
        Args:
            trajectory_dataset: Standard trajectory dataset
            belief_manager: Manager with radio call belief states
        """
        self.trajectory_dataset = trajectory_dataset
        self.belief_manager = belief_manager
        
    def __len__(self):
        return len(self.trajectory_dataset)
    
    def __getitem__(self, index):
        # Get standard trajectory data
        trajectory_data = self.trajectory_dataset[index]
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, timestamp, tail = trajectory_data
        
        # Extract belief information
        belief_sequences, belief_lengths = self._get_belief_data(timestamp, tail, index)
        
        # Return enhanced data tuple
        return obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, timestamp, tail, belief_sequences, belief_lengths
    
    def _get_belief_data(self, timestamp: torch.Tensor, tail: torch.Tensor, seq_index: int) -> Tuple[List[List[int]], torch.Tensor]:
        """
        Extract belief sequences for agents in this trajectory sequence.
        
        Args:
            timestamp: [num_agents, 1, seq_len] timestamp data
            tail: [num_agents, 1, seq_len] tail number data  
            seq_index: Index of this sequence
            
        Returns:
            belief_sequences: List of belief sequences for each agent
            belief_lengths: Tensor of sequence lengths
        """
        num_agents = timestamp.shape[0]
        belief_sequences = []
        lengths = []
        
        # Get sequence timestamp (end of observation period)
        seq_timestamp = timestamp[0, 0, -1].item()  # Use last observed timestamp
        
        for agent_idx in range(num_agents):
            # Convert tail number from base-36 back to string
            tail_numeric = int(tail[agent_idx, 0, -1].item())
            tail_str = np.base_repr(tail_numeric, base=36) if tail_numeric > 0 else "UNKNOWN"
            
            # Get belief state for this agent at sequence time
            belief_state = self.belief_manager.get_most_recent_belief(tail_str, seq_timestamp)
            
            if belief_state and belief_state.intent_sequence:
                # Convert belief to indices
                belief_indices = belief_state.to_indices()
                belief_sequences.append(belief_indices)
                lengths.append(len(belief_indices))
            else:
                # No belief available - use unknown intent
                belief_sequences.append([INTENT_VOCABULARY['unknown']])  # 'unknown' intent
                lengths.append(1)
        
        return belief_sequences, torch.tensor(lengths, dtype=torch.long)


def belief_collate(data):
    """
    Custom collate function for belief-aware trajectory data.
    
    Extends the standard seq_collate to handle belief sequences.
    """
    # Unpack data
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, 
     context_list, timestamp_list, tail_list, belief_seq_list, belief_len_list) = zip(*data)
    
    # Use standard collate for trajectory data
    trajectory_batch = seq_collate(list(zip(
        obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
        context_list, timestamp_list, tail_list
    )))
    
    # Handle belief sequences - flatten and pad
    all_belief_sequences = []
    all_belief_lengths = []
    
    for seq_beliefs, seq_lengths in zip(belief_seq_list, belief_len_list):
        all_belief_sequences.extend(seq_beliefs)
        all_belief_lengths.extend(seq_lengths.tolist())
    
    # Pad belief sequences for batch processing
    padded_beliefs, padded_lengths = pad_belief_sequences(all_belief_sequences)
    
    # Combine trajectory batch with belief data
    return trajectory_batch + (padded_beliefs, torch.tensor(all_belief_lengths, dtype=torch.long))


def create_belief_manager_from_transcripts(transcript_path: str) -> BeliefManager:
    """
    Create BeliefManager from existing transcript data.
    
    Args:
        transcript_path: Path to transcripts_with_goals.csv
        
    Returns:
        Populated BeliefManager
    """
    belief_manager = BeliefManager("belief_cache")
    
    if not os.path.exists(transcript_path):
        print(f"Warning: Transcript file {transcript_path} not found. Using empty belief manager.")
        return belief_manager
    
    # Load transcript data
    df = pd.read_csv(transcript_path)
    print(f"Loading {len(df)} radio calls from {transcript_path}")
    
    # Create LLM updater (will fallback to rule-based if LLMs unavailable)
    llm_updater = LLMBeliefUpdater()
    
    # Process each radio call
    for idx, row in tqdm(df.iterrows(), desc="Processing radio calls", total=len(df)):
        try:
            # Create radio call object
            radio_call = RadioCall(
                timestamp=pd.to_datetime(row['start_time']).timestamp(),
                tail_number=row['speaker_tail'],
                transcript=row['whisper_transcript']
            )
            
            # Update belief state
            belief_manager.process_radio_call(radio_call, llm_updater.update_belief)
            
        except Exception as e:
            print(f"Warning: Failed to process radio call {idx}: {e}")
            continue
    
    # Print statistics
    stats = belief_manager.get_statistics()
    print(f"Belief Manager Statistics:")
    print(f"  Total aircraft: {stats['total_aircraft']}")
    print(f"  Total beliefs: {stats['total_beliefs']}")
    print(f"  Average belief length: {stats['average_belief_length']:.2f}")
    
    return belief_manager


def train():
    """Main training function with belief integration."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Belief-Aware TrajAirNet model')
    
    # Dataset params
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    
    # Network params
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--tcn_channel_size', type=int, default=256)
    parser.add_argument('--tcn_layers', type=int, default=2)
    parser.add_argument('--tcn_kernels', type=int, default=4)
    
    parser.add_argument('--num_context_input_c', type=int, default=2)
    parser.add_argument('--num_context_output_c', type=int, default=7)
    parser.add_argument('--cnn_kernels', type=int, default=2)
    
    parser.add_argument('--gat_heads', type=int, default=16)
    parser.add_argument('--graph_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--cvae_hidden', type=int, default=128)
    parser.add_argument('--cvae_channel_size', type=int, default=128)
    parser.add_argument('--cvae_layers', type=int, default=2)
    parser.add_argument('--mlp_layer', type=int, default=32)
    
    # Belief-specific params (NEW)
    parser.add_argument('--belief_embed_dim', type=int, default=64)
    parser.add_argument('--belief_vocab_size', type=int, default=VOCAB_SIZE)
    parser.add_argument('--belief_integration_mode', type=str, default='concatenate',
                       choices=['concatenate', 'add', 'gated'])
    parser.add_argument('--transcripts_path', type=str, 
                       default='../main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv')
    
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--total_epochs', type=int, default=50)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--model_pth', type=str, default="/saved_models/")
    
    args = parser.parse_args()
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trajectory data
    datapath = args.dataset_folder + args.dataset_name + "/processed_data/"
    
    print("Loading Train Data from", datapath + "train")
    trajectory_train = TrajectoryDataset(datapath + "train", obs_len=args.obs, 
                                       pred_len=args.preds, step=args.preds_step, delim=args.delim)
    
    print("Loading Test Data from", datapath + "test")  
    trajectory_test = TrajectoryDataset(datapath + "test", obs_len=args.obs,
                                      pred_len=args.preds, step=args.preds_step, delim=args.delim)
    
    # Create belief manager from radio call transcripts
    belief_manager = create_belief_manager_from_transcripts(args.transcripts_path)
    
    # Create belief-aware datasets
    dataset_train = BeliefTrajectoryDataset(trajectory_train, belief_manager)
    dataset_test = BeliefTrajectoryDataset(trajectory_test, belief_manager)
    
    # Create data loaders with custom collate function
    loader_train = DataLoader(dataset_train, batch_size=1, num_workers=4, 
                            shuffle=True, collate_fn=belief_collate)
    loader_test = DataLoader(dataset_test, batch_size=1, num_workers=4,
                           shuffle=True, collate_fn=belief_collate)
    
    # Create belief-aware model
    model = BeliefAwareTrajAirNet(args)
    model.to(device)
    print(f"Created BeliefAwareTrajAirNet with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    num_batches = len(loader_train)
    print("Starting Training....")
    
    for epoch in range(1, args.total_epochs + 1):
        model.train()
        loss_batch = 0
        batch_count = 0
        tot_batch_count = 0
        tot_loss = 0
        
        for batch in tqdm(loader_train):
            batch_count += 1
            tot_batch_count += 1
            batch = [tensor.to(device) for tensor in batch]
            
            # Unpack batch with belief data
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, timestamp, tail, belief_padded, belief_lengths = batch
            
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            adj = torch.ones((num_agents, num_agents))
            
            # Convert padded belief data back to sequences
            belief_sequences = []
            start_idx = 0
            for agent_idx in range(num_agents):
                length = belief_lengths[start_idx].item()
                seq = belief_padded[start_idx][:length].tolist()
                belief_sequences.append(seq)
                start_idx += 1
            
            agent_belief_lengths = belief_lengths[:num_agents]
            
            optimizer.zero_grad()
            
            # Forward pass with beliefs
            recon_y, m, var = model(
                torch.transpose(obs_traj, 1, 2), pred_traj, adj[0],
                torch.transpose(context, 1, 2), belief_sequences, agent_belief_lengths
            )
            
            loss = 0
            for agent in range(num_agents):
                loss += loss_func(
                    recon_y[agent],
                    torch.transpose(pred_traj[:, :, agent], 0, 1).unsqueeze(0),
                    m[agent], var[agent]
                )
            
            loss_batch += loss
            tot_loss += loss.item()
            
            if batch_count > 8:
                loss_batch.backward()
                optimizer.step()
                loss_batch = 0
                batch_count = 0
        
        print("EPOCH:", epoch, "Train Loss:", loss)
        
        # Save model
        if args.save_model:
            loss = tot_loss / tot_batch_count
            model_path = os.getcwd() + args.model_pth + "belief_model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            print("Saving model at", model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'args': args
            }, model_path)
        
        # Evaluate
        if args.evaluate:
            print("Starting Testing....")
            model.eval()
            test_ade_loss, test_fde_loss = test(model, loader_test, device)
            print("EPOCH:", epoch, "Train Loss:", loss, "Test ADE Loss:", test_ade_loss, "Test FDE Loss:", test_fde_loss)


if __name__ == '__main__':
    train()