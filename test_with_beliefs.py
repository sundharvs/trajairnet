"""
Test Script for Belief-Aware TrajAirNet

This script evaluates the belief-aware trajectory prediction model.
"""

import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

from model.belief_trajairnet import BeliefAwareTrajAirNet
from model.utils import TrajectoryDataset
from train_with_beliefs import BeliefTrajectoryDataset, belief_collate, create_belief_manager_from_transcripts


def ade(pred_traj, true_traj):
    """Average Displacement Error"""
    return np.mean(np.linalg.norm(pred_traj - true_traj, axis=1))


def fde(pred_traj, true_traj):
    """Final Displacement Error"""
    return np.linalg.norm(pred_traj[-1] - true_traj[-1])


def test(model, loader_test, device):
    """
    Test the belief-aware model.
    
    Args:
        model: BeliefAwareTrajAirNet model
        loader_test: Test data loader
        device: Torch device
        
    Returns:
        average_ade: Average ADE across all test samples
        average_fde: Average FDE across all test samples
    """
    model.eval()
    
    total_ade = 0
    total_fde = 0
    tot_batch = 0
    
    loss_records = []
    
    for batch_idx, batch in enumerate(tqdm(loader_test)):
        tot_batch += 1
        batch = [tensor.to(device) for tensor in batch]
        
        # Unpack batch with belief data
        obs_traj_all, pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, timestamp, tail, belief_padded, belief_lengths = batch
        num_agents = obs_traj_all.shape[1]
        
        best_ade_loss = float('inf')
        best_fde_loss = float('inf')
        
        # Generate multiple predictions and keep the best
        for i in range(1):  # Can increase for better predictions
            z = torch.randn([1, 1, 128]).to(device)
            
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
            
            # Generate predictions
            recon_y_all = model.inference(
                torch.transpose(obs_traj_all, 1, 2), z, adj,
                torch.transpose(context, 1, 2), belief_sequences, agent_belief_lengths
            )
            
            # Calculate metrics
            ade_loss = 0
            fde_loss = 0
            for agent in range(num_agents):
                obs_traj = np.squeeze(obs_traj_all[:, agent, :].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:, agent, :].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                
                ade_loss += ade(recon_pred, pred_traj)
                fde_loss += fde(recon_pred, pred_traj)
            
            ade_loss = ade_loss / num_agents
            fde_loss = fde_loss / num_agents
            
            if ade_loss < best_ade_loss:
                best_ade_loss = ade_loss
                best_fde_loss = fde_loss
        
        total_ade += best_ade_loss
        total_fde += best_fde_loss
        
        loss_records.append({
            'batch_idx': batch_idx,
            'ade': best_ade_loss,
            'fde': best_fde_loss,
            'num_agents': num_agents
        })
    
    average_ade = total_ade / tot_batch
    average_fde = total_fde / tot_batch
    
    return average_ade, average_fde


def main():
    """Main testing function."""
    
    parser = argparse.ArgumentParser(description='Test Belief-Aware TrajAirNet model')
    
    # Dataset params
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--skip', type=int, default=1)
    
    # Model params
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
    
    # Belief params
    parser.add_argument('--belief_embed_dim', type=int, default=64)
    parser.add_argument('--belief_vocab_size', type=int, default=35)
    parser.add_argument('--belief_integration_mode', type=str, default='concatenate')
    parser.add_argument('--transcripts_path', type=str, 
                       default='../main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv')
    
    # Test params
    parser.add_argument('--model_dir', type=str, default='/saved_models/')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch to load the model')
    
    args = parser.parse_args()
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    datapath = args.dataset_folder + args.dataset_name + "/processed_data/"
    
    print("Loading Test Data from", datapath + "test")
    trajectory_test = TrajectoryDataset(datapath + "test", obs_len=args.obs,
                                      pred_len=args.preds, step=args.preds_step, 
                                      delim=args.delim, skip=args.skip)
    
    # Create belief manager
    belief_manager = create_belief_manager_from_transcripts(args.transcripts_path)
    
    # Create belief-aware dataset
    dataset_test = BeliefTrajectoryDataset(trajectory_test, belief_manager)
    loader_test = DataLoader(dataset_test, batch_size=1, num_workers=4,
                           shuffle=False, collate_fn=belief_collate)
    
    # Load model
    model = BeliefAwareTrajAirNet(args)
    model.to(device)
    
    # Load trained weights
    model_path = os.getcwd() + args.model_dir + "belief_model_" + args.dataset_name + "_" + str(args.epoch) + ".pt"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Starting evaluation...")
    
    # Test the model
    test_ade_loss, test_fde_loss = test(model, loader_test, device)
    
    print(f"Test Results:")
    print(f"  ADE: {test_ade_loss:.4f}")
    print(f"  FDE: {test_fde_loss:.4f}")


if __name__ == '__main__':
    main()