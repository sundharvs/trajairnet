import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim
import wandb

from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate, loss_func
from test import test

# Default config dictionary (everything that doesn't change in sweeps)
DEFAULT_CONFIG = {
    # Dataset params
    'dataset_folder': '../dataset/',
    'dataset_name': '7daysJune',
    'obs': 11,
    'preds': 120,
    'preds_step': 10,

    # Network params
    'input_channels': 3,
    'tcn_channel_size': 256,
    'tcn_layers': 2,
    'tcn_kernels': 4,
    'num_context_input_c': 2,
    'num_context_output_c': 7,
    'cnn_kernels': 2,
    'gat_heads': 16,
    'graph_hidden': 256,
    'dropout': 0.05,
    'alpha': 0.2,
    'cvae_hidden': 128,
    'cvae_channel_size': 128,
    'cvae_layers': 2,
    'mlp_layer': 32,

    # Intent embedding
    'intent_embed_dim': 32,
    'num_intent_classes': 16,

    # Optimization
    'lr': 0.001,
    'total_epochs': 20,

    # Other
    'delim': ' ',
    'evaluate': True,
    'save_model': True,
    'model_pth': '/saved_models/',

    # Wandb
    'use_wandb': True,
    'wandb_project': 'trajairnet-intent-interaction-matrix',
    'wandb_entity': None
}


def train_with_config(config=None):
    """
    Train function that can be called with a config dictionary for sweeps.
    It merges DEFAULT_CONFIG with sweep overrides.
    """
    # Start from defaults
    merged_config = DEFAULT_CONFIG.copy()

    # If user provides a dict (sweep), overlay keys
    if config is not None:
        merged_config.update(config)

    # Convert to argparse.Namespace for legacy code
    args = argparse.Namespace(**merged_config)

    # Initialize wandb (only if not already initialized by sweep)
    if args.use_wandb and not wandb.run:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   config=vars(args))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    datapath = os.path.join(args.dataset_folder, args.dataset_name, "processed_data")
    dataset_train = TrajectoryDataset(
        os.path.join(datapath, "train"),
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        intent_csv_path="../main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv"
    )
    dataset_test = TrajectoryDataset(
        os.path.join(datapath, "test"),
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        intent_csv_path="../main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv"
    )

    loader_train = DataLoader(dataset_train, batch_size=1, num_workers=4, shuffle=True, collate_fn=seq_collate)
    loader_test = DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=True, collate_fn=seq_collate)

    model = TrajAirNet(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.total_epochs + 1):
        model.train()
        tot_loss = 0
        for i, batch in enumerate(tqdm(loader_train)):
            batch = [t.to(device) for t in batch]
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            adj = torch.ones((num_agents, num_agents))

            optimizer.zero_grad()
            recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0],
                                    torch.transpose(context, 1, 2), intent_labels)

            loss = sum(loss_func(recon_y[a], torch.transpose(pred_traj[:, :, a], 0, 1).unsqueeze(0), m[a], var[a])
                       for a in range(num_agents))
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        avg_loss = tot_loss / len(loader_train)
        print(f"Epoch {epoch}, Train Loss: {avg_loss}")

        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_loss})

        if args.evaluate:
            model.eval()
            test_ade_loss, test_fde_loss = test(model, loader_test, device)
            print(f"Test ADE: {test_ade_loss}, FDE: {test_fde_loss}")
            if args.use_wandb:
                wandb.log({"test_ade_loss": test_ade_loss, "test_fde_loss": test_fde_loss})
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    """
    Command line entry point for wandb sweeps.
    Parses command line args and calls train_with_config.
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Add all the sweep parameters as command line arguments
    parser.add_argument('--tcn_channel_size', type=int)
    parser.add_argument('--tcn_layers', type=int) 
    parser.add_argument('--gat_heads', type=int)
    parser.add_argument('--graph_hidden', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--intent_embed_dim', type=int)
    parser.add_argument('--lr', type=float)
    
    args = parser.parse_args()
    
    # Convert to dictionary, filtering out None values
    config_overrides = {k: v for k, v in vars(args).items() if v is not None}
    
    train_with_config(config_overrides)