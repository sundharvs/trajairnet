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
from config_loader import get_training_args

def train_with_config(config=None):
    """
    Train function that can be called with a config dictionary for sweeps
    """
    if config is None:
        # Default mode - load from config file
        args = get_training_args()
    else:
        # Sweep mode - merge sweep config with base config
        args = get_training_args(wandb_config=config)
        # Disable model saving for sweeps by default
        args.save_model = config.get('save_model', False)

    ##Initialize wandb (only if not already initialized by sweep)
    if args.use_wandb and not wandb.run:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )

    ##Select device
    device = torch.device(args.device)

    ##Load test and train data
    from config_loader import load_config
    config_loader = load_config()
    dataset_paths = config_loader.get_dataset_paths()

    print("Loading Train Data from ",dataset_paths['train'])
    dataset_train = TrajectoryDataset(dataset_paths['train'], obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim,
                                    intent_csv_path=args.intent_csv_path)

    print("Loading Test Data from ",dataset_paths['test'])
    dataset_test = TrajectoryDataset(dataset_paths['test'], obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim,
                                   intent_csv_path=args.intent_csv_path)

    loader_train = DataLoader(dataset_train,batch_size=args.train_batch_size,num_workers=args.train_num_workers,shuffle=args.shuffle_train,collate_fn=seq_collate)
    loader_test = DataLoader(dataset_test,batch_size=args.test_batch_size,num_workers=args.test_num_workers,shuffle=args.shuffle_test,collate_fn=seq_collate)

    model = TrajAirNet(args)
    model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    num_batches = len(loader_train)
 
    print("Starting Training....")

    for epoch in range(1, args.total_epochs+1):

        model.train()
        loss_batch = 0 
        batch_count = 0
        tot_batch_count = 0
        tot_loss = 0
        for batch in tqdm(loader_train):
            batch_count += 1
            tot_batch_count += 1
            batch = [tensor.to(device) for tensor in batch]
            obs_traj , pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, seq_start = batch 
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj,1,2)
            adj = torch.ones((num_agents,num_agents))

            optimizer.zero_grad()
            recon_y,m,var = model(torch.transpose(obs_traj,1,2),pred_traj, adj[0],torch.transpose(context,1,2),intent_labels)
            loss = 0
            
            for agent in range(num_agents):
                loss += loss_func(recon_y[agent],torch.transpose(pred_traj[:,:,agent],0,1).unsqueeze(0),m[agent],var[agent])
            
            loss_batch += loss
            tot_loss += loss.item()
            if batch_count>8:
                loss_batch.backward()
                optimizer.step()
                loss_batch = 0 
                batch_count = 0

        avg_train_loss = tot_loss/tot_batch_count
        print("EPOCH: ",epoch,"Train Loss: ",avg_train_loss)
        
        # Log training loss to wandb (always log this)
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
            })

        if args.save_model:  
            loss = avg_train_loss
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            print("Saving model at",model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, model_path)
        
        if args.evaluate:
            print("Starting Testing....")
        
            model.eval()
            test_ade_loss, test_fde_loss = test(model,loader_test,device)

            print("EPOCH: ",epoch,"Train Loss: ",loss,"Test ADE Loss: ",test_ade_loss,"Test FDE Loss: ",test_fde_loss)
            
            # Log evaluation metrics to wandb
            if args.use_wandb:
                wandb.log({
                    "test_ade_loss": test_ade_loss,
                    "test_fde_loss": test_fde_loss,
                })

if __name__=='__main__':
    # Check if called with command line args (for wandb sweep)
    import sys
    if len(sys.argv) > 1:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float)
        parser.add_argument('--total_epochs', type=int)
        parser.add_argument('--intent_embed_dim', type=int)
        parser.add_argument('--gat_heads', type=int)
        parser.add_argument('--graph_hidden', type=int)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--alpha', type=float)
        parser.add_argument('--tcn_channel_size', type=int)
        parser.add_argument('--tcn_layers', type=int)
        
        args = parser.parse_args()
        config_overrides = {k: v for k, v in vars(args).items() if v is not None}
        train_with_config(config_overrides)
    else:
        train_with_config()