import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate, loss_func
from test import test
from config_loader import get_training_args, load_config


def train():
    # Load configuration from config file
    args = get_training_args()
    
    dist.init_process_group("nccl")
    # Get the rank of the current process (0 for the first GPU, 1 for the second, etc.)
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}' # Use the rank to create a device string

    # Load config loader for dataset and model paths
    config_loader = load_config()

    if args.use_wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )

    dataset_paths = config_loader.get_dataset_paths()
    
    if rank == 0:
        print("Loading Train Data from ", dataset_paths['train'])
    dataset_train = TrajectoryDataset(dataset_paths['train'], obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim,
                                    intent_csv_path=args.intent_csv_path)

    if rank == 0:
        print("Loading Test Data from ", dataset_paths['test'])
    dataset_test = TrajectoryDataset(dataset_paths['test'], obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim,
                                   intent_csv_path=args.intent_csv_path)

    train_sampler = DistributedSampler(dataset_train, shuffle=args.shuffle_train)
    
    loader_train = DataLoader(dataset_train, batch_size=args.train_batch_size, num_workers=args.train_num_workers,
                              sampler=train_sampler, collate_fn=seq_collate, pin_memory=True)
    
    loader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.test_num_workers,
                             shuffle=args.shuffle_test, collate_fn=seq_collate)

    model = TrajAirNet(args).to(device)

    ## Load and freeze weights
    model_paths = config_loader.get_model_paths()
    dist.barrier()
    checkpoint = torch.load(model_paths['pretrained_model'], map_location=device)
    
    checkpoint_state = checkpoint['model_state_dict']
    current_model_state = model.state_dict()

    # Create a new state_dict with only the weights we want to load
    # (TCN and context CNN layers)
    state_to_load = {}
    for name, param in checkpoint_state.items():
        if name in current_model_state and (
            name.startswith('tcn_encoder_x.') or 
            name.startswith('tcn_encoder_y.') or 
            name.startswith('context_conv.')
        ):
            # Check if shapes match before adding to the dict
            if param.shape == current_model_state[name].shape:
                state_to_load[name] = param
            else:
                if rank == 0:
                    print(f"Skipping {name} due to shape mismatch.")
    
    current_model_state.update(state_to_load)
    model.load_state_dict(current_model_state)
    
    for param in model.tcn_encoder_x.parameters(): param.requires_grad = False
    for param in model.tcn_encoder_y.parameters(): param.requires_grad = False
    for param in model.context_conv.parameters(): param.requires_grad = False
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        print("Loaded and frozen TCN and wind CNN weights from checkpoint")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    scaler = GradScaler()
    
    accumulation_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 9

    if rank == 0:
        print("Starting Training....")

    for epoch in range(1, args.total_epochs + 1):
        loader_train.sampler.set_epoch(epoch)
        model.train()
        
        tot_loss = 0
        train_iterator = tqdm(loader_train) if rank == 0 else loader_train

        for i, batch in enumerate(train_iterator):
            batch = [tensor.to(device, non_blocking=True) for tensor in batch]
            obs_traj, pred_traj, _, _, context, intent_labels, time_delta_features, _ = batch
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            adj = torch.ones((num_agents, num_agents), device=device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2), intent_labels, time_delta_features)
                loss = 0
                for agent in range(num_agents):
                    loss += loss_func(recon_y[agent], torch.transpose(pred_traj[:, :, agent], 0, 1).unsqueeze(0), m[agent], var[agent])
                
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            tot_loss += loss.item() * accumulation_steps # Un-normalize for logging

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        total_loss_tensor = torch.tensor(tot_loss, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = total_loss_tensor.item() / len(loader_train.dataset)

        if rank == 0:
            print(f"EPOCH: {epoch}, Train Loss: {avg_train_loss:.6f}")
            
            if args.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

            if args.save_model:
                model_path = os.path.join(os.getcwd(), args.model_pth, f"model_{args.dataset_name}_{epoch}.pt")
                print("Saving model at", model_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                }, model_path)
            
            if args.evaluate:
                print("Starting Testing....")
                model.eval()
                with torch.no_grad():
                    (average_ade, average_fde, all_ade_mean, all_ade_var, all_ade_std,
                     all_fde_mean, all_fde_var, all_fde_std, detailed_results_df) = test(model.module, loader_test, device, dataset_test)

                print(f"Test Results (Best per batch): ADE: {average_ade:.4f}, FDE: {average_fde:.4f}")
                print(f"Test Results (All inferences): ADE Mean: {all_ade_mean:.4f}, FDE Mean: {all_fde_mean:.4f}, ADE Var: {all_ade_var:.4f}, FDE Var: {all_fde_var:.4f}, ADE Std: {all_ade_std:.4f}, FDE Std: {all_fde_std:.4f}")

                if args.use_wandb:
                    wandb.log({"test_ade_loss": average_ade, "test_fde_loss": average_fde})

    dist.destroy_process_group()


if __name__ == '__main__':

    try:
        torch.multiprocessing.set_start_method('forkserver')
    except RuntimeError:
        pass

    train()