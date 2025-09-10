import os
import sys
import argparse
import contextlib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler

from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate, loss_func
from test import test
from config_loader import get_training_args, load_config


# --- HELPER FUNCTION FOR CREATING SCHEDULERS ---
def get_scheduler(optimizer, args, total_steps_per_epoch):
    if args.scheduler_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(args.total_epochs - args.warmup_epochs) * total_steps_per_epoch
        )
    elif args.scheduler_type == 'plateau':
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=3
        )
    else:
        return None

    if args.warmup_epochs > 0:
        warmup_steps = args.warmup_epochs * total_steps_per_epoch
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )

    return main_scheduler


def train():
    # --- UPDATED ARGUMENT PARSING FOR NEW SCHEDULER HPARAMS ---
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('--alpha', type=float)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--gat_heads', type=int)
        parser.add_argument('--graph_hidden', type=int)
        parser.add_argument('--intent_embed_dim', type=int)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--tcn_channel_size', type=int)
        parser.add_argument('--tcn_layers', type=int)
        parser.add_argument('--total_epochs', type=int)
        parser.add_argument('--force_single_gpu', type=bool)
        parser.add_argument('--scheduler_type', type=str, default='none')
        parser.add_argument('--warmup_epochs', type=int, default=0)

        sweep_args, _ = parser.parse_known_args()
        wandb_config = {k: v for k, v in vars(sweep_args).items() if v is not None}
        args = get_training_args(wandb_config=wandb_config)
    else:
        args = get_training_args()

    force_single_gpu = getattr(args, 'force_single_gpu', False)
    if force_single_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        rank = 0
        is_ddp = False
    else:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        device = f'cuda:{rank}'
        is_ddp = True

    config_loader = load_config()

    if args.use_wandb and rank == 0:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        for key, value in wandb.config.items():
            if hasattr(args, key): setattr(args, key, value)

    dataset_paths = config_loader.get_dataset_paths()
    if rank == 0: print("Loading Train Data from ", dataset_paths['train'])
    dataset_train = TrajectoryDataset(dataset_paths['train'], obs_len=args.obs, pred_len=args.preds,
                                      step=args.preds_step, delim=args.delim,
                                      intent_csv_path=args.intent_csv_path)

    if rank == 0: print("Loading Test Data from ", dataset_paths['test'])
    dataset_test = TrajectoryDataset(dataset_paths['test'], obs_len=args.obs, pred_len=args.preds,
                                     step=args.preds_step, delim=args.delim,
                                     intent_csv_path=args.intent_csv_path)

    # --- DataLoader tweaks (persistent workers, prefetch) ---
    common_dl_kwargs = dict(
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        num_workers=args.train_num_workers
    )

    if is_ddp:
        train_sampler = DistributedSampler(dataset_train, shuffle=args.shuffle_train)
        loader_train = DataLoader(dataset_train,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  collate_fn=seq_collate,
                                  drop_last=True,
                                  **common_dl_kwargs)
    else:
        loader_train = DataLoader(dataset_train,
                                  batch_size=args.train_batch_size,
                                  shuffle=args.shuffle_train,
                                  collate_fn=seq_collate,
                                  drop_last=True,
                                  **common_dl_kwargs)

    loader_test = DataLoader(dataset_test,
                             batch_size=args.test_batch_size,
                             shuffle=args.shuffle_test,
                             collate_fn=seq_collate,
                             **common_dl_kwargs)

    model = TrajAirNet(args).to(device)

    if not force_single_gpu: dist.barrier()
    model_paths = config_loader.get_model_paths()
    checkpoint = torch.load(model_paths['pretrained_model'], map_location=device)

    checkpoint_state = checkpoint['model_state_dict']
    current_model_state = model.state_dict()
    state_to_load = {}
    for name, param in checkpoint_state.items():
        if name in current_model_state and (name.startswith('tcn_encoder_x.')
                                            or name.startswith('tcn_encoder_y.')
                                            or name.startswith('context_conv.')):
            if param.shape == current_model_state[name].shape:
                state_to_load[name] = param
            else:
                if rank == 0: print(f"Skipping {name} due to shape mismatch.")

    model.load_state_dict(state_to_load, strict=False)

    for param in model.tcn_encoder_x.parameters(): param.requires_grad = False
    for param in model.tcn_encoder_y.parameters(): param.requires_grad = False
    for param in model.context_conv.parameters(): param.requires_grad = False

    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0: print("Loaded and frozen TCN and wind CNN weights from checkpoint")

    # --- Optimizer: fused AdamW ---
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=1e-2,
                            fused=True)

    total_steps_per_epoch = len(loader_train)
    scheduler = get_scheduler(optimizer, args, total_steps_per_epoch)
    if rank == 0: print(f"Using scheduler: '{args.scheduler_type}' with {args.warmup_epochs} warmup epochs.")

    scaler = GradScaler()
    accumulation_steps = args.grad_accum_steps

    if rank == 0: print("Starting Training....")

    for epoch in range(1, args.total_epochs + 1):
        if is_ddp: loader_train.sampler.set_epoch(epoch)
        model.train()
        tot_loss = 0
        train_iterator = tqdm(loader_train) if rank == 0 else loader_train

        for i, batch in enumerate(train_iterator):
            batch = [tensor.to(device, non_blocking=True) for tensor in batch]
            obs_traj, pred_traj, _, _, context, intent_labels, time_delta_features, _ = batch
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            adj = torch.ones((num_agents, num_agents), device=device)

            # --- no_sync for gradient accumulation ---
            ddp_sync = ((i + 1) % accumulation_steps == 0)
            sync_ctx = (model.no_sync if is_ddp and not ddp_sync else contextlib.nullcontext)

            with sync_ctx():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    recon_y, m, var = model(torch.transpose(obs_traj, 1, 2),
                                            pred_traj, adj,
                                            torch.transpose(context, 1, 2),
                                            intent_labels, time_delta_features)
                    loss = 0
                    for agent in range(num_agents):
                        loss += loss_func(recon_y[agent],
                                          torch.transpose(pred_traj[:, :, agent], 0, 1).unsqueeze(0),
                                          m[agent], var[agent])
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                tot_loss += loss.item() * accumulation_steps

            if ddp_sync:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler and args.scheduler_type != 'plateau':
                    scheduler.step()

        if not is_ddp:
            avg_train_loss = tot_loss / len(loader_train)
        else:
            total_loss_tensor = torch.tensor(tot_loss, device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = total_loss_tensor.item() / len(loader_train.dataset)

        if rank == 0:
            print(f"EPOCH: {epoch}, Train Loss: {avg_train_loss:.6f}")
            if args.use_wandb:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "learning_rate": current_lr})

            if args.save_model:
                model_path = os.path.join(os.getcwd(), args.model_pth,
                                          f"model_{args.dataset_name}_{epoch}.pt")
                print("Saving model at", model_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                }, model_path)

            if args.evaluate:
                print("Starting Testing....")
                model.eval()
                with torch.no_grad():
                    test_model = model.module if is_ddp else model
                    (average_ade, average_fde, _, _, all_ade_std, _, _, all_fde_std, _) = \
                        test(test_model, loader_test, device, dataset_test)

                print(f"Test Results (Best per batch): ADE: {average_ade:.4f}, FDE: {average_fde:.4f}")

                if args.use_wandb:
                    wandb.log({
                        "test_ade_loss": average_ade,
                        "test_fde_loss": average_fde,
                        "test_ade_std": all_ade_std,
                        "test_fde_std": all_fde_std
                    })

                if scheduler and args.scheduler_type == 'plateau':
                    scheduler.step(average_ade)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('forkserver')
    except RuntimeError:
        pass
    train()
