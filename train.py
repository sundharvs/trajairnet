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



def train():

    ##Dataset params
    parser=argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)

    ##Network params
    parser.add_argument('--input_channels',type=int,default=3)
    parser.add_argument('--tcn_channel_size',type=int,default=256)
    parser.add_argument('--tcn_layers',type=int,default=2)
    parser.add_argument('--tcn_kernels',type=int,default=4)

    parser.add_argument('--num_context_input_c',type=int,default=2)
    parser.add_argument('--num_context_output_c',type=int,default=7)
    parser.add_argument('--cnn_kernels',type=int,default=2)

    parser.add_argument('--gat_heads',type=int, default=16)
    parser.add_argument('--graph_hidden',type=int,default=256)
    parser.add_argument('--dropout',type=float,default=0.05)
    parser.add_argument('--alpha',type=float,default=0.2)
    parser.add_argument('--cvae_hidden',type=int,default=128)
    parser.add_argument('--cvae_channel_size',type=int,default=128)
    parser.add_argument('--cvae_layers',type=int,default=2)
    parser.add_argument('--mlp_layer',type=int,default=32)

    ##Intent embedding params
    parser.add_argument('--intent_embed_dim',type=int,default=32)
    parser.add_argument('--num_intent_classes',type=int,default=16)

    parser.add_argument('--lr',type=float,default=0.001)


    parser.add_argument('--total_epochs',type=int, default=50)
    parser.add_argument('--delim',type=str,default=' ')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)

    parser.add_argument('--model_pth', type=str , default="/saved_models/")
    
    ##Wandb params
    parser.add_argument('--use_wandb', type=bool, default=True)
    parser.add_argument('--wandb_project', type=str, default="trajairnet-intent-attention-head")
    parser.add_argument('--wandb_entity', type=str, default=None)

    args=parser.parse_args()

    ##Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )

    ##Select device

    device = torch.device('cuda')

    ##Load test and train data
    datapath = args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Train Data from ",datapath + "train")
    dataset_train = TrajectoryDataset(datapath + "train", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    loader_train = DataLoader(dataset_train,batch_size=1,num_workers=8,shuffle=True,collate_fn=seq_collate)
    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=8,shuffle=False,collate_fn=seq_collate)

    model = TrajAirNet(args)
    model.to(device)

    ##Load and freeze TCN and wind CNN weights from saved_models/model_7daysJune_50.pt
    checkpoint = torch.load('saved_models/model_7daysJune_50.pt', map_location=device)
    checkpoint_state = checkpoint['model_state_dict']
    
    # Selectively load TCN encoder weights
    tcn_x_keys = [k for k in checkpoint_state.keys() if k.startswith('tcn_encoder_x.')]
    tcn_y_keys = [k for k in checkpoint_state.keys() if k.startswith('tcn_encoder_y.')]
    wind_cnn_keys = [k for k in checkpoint_state.keys() if k.startswith('context_conv.')]
    
    # Load the specific weights
    current_state = model.state_dict()
    for key in tcn_x_keys + tcn_y_keys + wind_cnn_keys:
        if key in current_state:
            current_state[key] = checkpoint_state[key]
    
    model.load_state_dict(current_state)
    
    # Freeze TCN encoder weights
    for param in model.tcn_encoder_x.parameters():
        param.requires_grad = False
    for param in model.tcn_encoder_y.parameters():
        param.requires_grad = False
    
    # Freeze wind CNN weights
    for param in model.context_conv.parameters():
        param.requires_grad = False
    
    print("Loaded and frozen TCN and wind CNN weights from checkpoint")

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
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_attention_head_" + str(epoch) + ".pt"
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

    train()
