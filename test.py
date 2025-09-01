import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset, seq_collate

def main():
    
    parser=argparse.ArgumentParser(description='Test TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--epoch',type=int,required=True)

    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)
    parser.add_argument('--skip', type=int, default=1)
    
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

    parser.add_argument('--delim',type=str,default=' ')

    parser.add_argument('--model_dir', type=str , default="/saved_models/")
    
    args=parser.parse_args()


    ##Select device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load data

    datapath = args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim, skip=args.skip)
    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=4,shuffle=False,collate_fn=seq_collate)

    ##Load model
    model = TrajAirNet(args)
    model.to(device)

    model_path =  os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + str(args.epoch) + ".pt"


    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_ade_loss, test_fde_loss = test(model,loader_test,device)

    print("Test ADE Loss: ",test_ade_loss,"Test FDE Loss: ",test_fde_loss)

def test(model,loader_test,device):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_batch = 0
    
    loss_records = []
    
    for batch_idx, batch in enumerate(tqdm(loader_test)):
        tot_batch += 1
        batch = [tensor.to(device) for tensor in batch]

        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, intent_labels, seq_start = batch 
        num_agents = obs_traj_all.shape[1]
        
        best_ade_loss = float('inf')
        best_fde_loss = float('inf')
        
        for i in range(1):
            z = torch.randn([1,1 ,128]).to(device)
            
            adj = torch.ones((num_agents,num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all,1,2),z,adj,torch.transpose(context,1,2), intent_labels)
            
            ade_loss = 0
            fde_loss = 0
            for agent in range(num_agents):
                obs_traj = np.squeeze(obs_traj_all[:,agent,:].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:,agent,:].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                ade_loss += ade(recon_pred, pred_traj)
                fde_loss += fde((recon_pred), (pred_traj))
           
            
            ade_total_loss = ade_loss/num_agents
            fde_total_loss = fde_loss/num_agents
            if ade_total_loss<best_ade_loss:
                best_ade_loss = ade_total_loss
                best_fde_loss = fde_total_loss

        tot_ade_loss += best_ade_loss
        tot_fde_loss += best_fde_loss
        
        loss_records.append((batch_idx, best_ade_loss, best_fde_loss))
        
    worst_cases = sorted(loss_records, key=lambda x: x[1], reverse=True)[:10]  # Sorting by ADE (change x[1] to x[2] for FDE)

    print("\nTop 10 Worst Cases (By ADE):")
    for idx, ade_loss, fde_loss in worst_cases:
        print(f"Batch Index: {idx}, ADE: {ade_loss:.4f}, FDE: {fde_loss:.4f}")

    return tot_ade_loss/(tot_batch),tot_fde_loss/(tot_batch)


if __name__=='__main__':
    main()

