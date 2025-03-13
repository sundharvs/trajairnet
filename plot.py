import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset, seq_collate

import pdb
import matplotlib.pyplot as plt

from datetime import datetime
import heapq

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

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim, skip=args.skip)
    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=8,shuffle=False,collate_fn=seq_collate)

    breakpoint()
    
    ##Load model
    model = TrajAirNet(args)
    model.to(device)

    model_path =  os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + str(args.epoch) + ".pt"

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
   
# skip = 1 
#     worst_sequences = [
#     [21.4351, 325734],
#     [20.9715, 413683],
#     [20.8644, 413681],
#     [20.7495, 413680],
#     [20.7490, 413682],
#     [20.7290, 413679],
#     [20.2970, 325733],
#     [20.2904, 413678],
#     [20.1791, 325732],
#     [19.9204, 325731],
#     [19.9069, 413676],
#     [19.8754, 413677],
#     [19.4922, 325730],
#     [19.4291, 325729],
#     [19.2821, 413675],
#     [19.1871, 325728],
#     [18.1062, 432452],
#     [17.6574, 432451],
#     [17.3640, 432450],
#     [17.3522, 432449]
# ]

    # skip = 10
    # worst_sequences = [
    #     [19.3617, 41657],
    #     [18.2296, 175326],
    #     [17.2494, 209374],
    #     [16.4516, 220595],
    #     [16.2450, 269512],
    #     [14.1463, 107860],
    #     [13.7680, 106808],
    #     [13.4298, 231825],
    #     [13.1040, 181895],
    #     [12.4729, 239646],
    #     [12.4499, 106769],
    #     [12.4253, 209216],
    #     [12.3755, 209251],
    #     [12.2719, 240040],
    #     [12.2529, 2398],
    #     [12.1337, 273301],
    #     [11.8728, 41188],
    #     [11.3334, 187638],
    #     [11.0013, 41157],
    #     [10.9475, 96957]
    # ]

    test_ade_loss, worst_sequences = test(model,loader_test,device,dataset_test)
    plot(model,dataset_test,worst_sequences[4980:5000],device)
    breakpoint()

def test(model,loader_test,device,dataset_test):
    tot_ade_loss = 0
    # tot_fde_loss = 0
    tot_batch = 0
    
    # loss_records = []
    # Min-heap to track the top 20 worst cases (by ADE loss)
    worst_cases_heap = []

    stop_idx = len(dataset_test)/4
    debug_var = 0
    pbar = tqdm(total=stop_idx)
    
    for batch_idx, batch in enumerate(loader_test):
        tot_batch += 1
        batch = [tensor.to(device) for tensor in batch]

        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, timestamp, seq_start  = batch
        num_agents = obs_traj_all.shape[1]
        
        best_ade_loss = float('inf')
        # best_fde_loss = float('inf')
        
        for i in range(3):
            z = torch.randn([1,1 ,128]).to(device)
            
            adj = torch.ones((num_agents,num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all,1,2),z,adj,torch.transpose(context,1,2))
            
            ade_loss = 0
            # fde_loss = 0
            for agent in range(num_agents):
                # obs_traj = np.squeeze(obs_traj_all[:,agent,:].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:,agent,:].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                ade_loss += ade(recon_pred, pred_traj)
                # fde_loss += fde((recon_pred), (pred_traj))
           
            
            ade_total_loss = ade_loss/num_agents
            # fde_total_loss = fde_loss/num_agents
            if ade_total_loss<best_ade_loss:
                best_ade_loss = ade_total_loss
                # best_fde_loss = fde_total_loss

        tot_ade_loss += best_ade_loss
        # tot_fde_loss += best_fde_loss
        
        # loss_records.append((batch_idx, best_ade_loss))
        # loss_records.append((batch_idx, best_ade_loss, best_fde_loss))
        heapq.heappush(worst_cases_heap, (best_ade_loss, batch_idx))

        # If the heap exceeds 100, remove the smallest element (min-heap keeps the worst cases)
        if len(worst_cases_heap) > 5000:
            heapq.heappop(worst_cases_heap)
        
        debug_var += 1
        pbar.update(1)

        if(debug_var >= stop_idx):
            break
    
    pbar.close()
    worst_cases = sorted(worst_cases_heap, key=lambda x: x[0], reverse=True)
    # worst_cases = sorted(loss_records, key=lambda x: x[1], reverse=True)[:25]  # Sorting by ADE (change x[1] to x[2] for FDE)

    print("\nTop 20 Worst Cases (By ADE):")
    for ade_loss, idx in worst_cases[0:20]:
        print(f"Batch Index: {idx}, ADE: {ade_loss:.4f}")

    return tot_ade_loss/(tot_batch),worst_cases
    # return tot_ade_loss/(tot_batch),tot_fde_loss/(tot_batch),worst_cases

def plot(model,dataset_test,worst_sequences,device):
    rank = 4980
    for ade_loss, investigation_idx in worst_sequences:

        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, timestamp, seq_start  = seq_collate([dataset_test[investigation_idx]])
        
        obs_traj_all = obs_traj_all.to(device)
        pred_traj_all = pred_traj_all.to(device)
        context = context.to(device)
        
        num_agents = obs_traj_all.shape[1]
                
        plt.figure()
        plt.grid()
        plt.xlim([-4, 4])
        plt.ylim([-2, 2])

        for i in range(100):
            z = torch.randn([1,1 ,128]).to(device)
            adj = torch.ones((num_agents,num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all,1,2),z,adj,torch.transpose(context,1,2))
            
            for agent in range(num_agents):
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                plt.plot(recon_pred[:,0],recon_pred[:,1], color='green', alpha=0.2, linewidth=2)
        
        for agent in range(num_agents):
            obs_traj = np.squeeze(obs_traj_all[:,agent,:].cpu().numpy())
            pred_traj = np.squeeze(pred_traj_all[:,agent,:].cpu().numpy())
            plt.plot(obs_traj[:,0],obs_traj[:,1], color='blue', linewidth=2)
            plt.arrow(obs_traj[-1,0], obs_traj[-1,1], pred_traj[0,0] - obs_traj[-1,0], pred_traj[0,1] - obs_traj[-1,1], color='black',width=0.03)
            plt.plot(pred_traj[:,0],pred_traj[:,1], color='red', linewidth=2)
        
        start_timestamp = datetime.utcfromtimestamp(timestamp.min()).strftime('%Y-%m-%d %H:%M:%S')
        end_timestamp = datetime.utcfromtimestamp(timestamp.max()).strftime('%Y-%m-%d %H:%M:%S')

        plt.title(f'{start_timestamp} - {end_timestamp}\n#{rank} ADE: {ade_loss:.4f}')

        plt.savefig(f'{rank}.png')
        print('-----------')
        print(f'Rank #{rank}, Sequence #{investigation_idx}, File #{dataset_test.get_file_index(investigation_idx)}')
        print(f'First coordinates: {obs_traj_all[0]}')
        print(f'Timestamp range: {start_timestamp} - {end_timestamp}')
        rank += 1 

if __name__=='__main__':
    main()

