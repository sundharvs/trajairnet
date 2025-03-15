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

from datetime import datetime, timezone
import heapq

import pickle
import glob

from molmo import Molmo

def main():
    parser=argparse.ArgumentParser(description='Test TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='tartansubset')
    parser.add_argument('--epoch',type=int,default=5)

    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)
    parser.add_argument('--skip', type=int, default=10)
    
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

    dataset_filename = f'{args.dataset_name}_{args.skip}.pkl'
    if os.path.exists(dataset_filename):
        # Load the dataset from the pickle file
        with open(dataset_filename, "rb") as file:
            dataset_test = pickle.load(file)
    else:
        # Create a new dataset object
        dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim, skip=args.skip)
        
        # Save the dataset to the pickle file
        with open(dataset_filename, "wb") as file:
            pickle.dump(dataset_test, file)

    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=8,shuffle=False,collate_fn=seq_collate)
    
    ##Load model
    model = TrajAirNet(args)
    model.to(device)
    model_path =  os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + str(args.epoch) + ".pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    transcripts = get_transcript_timestamps()

    timestamp_indices_filename = f'timestamp_indices_{args.dataset_name}.pkl'
    if os.path.exists(timestamp_indices_filename):
        with open(timestamp_indices_filename, 'rb') as f:
            timestamp_indices = pickle.load(f)
    else:
        timestamp_indices = find_timestamps(model,loader_test,device,dataset_test,transcripts)
        with open(timestamp_indices_filename, 'wb') as f:
            pickle.dump(timestamp_indices, f)

    breakpoint()
    # Remove indices which are -1 and corresponding timestamps and transcripts
    timestamp_indices = np.array(timestamp_indices)
    valid_indices = timestamp_indices != -1
    timestamp_indices = timestamp_indices[valid_indices]
    transcripts = [transcript for i, transcript in enumerate(transcripts) if valid_indices[i]]

    timestamp_indices = timestamp_indices.astype(int)
    timestamp_indices = timestamp_indices.tolist()
        
    torch.cuda.empty_cache()
    vlm = Molmo()
    plot(model,dataset_test,timestamp_indices,transcripts,device,vlm)
    breakpoint()

def get_transcript_timestamps():
    # Define paths
    transcripts_path = "~/git/TartanAviation/audio/processed_audio/transcripts"
    audio_path = "~/git/TartanAviation/audio/processed_audio"

    # Expand user directory
    transcripts_path = os.path.expanduser(transcripts_path)
    audio_path = os.path.expanduser(audio_path)

    # Get all transcript files
    transcript_files = glob.glob(os.path.join(transcripts_path, "*_transcript.txt"))

    timestamps = []
    base_names = []
    transcripts = []

    for transcript_file in transcript_files:
        # Derive corresponding *.txt filename
        base_name = os.path.basename(transcript_file).replace("_transcript.txt", ".txt")
        txt_file_path = os.path.join(audio_path, base_name)
        
        with open(transcript_file, "r") as f:
            transcript = f.read()
            transcripts.append(transcript)
        
        if os.path.exists(txt_file_path):
            with open(txt_file_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 2 and lines[0].strip() == "Start Time:":
                    # Parse the timestamp
                    timestamp_str = lines[1].strip()
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                    unix_timestamp = round(dt.timestamp())
                    base_names.append(base_name)
                    timestamps.append(unix_timestamp)

    return zip(base_names, timestamps, transcripts)


def find_timestamps(model,loader_test,device,dataset_test,transcripts):
    _, search_timestamps, _ = list(zip(*transcripts))
    
    timestamp_indices = -1*np.ones(len(search_timestamps))
    
    timestamp_pbar = tqdm(total=len(timestamp_indices), position=1, desc='Timestamps')
    dataset_pbar = tqdm(total=len(dataset_test), position=0, desc='Dataset')
    
    for batch_idx, batch in enumerate(loader_test):
        batch = [tensor.to(device) for tensor in batch]

        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, timestamp, tail, seq_start  = batch
        num_agents = obs_traj_all.shape[1]
                
        for agent in range(num_agents):
            # obs_traj = np.squeeze(obs_traj_all[:,agent,:].cpu().numpy())
            pred_traj = np.squeeze(pred_traj_all[:,agent,:].cpu().numpy())

        for i, ts in enumerate(search_timestamps):
            if ts in timestamp:
                timestamp_indices[i] = batch_idx
                timestamp_pbar.update(1)
        
        dataset_pbar.update(1)
        
        if(not any(timestamp_indices == -1)):
            break
    
    return timestamp_indices

def plot(model,dataset_test,timestamp_indices,transcripts, device,vlm):
    for list_idx, investigation_idx in enumerate(timestamp_indices):

        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, timestamp, tail, seq_start  = seq_collate([dataset_test[investigation_idx]])
        
        obs_traj_all = obs_traj_all.to(device)
        pred_traj_all = pred_traj_all.to(device)
        context = context.to(device)
                
        num_agents = obs_traj_all.shape[1]
        
        # Convert base 36 encoded tail numbers to strings
        base_repr_vec = np.vectorize(np.base_repr)
        tail_numbers = base_repr_vec(tail[0].numpy().flatten(), 36)
                
        plt.figure()
        plt.grid()
        plt.xlim([-4, 4])
        plt.ylim([-2, 2])

        for i in range(100):
            z = torch.randn([1,1 ,128]).to(device)
            adj = torch.ones((num_agents,num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all,1,2),z,adj,torch.transpose(context,1,2))
            
            # TODO - calculate ADE
            
            for agent in range(num_agents):
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                plt.plot(recon_pred[:,0],recon_pred[:,1], color='green', alpha=0.2, linewidth=2)
        
        for agent in range(num_agents):
            obs_traj = np.squeeze(obs_traj_all[:,agent,:].cpu().numpy())
            pred_traj = np.squeeze(pred_traj_all[:,agent,:].cpu().numpy())
            plt.plot(obs_traj[:,0],obs_traj[:,1], color='blue', linewidth=2)
            plt.arrow(obs_traj[-1,0], obs_traj[-1,1], pred_traj[0,0] - obs_traj[-1,0], pred_traj[0,1] - obs_traj[-1,1], color='black',width=0.03)
            plt.text((obs_traj[-1, 0] + pred_traj[0, 0]) / 2 - 0.3, (obs_traj[-1, 1] + pred_traj[0, 1]) / 2 - 0.3, tail_numbers[agent], fontsize=10, color='black', ha='center', va='center', backgroundcolor='yellow')

            # plt.plot(pred_traj[:,0],pred_traj[:,1], color='red', linewidth=2)
                
        start_timestamp = datetime.fromtimestamp(timestamp.min().item(), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        end_timestamp = datetime.fromtimestamp(timestamp.max().item(), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        # plt.title(f'{start_timestamp} - {end_timestamp}\n#{rank} ADE: {ade_loss:.4f}')

        plt.savefig(f'temp.jpg')
        
        radio_call = transcripts[list_idx][2]
        
        system_prompt = f'''This is a top down plot of aircraft in the terminal airspace around an untowered airport.
        The black arrows represent aircraft, and the green regions represent predicted trajectories.
        An aircraft has made the following radio call: {radio_call}.
        Does the predicted trajectory for this aircraft (which is in green), match with the stated intentions of the pilot in his radio call?
        Explain your reasoning then at the end of your response say "Decision: [Match, Mismatch, Inconclusive]" choosing one of the options. 
        '''
        
        vlm_output = vlm.call('temp.jpg', system_prompt)
        
        print(vlm_output)
        
        print('-----------')
        print(f'Sequence #{investigation_idx}, File #{dataset_test.get_file_index(investigation_idx)}')
        print(f'Prediction Timestamp range: {start_timestamp} - {end_timestamp}')
        print(f'Radio Call File: {transcripts[list_idx][0]}')
        print(f'Radio call: {radio_call}')
        # print(f'Radio Call Timestamp: {datetime.fromtimestamp(transcripts[list_idx][1], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}')

if __name__=='__main__':
    main()
