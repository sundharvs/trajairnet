import math
import os
from tqdm import tqdm

from torch import nn
import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial import distance_matrix
import random
import pandas as pd
from datetime import datetime, timezone


##Intent Lookup class

class IntentLookup:
    """
    Class to lookup intent labels for aircraft based on their most recent radio call.
    Loads intent-labeled radio calls from main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv
    """
    
    def __init__(self, intent_csv_path=None, time_delta_threshold_minutes=None):
        """
        Initialize IntentLookup with radio call data.
        
        Args:
            intent_csv_path (str): Path to transcripts_with_goals.csv file
            time_delta_threshold_minutes (float): Time threshold in minutes. If the time delta
                between radio call and prediction timestamp exceeds this, return intent 15 (Unknown)
        """
        if intent_csv_path is None:
            # Default path relative to trajairnet directory
            intent_csv_path = "/home/ssangeetha3/git/ctaf-intent-inference/main_pipeline/2_categorize_radio_calls/7daysJune_FIXED_DEBUG_with_goals.csv"
        
        self.time_delta_threshold_minutes = time_delta_threshold_minutes
        self.intent_data = {}  # {tail_number: [(timestamp, intent_category), ...]}
        self._load_intent_data(intent_csv_path)
    
    def _load_intent_data(self, csv_path):
        """Load and process radio call intent data."""
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} radio call records for intent lookup")
            
            for _, row in df.iterrows():
                tail = row['speaker_tail']
                if tail == 'Unknown' or pd.isna(tail):
                    continue
                    
                # Parse timestamp - treat as UTC (consistent with main_pipeline)
                try:
                    timestamp_str = row['start_time']
                    # Only use fromisoformat if there's explicit timezone info
                    if 'Z' in timestamp_str or '+' in timestamp_str or 'T' in timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
                    else:
                        # No timezone info, treat as UTC like main_pipeline does
                        timestamp = datetime.strptime(timestamp_str.split('.')[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
                except:
                    # Fallback: parse as UTC to match trajectory timestamps
                    timestamp = datetime.strptime(timestamp_str.split('.')[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
                
                intent_category = int(row['Goal Category']) if not pd.isna(row['Goal Category']) else 15
                
                if tail not in self.intent_data:
                    self.intent_data[tail] = []
                self.intent_data[tail].append((timestamp, intent_category))
            
            # Sort each aircraft's radio calls by timestamp
            for tail in self.intent_data:
                self.intent_data[tail].sort(key=lambda x: x[0])
                
            print(f"Processed intent data for {len(self.intent_data)} unique aircraft")
            
        except Exception as e:
            print(f"Error loading intent data from {csv_path}: {e}")
            print("Will default to intent category 15 (Insufficient information) for all aircraft")
    
    def _convert_tail_to_string(self, tail_numeric):
        """Convert base-36 numeric tail to string format."""
        try:
            if tail_numeric == 66728889815:  # Special case for 'UNKNOWN'
                return 'Unknown'
            return np.base_repr(int(tail_numeric), 36)
        except:
            return 'Unknown'
    
    def get_most_recent_intent_with_time_delta(self, tail_number, sequence_end_timestamp):
        """
        Get the most recent intent category and time delta for an aircraft before a given timestamp.
        
        Args:
            tail_number (float or str): Aircraft tail number (numeric base-36 or string)
            sequence_end_timestamp (float): Unix timestamp of sequence end
            
        Returns:
            tuple: (intent_category, time_delta_seconds)
                - intent_category (int): Intent category (1-16), defaults to 15 if no radio call found or time delta exceeds threshold
                - time_delta_seconds (float): Time delta in seconds, or threshold value for missing/stale data
        """
        # Convert numeric tail to string format
        if isinstance(tail_number, (int, float)):
            tail_str = self._convert_tail_to_string(tail_number)
        else:
            tail_str = tail_number
        
        # Calculate threshold in seconds for max time delta
        threshold_seconds = (self.time_delta_threshold_minutes * 60.0 
                           if self.time_delta_threshold_minutes is not None else float('inf'))
        
        if tail_str not in self.intent_data:
            return 15, threshold_seconds  # No data found - return max time delta
        
        # Find most recent radio call before sequence end
        radio_calls = self.intent_data[tail_str]
        most_recent_intent = 15  # Default
        most_recent_timestamp = None
        
        for timestamp, intent in radio_calls:
            if timestamp <= sequence_end_timestamp:
                most_recent_intent = intent
                most_recent_timestamp = timestamp
            else:
                break  # Radio calls are sorted by timestamp
        
        # Calculate time delta
        if most_recent_timestamp is None:
            # No radio call found before sequence end
            return 15, threshold_seconds
        
        time_delta_seconds = sequence_end_timestamp - most_recent_timestamp
        
        # Check time delta threshold if configured
        if (self.time_delta_threshold_minutes is not None and 
            time_delta_seconds > threshold_seconds):
            return 15, threshold_seconds  # Return "Unknown" with max time delta if exceeds threshold
        
        return most_recent_intent, time_delta_seconds
    
    def get_most_recent_intent(self, tail_number, sequence_end_timestamp):
        """
        Get the most recent intent category for an aircraft before a given timestamp.
        Backward compatibility method - returns only intent.
        
        Args:
            tail_number (float or str): Aircraft tail number (numeric base-36 or string)
            sequence_end_timestamp (float): Unix timestamp of sequence end
            
        Returns:
            int: Intent category (1-16), defaults to 15 if no radio call found or time delta exceeds threshold
        """
        intent, _ = self.get_most_recent_intent_with_time_delta(tail_number, sequence_end_timestamp)
        return intent


##Dataloader class

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets
    Modified from https://github.com/alexmonti19/dagnet"""
    
    def __init__(
        self, data_dir, obs_len=11, pred_len=120, skip=1,step=10,
        min_agent=0, delim=' ', intent_csv_path=None, time_delta_threshold_minutes=None):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <agent_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_agent: Minimum number of agents that should be in a seqeunce
        - step: Subsampling for pred
        - delim: Delimiter in the dataset files
        - intent_csv_path: Path to transcripts_with_goals.csv for intent lookup
        - time_delta_threshold_minutes: Time threshold for intent lookup
        """
        super(TrajectoryDataset, self).__init__()
        
        # Initialize intent lookup
        self.intent_lookup = IntentLookup(intent_csv_path, time_delta_threshold_minutes)

        self.max_agents_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.step = step
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len/self.step))
        all_files = os.listdir(self.data_dir)
        all_files = sorted(all_files, key=lambda x: int(os.path.splitext(x)[0])) # sort txt files in number order - sundhar
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_agents_in_seq = []
        seq_list = []
        seq_list_rel = []
        context_list = []
        timestamp_list = []
        tail_list = []
        intent_list = []  # New list to store intent labels
        time_delta_list = []  # New list to store time delta values
        
        self.seq_file_map = []  # New list to store file indices
        
        for path_idx, path in enumerate(tqdm(all_files)):
            # print(path)
            data = read_file(path, delim)
            if ((data.ndim == 1 and len(data[0])==0) or len(data[:,0])==0):
                print("File is empty")
                continue
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_agents_in_frame = max(self.max_agents_in_frame,len(agents_in_curr_seq))
                
                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 3,
                                         self.seq_final_len))
                curr_seq = np.zeros((len(agents_in_curr_seq), 3,self.seq_final_len ))
                curr_context =  np.zeros((len(agents_in_curr_seq), 2,self.seq_final_len ))
                curr_timestamp = np.zeros((len(agents_in_curr_seq), 1, self.seq_final_len))
                curr_tail = np.zeros((len(agents_in_curr_seq), 1, self.seq_final_len))
                curr_intent = np.full((len(agents_in_curr_seq),), 15)  # Default to "Insufficient information"
                curr_time_delta = np.zeros((len(agents_in_curr_seq),))  # Default to 0 time delta
                num_agents_considered = 0
                for _, agent_id in enumerate(agents_in_curr_seq):
                    curr_agent_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 agent_id, :]
                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx
                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    obs = curr_agent_seq[:,:obs_len]
                    pred = curr_agent_seq[:,obs_len+step-1::step]
                    curr_agent_seq = np.hstack((obs,pred))
                    context = curr_agent_seq[-4:-2,:] # Wind columns
                    assert(~np.isnan(context).any())
                    timestamp = curr_agent_seq[5,:]
                    tail = curr_agent_seq[6,:]
                    
                    # Get intent label and time delta for this agent at sequence end timestamp
                    sequence_end_timestamp = timestamp[-1]  # Last timestamp in sequence
                    agent_tail_number = tail[0]  # Tail number (same for entire sequence)
                    intent_label, time_delta_seconds = self.intent_lookup.get_most_recent_intent_with_time_delta(agent_tail_number, sequence_end_timestamp)
                    
                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]

                    _idx = num_agents_considered

                    if (curr_agent_seq.shape[1]!=self.seq_final_len):
                        continue

                   
                    curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq[:3,:]
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq[:3,:]
                    curr_context[_idx,:,pad_front:pad_end] = context
                    curr_timestamp[_idx,:,pad_front:pad_end] = timestamp
                    curr_tail[_idx,:,pad_front:pad_end] = tail
                    curr_intent[_idx] = intent_label  # Store intent label for this agent
                    curr_time_delta[_idx] = time_delta_seconds  # Store time delta for this agent
                    num_agents_considered += 1
            

                if num_agents_considered > min_agent:
                    num_agents_in_seq.append(num_agents_considered)
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    context_list.append(curr_context[:num_agents_considered])
                    timestamp_list.append(curr_timestamp[:num_agents_considered])
                    tail_list.append(curr_tail[:num_agents_considered])
                    intent_list.append(curr_intent[:num_agents_considered])
                    time_delta_list.append(curr_time_delta[:num_agents_considered])
                    self.seq_file_map.append(path_idx)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        context_list = np.concatenate(context_list, axis=0)
        timestamp_list = np.concatenate(timestamp_list, axis=0)
        tail_list = np.concatenate(tail_list, axis=0)
        intent_list = np.concatenate(intent_list, axis=0)
        time_delta_list = np.concatenate(time_delta_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_context = torch.from_numpy(
            context_list[:,:,:self.obs_len]).type(torch.float)
        self.obs_timestamp = torch.from_numpy(
            timestamp_list[:,:,:self.obs_len]).type(torch.double)
        self.obs_tail = torch.from_numpy(
            tail_list[:,:,:self.obs_len]).type(torch.double)
        self.intent_labels = torch.from_numpy(
            intent_list).type(torch.long)  # Intent labels as long integers
        
        # Normalize time delta using log transform
        normalized_time_delta = np.array([math.log(1 + td) for td in time_delta_list])
        self.time_delta_features = torch.from_numpy(
            normalized_time_delta).type(torch.float)  # Normalized time delta features
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.max_agents = -float('Inf')
        for (start, end) in self.seq_start_end:
            n_agents = end - start
            self.max_agents = n_agents if n_agents > self.max_agents else self.max_agents

    def __len__(self):
        return self.num_seq
    
    def __max_agents__(self):
        return self.max_agents

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :], self.obs_context[start:end, :],
            self.intent_labels[start:end],  # Add intent labels
            self.time_delta_features[start:end],  # Add time delta features
            self.obs_timestamp[start:end, :],
            self.obs_tail[start:end, :]
        ]
        return out
    
    def get_file_index(self, seq_index):
        return self.seq_file_map[seq_index]

### Metrics 

def ade(y1,y2):
    """
    y: (seq_len,2)
    """

    loss = y1 -y2
    loss = loss**2
    loss = np.sqrt(np.sum(loss,1))

    return np.mean(loss)

def fde(y1,y2):
    loss = (y1[-1,:] - y2[-1,:])**2
    return np.sqrt(np.sum(loss))

def rel_to_abs(obs,rel_pred):

    pred = rel_pred.copy()
    pred[0] += obs[-1]
    for i in range(1,len(pred)):
        pred[i] += pred[i-1]
    
    return pred 

def rmse(y1,y2):
    criterion = nn.MSELoss()

    # return loss
    return torch.sqrt(criterion(y1, y2))

## General utils

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    elif delim == 'comma':
        delim = ','
    with open(_path, 'r') as f:
        for line in f:
            if(line != '\n'):
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
    if data == []:
        return np.empty((0,5))
    return np.asarray(data)

def acc_to_abs(acc,obs,delta=1):
    acc = acc.permute(2,1,0)
    pred = torch.empty_like(acc)
    pred[0] = 2*obs[-1] - obs[0] + acc[0]
    pred[1] = 2*pred[0] - obs[-1] + acc[1]
    
    for i in range(2,acc.shape[0]):
        pred[i] = 2*pred[i-1] - pred[i-2] + acc[i]
    return pred.permute(2,1,0)
    

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,context_list,intent_list,time_delta_list,timestamp_list,tail_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    context = torch.cat(context_list, dim=0 ).permute(2,0,1)
    intent_labels = torch.cat(intent_list, dim=0)  # Intent labels: [total_agents]
    time_delta_features = torch.cat(time_delta_list, dim=0)  # Time delta features: [total_agents]
    timestamp = torch.cat(timestamp_list, dim=0 ).permute(2,0,1)
    tail = torch.cat(tail_list, dim=0 ).permute(2,0,1)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, intent_labels, time_delta_features, seq_start_end
    ]
    return tuple(out)


def loss_func(recon_y,y,mean,log_var):
    traj_loss = rmse(recon_y,y)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return traj_loss + KLD


