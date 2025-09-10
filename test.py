import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import torch
from torch.utils.data import DataLoader

from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset, seq_collate

def save_inference_csv(detailed_results_df, dataset_name, model_path):
    """Save CSV with columns: index, inference_num, ADE, FDE, intent_label, time_delta"""
    model_name = os.path.basename(model_path).replace('.pth', '').replace('.pt', '')
    
    # Select only the required columns in the specified order
    csv_columns = ['index', 'inference_num', 'ade', 'fde', 'intent_label', 'time_delta']
    output_df = detailed_results_df[csv_columns].copy()
    
    # Save the CSV
    csv_path = f'inference_results_{dataset_name}_{model_name}.csv'
    output_df.to_csv(csv_path, index=False)
    print(f"Inference results CSV saved to: {csv_path}")
    return csv_path

def create_analysis_plots(detailed_results_df, dataset_name, model_path):
    model_name = os.path.basename(model_path).replace('.pth', '').replace('.pt', '')
    
    # Filter out rows with missing timestamp data for correlation analysis
    valid_data = detailed_results_df.dropna(subset=['time_delta'])
    
    print(f"\nDetailed Analysis:")
    print(f"Total predictions: {len(detailed_results_df)}")
    print(f"Predictions with radio call data: {len(valid_data)}")
    print(f"Predictions missing radio call data: {len(detailed_results_df) - len(valid_data)}")
    
    if len(valid_data) > 0:
        # Calculate correlation
        correlation, p_value = stats.pearsonr(valid_data['time_delta'], valid_data['ade'])
        print(f"Pearson correlation (time_delta vs ADE): {correlation:.4f} (p-value: {p_value:.4f})")
    
    # Create comprehensive plot layout
    fig = plt.figure(figsize=(20, 12))
    
    # 1. ADE distribution across inference runs
    ax1 = plt.subplot(2, 4, 1)
    inference_ade_data = [detailed_results_df[detailed_results_df['inference_num'] == i]['ade'].values 
                         for i in range(1, 6)]
    ax1.boxplot(inference_ade_data, labels=[f'Run {i}' for i in range(1, 6)])
    ax1.set_title('ADE Distribution Across 5 Inference Runs')
    ax1.set_ylabel('ADE Error')
    ax1.grid(True, alpha=0.3)
    
    # 2. FDE distribution across inference runs  
    ax2 = plt.subplot(2, 4, 2)
    inference_fde_data = [detailed_results_df[detailed_results_df['inference_num'] == i]['fde'].values 
                         for i in range(1, 6)]
    ax2.boxplot(inference_fde_data, labels=[f'Run {i}' for i in range(1, 6)])
    ax2.set_title('FDE Distribution Across 5 Inference Runs')
    ax2.set_ylabel('FDE Error')
    ax2.grid(True, alpha=0.3)
    
    # 3. Overall ADE distribution
    ax3 = plt.subplot(2, 4, 3)
    ax3.boxplot(detailed_results_df['ade'])
    ax3.set_title(f'Overall ADE Distribution\nDataset: {dataset_name}')
    ax3.set_ylabel('ADE Error')
    ax3.grid(True, alpha=0.3)
    
    # 4. Overall FDE distribution
    ax4 = plt.subplot(2, 4, 4)
    ax4.boxplot(detailed_results_df['fde'])
    ax4.set_title(f'Overall FDE Distribution\nDataset: {dataset_name}')
    ax4.set_ylabel('FDE Error')
    ax4.grid(True, alpha=0.3)
    
    if len(valid_data) > 0:
        # 5. Scatter plot: Time delta vs ADE
        ax5 = plt.subplot(2, 4, 5)
        scatter = ax5.scatter(valid_data['time_delta'], valid_data['ade'], 
                            alpha=0.6, s=10, c=valid_data['inference_num'], cmap='viridis')
        ax5.set_xlabel('Time Delta (seconds)')
        ax5.set_ylabel('ADE Error')
        ax5.set_title(f'Time Delta vs ADE\nr={correlation:.3f}, p={p_value:.3f}')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Inference Run')
        
        # Add trend line
        z = np.polyfit(valid_data['time_delta'], valid_data['ade'], 1)
        p = np.poly1d(z)
        ax5.plot(valid_data['time_delta'], p(valid_data['time_delta']), "r--", alpha=0.8)
        
        # 6. Time delta distribution
        ax6 = plt.subplot(2, 4, 6)
        ax6.hist(valid_data['time_delta'], bins=30, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Time Delta (seconds)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Distribution of Time Deltas')
        ax6.grid(True, alpha=0.3)
        
        # 7. ADE vs Time Delta (binned)
        ax7 = plt.subplot(2, 4, 7)
        
        # Create time bins
        time_bins = pd.cut(valid_data['time_delta'], bins=10)
        binned_stats = valid_data.groupby(time_bins)['ade'].agg(['mean', 'std', 'count'])
        bin_centers = [(interval.left + interval.right) / 2 for interval in binned_stats.index]
        
        ax7.errorbar(bin_centers, binned_stats['mean'], yerr=binned_stats['std'], 
                    fmt='o-', capsize=5, capthick=2)
        ax7.set_xlabel('Time Delta (seconds)')
        ax7.set_ylabel('Mean ADE ± Std')
        ax7.set_title('ADE vs Time Delta (Binned)')
        ax7.grid(True, alpha=0.3)
        
        # 8. ADE statistics per inference run
        ax8 = plt.subplot(2, 4, 8)
        
        inference_stats = detailed_results_df.groupby('inference_num')['ade'].agg(['mean', 'std'])
        x_pos = np.arange(len(inference_stats))
        
        bars = ax8.bar(x_pos, inference_stats['mean'], yerr=inference_stats['std'], 
                      capsize=5, alpha=0.7)
        ax8.set_xlabel('Inference Run')
        ax8.set_ylabel('Mean ADE ± Std')
        ax8.set_title('ADE Statistics per Inference Run')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels([f'Run {i}' for i in range(1, 6)])
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, inference_stats['mean'], inference_stats['std'])):
            ax8.text(bar.get_x() + bar.get_width()/2., mean_val + std_val + 0.05,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        # If no valid timestamp data, show message
        ax5 = plt.subplot(2, 4, 5)
        ax5.text(0.5, 0.5, 'No timestamp correlation data available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Time Delta vs ADE')
        
        ax6 = plt.subplot(2, 4, 6)
        ax6.text(0.5, 0.5, 'No timestamp data available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Time Delta Distribution')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_path = f'test_analysis_{dataset_name}_{model_name}_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nComprehensive analysis plots saved to: {output_path}")
    
    # Save detailed results CSV
    csv_path = f'test_results_{dataset_name}_{model_name}_detailed.csv'
    detailed_results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")

def main():
    
    parser=argparse.ArgumentParser(description='Test TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--epoch',type=int,required=False)
    parser.add_argument('--use_config',action='store_true',help='Use config.yaml instead of argparse for parameters')

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

    parser.add_argument('--model_path', type=str , required=True)
    
    # Intent parameters
    parser.add_argument('--intent_embed_dim', type=int, default=32)
    parser.add_argument('--num_intent_classes', type=int, default=16)
    parser.add_argument('--intent_csv_path', type=str, default="../main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv")
    parser.add_argument('--intent_time_delta_threshold_minutes', type=float, default=10.0)
    parser.add_argument('--use_time_delta_feature', type=bool, default=True)
    
    args=parser.parse_args()

    # Override with config if requested
    if args.use_config:
        try:
            from config_loader import get_training_args, load_config
            config_args = get_training_args()
            config_loader = load_config()
            
            # Override key parameters with config values
            args.intent_csv_path = config_args.intent_csv_path
            args.intent_embed_dim = config_args.intent_embed_dim
            args.num_intent_classes = config_args.num_intent_classes
            args.intent_time_delta_threshold_minutes = config_args.intent_time_delta_threshold_minutes
            args.use_time_delta_feature = config_args.use_time_delta_feature
            
            # Override dataset paths
            dataset_paths = config_loader.get_dataset_paths()
            args.dataset_folder = dataset_paths['test'].rsplit('/', 2)[0] + '/'  # Extract base path
            args.dataset_name = config_args.dataset_name
            
            print("Using configuration from config.yaml")
        except ImportError:
            print("Warning: config_loader not available, using argparse defaults")
        except Exception as e:
            print(f"Warning: Could not load config: {e}, using argparse defaults")

    ##Select device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load data

    datapath = args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim, skip=args.skip,
                                    intent_csv_path=args.intent_csv_path, time_delta_threshold_minutes=args.intent_time_delta_threshold_minutes)
    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=4,shuffle=False,collate_fn=seq_collate)

    ##Load model
    model = TrajAirNet(args)
    model.to(device)

    model_path =  args.model_path


    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    (average_ade, average_fde,
            all_ade_mean, all_ade_var, all_ade_std,
            all_fde_mean, all_fde_var, all_fde_std, 
            detailed_results_df) = test(model, loader_test, device, dataset_test)

    print(f"Test Results (Best per batch):")
    print(f"  ADE: {average_ade:.4f}")
    print(f"  FDE: {average_fde:.4f}")
    print(f"\nTest Results (All inferences):")
    print(f"  ADE Mean: {all_ade_mean:.4f}")
    print(f"  ADE Variance: {all_ade_var:.4f}")
    print(f"  ADE Std Dev: {all_ade_std:.4f}")
    print(f"  FDE Mean: {all_fde_mean:.4f}")
    print(f"  FDE Variance: {all_fde_var:.4f}")
    print(f"  FDE Std Dev: {all_fde_std:.4f}")
    
    # Save the main CSV with required columns
    save_inference_csv(detailed_results_df, args.dataset_name, args.model_path)
    
    # Create analysis plots (optional)
    create_analysis_plots(detailed_results_df, args.dataset_name, args.model_path)

def test(model, loader_test, device, dataset=None):
    """
    Test function that can be called from train.py or standalone.
    If dataset is None, will try to extract it from the loader (for backwards compatibility).
    """
    if dataset is None:
        # Backwards compatibility - try to get dataset from loader
        dataset = loader_test.dataset if hasattr(loader_test, 'dataset') else None
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_batch = 0
    
    all_ade = []
    all_fde = []
    detailed_results = []
    loss_records = []
    
    for batch_idx, batch in enumerate(tqdm(loader_test)):
        tot_batch += 1
        batch = [tensor.to(device) for tensor in batch]
        # Handle both old and new batch formats (with or without time_delta_features)
        if len(batch) == 7:
            obs_traj_all, pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, intent_labels, seq_start = batch
            time_delta_features = None
        else:
            obs_traj_all, pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, intent_labels, time_delta_features, seq_start = batch 
        num_agents = obs_traj_all.shape[1]
        
        # Get timestamps and tail numbers for this batch
        batch_data = dataset[batch_idx]
        # Handle both old and new dataset formats
        if len(batch_data) == 8:
            timestamp_data = batch_data[6]  # obs_timestamp
            tail_data = batch_data[7]       # obs_tail
        else:
            timestamp_data = batch_data[7]  # obs_timestamp (with time_delta_features)
            tail_data = batch_data[8]       # obs_tail (with time_delta_features)
        
        best_ade_loss = float('inf')
        best_fde_loss = float('inf')
        
        # Store ADE values for each of 5 inferences
        batch_ade_runs = []
        
        # Run 5 inferences for this batch
        for inference_run in range(5):
            z = torch.randn([1, 1, 128]).to(device)
            adj = torch.ones((num_agents, num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all, 1, 2), z, adj, 
                                        torch.transpose(context, 1, 2), intent_labels, time_delta_features)
            
            batch_ade_loss = 0
            batch_fde_loss = 0
            agent_ade_values = []
            
            for agent in range(num_agents):
                obs_traj = np.squeeze(obs_traj_all[:, agent, :].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:, agent, :].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()
                
                agent_ade = ade(recon_pred, pred_traj)
                agent_fde = fde(recon_pred, pred_traj)
                
                agent_ade_values.append(agent_ade)
                batch_ade_loss += agent_ade
                batch_fde_loss += agent_fde
                
                # Get time delta and intent information
                agent_tail = tail_data[agent, 0, -1].item()  # Tail number
                pred_start_timestamp = timestamp_data[agent, 0, -1].item()  # Last observed timestamp
                agent_intent = intent_labels[agent].item() if agent < len(intent_labels) else None
                
                # Get time delta - use normalized feature if available, otherwise calculate raw delta
                if time_delta_features is not None and agent < len(time_delta_features):
                    # Convert normalized time delta back to seconds: exp(normalized_value) - 1
                    normalized_time_delta = time_delta_features[agent].item()
                    raw_time_delta = np.exp(normalized_time_delta) - 1
                else:
                    # Fallback: calculate raw time delta from timestamps
                    intent_lookup = dataset.intent_lookup
                    radio_call_timestamp = None
                    
                    # Find most recent radio call before prediction
                    tail_str = intent_lookup._convert_tail_to_string(agent_tail)
                    if tail_str in intent_lookup.intent_data:
                        for timestamp, intent in intent_lookup.intent_data[tail_str]:
                            if timestamp <= pred_start_timestamp:
                                radio_call_timestamp = timestamp
                            else:
                                break
                    
                    # Calculate time delta (in seconds)
                    raw_time_delta = None
                    if radio_call_timestamp is not None:
                        raw_time_delta = pred_start_timestamp - radio_call_timestamp
                
                # Store detailed results
                detailed_results.append({
                    'index': batch_idx,
                    'inference_num': inference_run + 1,  # 1-indexed for clarity
                    'ade': agent_ade,
                    'fde': agent_fde,
                    'intent_label': agent_intent,
                    'time_delta': raw_time_delta,
                    # Additional metadata for analysis
                    'agent_idx': agent,
                    'tail_number': agent_tail,
                    'pred_start_timestamp': pred_start_timestamp
                })
            
            batch_ade_loss = batch_ade_loss / num_agents
            batch_fde_loss = batch_fde_loss / num_agents
            batch_ade_runs.append(batch_ade_loss)
            
            all_ade.append(batch_ade_loss)
            all_fde.append(batch_fde_loss)
            
            if batch_ade_loss < best_ade_loss:
                best_ade_loss = batch_ade_loss
                best_fde_loss = batch_fde_loss

        tot_ade_loss += best_ade_loss
        tot_fde_loss += best_fde_loss
        loss_records.append((batch_idx, best_ade_loss, best_fde_loss))
        
    average_ade = tot_ade_loss / tot_batch
    average_fde = tot_fde_loss / tot_batch

    all_ade_mean = np.mean(all_ade)
    all_ade_var = np.var(all_ade)
    all_ade_std = np.std(all_ade)
    all_fde_mean = np.mean(all_fde)
    all_fde_var = np.var(all_fde)
    all_fde_std = np.std(all_fde)    

    worst_cases = sorted(loss_records, key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Worst Cases (By ADE):")
    for idx, ade_loss, fde_loss in worst_cases:
        print(f"Batch Index: {idx}, ADE: {ade_loss:.4f}, FDE: {fde_loss:.4f}")

    # Convert detailed results to DataFrame
    detailed_results_df = pd.DataFrame(detailed_results)

    return (average_ade, average_fde,
            all_ade_mean, all_ade_var, all_ade_std,
            all_fde_mean, all_fde_var, all_fde_std, 
            detailed_results_df)


if __name__=='__main__':
    main()

