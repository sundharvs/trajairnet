#!/usr/bin/env python3

import wandb
import yaml
from train_sweep import train_with_config
import sys

def run_sweep():
    """
    Run a wandb sweep for intent-interaction-matrix hyperparameter tuning
    """
    # Initialize wandb run
    wandb.init()
    
    # Get hyperparameters from wandb config
    config = wandb.config
    
    # Set up full configuration with defaults and sweep parameters
    full_config = {
        # Dataset params
        'dataset_folder': '/dataset/',
        'dataset_name': '7days1',
        'obs': config.obs,
        'preds': config.preds,
        'preds_step': config.preds_step,
        
        # Network params
        'input_channels': 3,
        'tcn_channel_size': config.tcn_channel_size,
        'tcn_layers': config.tcn_layers,
        'tcn_kernels': 4,
        
        'num_context_input_c': 2,
        'num_context_output_c': 7,
        'cnn_kernels': 2,
        
        'gat_heads': config.gat_heads,
        'graph_hidden': config.graph_hidden,
        'dropout': config.dropout,
        'alpha': config.alpha,
        'cvae_hidden': config.cvae_hidden,
        'cvae_channel_size': config.cvae_channel_size,
        'cvae_layers': 2,
        'mlp_layer': 32,
        
        # Intent embedding params
        'intent_embed_dim': config.intent_embed_dim,
        'num_intent_classes': 16,
        
        # Training params
        'lr': config.lr,
        'total_epochs': config.total_epochs,
        'delim': ' ',
        'evaluate': True,
        'save_model': False,  # Disable model saving for sweep runs
        'model_pth': "/saved_models/",
        
        # Wandb params
        'use_wandb': True,
        'wandb_project': "trajairnet-intent-interaction-matrix",
        'wandb_entity': None
    }
    
    # Run training with the current configuration
    try:
        train_with_config(full_config)
    except Exception as e:
        print(f"Training failed with error: {e}")
        wandb.log({"training_error": str(e)})

def main():
    """
    Initialize and run the wandb sweep
    """
    # Load sweep configuration
    with open('sweep_config_intent_interaction_matrix.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="trajairnet-intent-interaction-matrix")
    
    # Start sweep agent
    print(f"Starting sweep with ID: {sweep_id}")
    print("Run the following command to start sweep agents:")
    print(f"wandb agent {sweep_id}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        # Run as sweep agent
        run_sweep()
    else:
        # Initialize sweep
        main()