"""
Configuration loader utility for TrajAirNet project.
Provides unified configuration management with support for:
- YAML configuration files
- Wandb sweep parameter integration  
- Argparse fallback for testing scripts
"""

import yaml
import os
from typing import Dict, Any, Optional
import argparse
import torch


class ConfigLoader:
    """Unified configuration loader for TrajAirNet project."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_training_config(self, wandb_config: Optional[Dict] = None) -> argparse.Namespace:
        """
        Get training configuration with wandb sweep parameter override support.
        
        Args:
            wandb_config: Wandb configuration dict (from wandb.config)
            
        Returns:
            argparse.Namespace with all training parameters
        """
        # Base configuration from YAML
        config_dict = {
            # Dataset params
            'dataset_folder': self.config['dataset']['folder'],
            'dataset_name': self.config['dataset']['name'],
            'obs': self.config['dataset']['obs_len'],
            'preds': self.config['dataset']['pred_len'],
            'preds_step': self.config['dataset']['pred_step'],
            'delim': self.config['dataset']['delim'],
            
            # Network params
            'input_channels': self.config['network']['input_channels'],
            'tcn_channel_size': self.config['network']['tcn_channel_size'],
            'tcn_layers': self.config['network']['tcn_layers'],
            'tcn_kernels': self.config['network']['tcn_kernels'],
            'num_context_input_c': self.config['network']['num_context_input_c'],
            'num_context_output_c': self.config['network']['num_context_output_c'],
            'cnn_kernels': self.config['network']['cnn_kernels'],
            'gat_heads': self.config['network']['gat_heads'],
            'graph_hidden': self.config['network']['graph_hidden'],
            'dropout': self.config['network']['dropout'],
            'alpha': self.config['network']['alpha'],
            'cvae_hidden': self.config['network']['cvae_hidden'],
            'cvae_channel_size': self.config['network']['cvae_channel_size'],
            'cvae_layers': self.config['network']['cvae_layers'],
            'mlp_layer': self.config['network']['mlp_layer'],
            
            # Intent params
            'intent_embed_dim': self.config['intent']['embed_dim'],
            'num_intent_classes': self.config['intent']['num_classes'],
            'intent_csv_path': self.config['intent']['csv_path'],
            'intent_time_delta_threshold_minutes': self.config['intent']['time_delta_threshold_minutes'],
            'use_time_delta_feature': self.config['intent']['use_time_delta_feature'],
            
            # Training params
            'lr': self.config['training']['lr'],
            'total_epochs': self.config['training']['total_epochs'],
            'evaluate': self.config['training']['evaluate'],
            'save_model': self.config['training']['save_model'],
            'model_pth': self.config['training']['model_path'],
            'scheduler_type': self.config['training']['scheduler_type'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'force_single_gpu': self.config['training']['force_single_gpu'],
            'grad_accum_steps': self.config['training']['grad_accum_steps'],
            
            # Wandb params
            'use_wandb': self.config['wandb']['use_wandb'],
            'wandb_project': self.config['wandb']['project'],
            'wandb_entity': self.config['wandb']['entity'],
            
            # DataLoader params
            'train_batch_size': self.config['dataloader']['train_batch_size'],
            'test_batch_size': self.config['dataloader']['test_batch_size'],
            'train_num_workers': self.config['dataloader']['train_num_workers'],
            'test_num_workers': self.config['dataloader']['test_num_workers'],
            'shuffle_train': self.config['batch']['shuffle_train'],
            'shuffle_test': self.config['batch']['shuffle_test']
        }
        
        # Override with wandb sweep parameters if provided
        if wandb_config:
            for key, value in wandb_config.items():
                if key in config_dict:
                    config_dict[key] = value
        
        args = argparse.Namespace(**config_dict)
        
        # Add device configuration
        if self.config['device']['force_cpu']:
            args.device = 'cpu'
        elif self.config['device']['use_cuda']:
            args.device = 'cuda'
        else:
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        return args
    
    def get_overfit_config(self, transcripts_path: Optional[str] = None) -> argparse.Namespace:
        """
        Get configuration for overfit testing.
        
        Args:
            transcripts_path: Optional override for transcripts path
            
        Returns:
            argparse.Namespace with overfit test parameters
        """
        config_dict = {
            # Dataset params
            'dataset_folder': self.config['dataset']['folder'],
            'dataset_name': self.config['dataset']['name'],
            'obs': self.config['dataset']['obs_len'],
            'preds': self.config['dataset']['pred_len'],
            'preds_step': self.config['dataset']['pred_step'],
            'delim': self.config['dataset']['delim'],
            'batch_size': self.config['batch']['size'],
            
            # Network params (same as training)
            'input_channels': self.config['network']['input_channels'],
            'tcn_channel_size': self.config['network']['tcn_channel_size'],
            'tcn_layers': self.config['network']['tcn_layers'],
            'tcn_kernels': self.config['network']['tcn_kernels'],
            'num_context_input_c': self.config['network']['num_context_input_c'],
            'num_context_output_c': self.config['network']['num_context_output_c'],
            'cnn_kernels': self.config['network']['cnn_kernels'],
            'gat_heads': self.config['network']['gat_heads'],
            'graph_hidden': self.config['network']['graph_hidden'],
            'dropout': self.config['network']['dropout'],
            'alpha': self.config['network']['alpha'],
            'cvae_hidden': self.config['network']['cvae_hidden'],
            'cvae_channel_size': self.config['network']['cvae_channel_size'],
            'cvae_layers': self.config['network']['cvae_layers'],
            'mlp_layer': self.config['network']['mlp_layer'],
            
            # Belief params
            'belief_embed_dim': self.config['belief']['embed_dim'],
            'belief_vocab_size': self.config['belief']['vocab_size'],
            'belief_integration_mode': self.config['belief']['integration_mode'],
            'transcripts_path': transcripts_path or self.config['belief']['transcripts_path'],
            
            # Overfit-specific training params
            'lr': 1e-4,  # Smaller LR for overfitting
            'epochs': 500,
        }
        
        return argparse.Namespace(**config_dict)
    
    def get_dataset_paths(self) -> Dict[str, str]:
        """Get standardized dataset paths."""
        base_path = self.config['dataset']['folder'] + self.config['dataset']['name'] + "/processed_data/"
        return {
            'train': base_path + "train",
            'test': base_path + "test",
            'base': base_path
        }
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get model file paths."""
        return {
            'saved_models': self.config['paths']['saved_models'],
            'pretrained_model': self.config['paths']['pretrained_model']
        }


# Convenience functions for backward compatibility and easy usage
def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """Load configuration from file."""
    return ConfigLoader(config_path)


def get_training_args(wandb_config: Optional[Dict] = None, config_path: str = "config.yaml") -> argparse.Namespace:
    """
    Get training arguments from config with optional wandb override.
    
    Args:
        wandb_config: Optional wandb configuration for sweeps
        config_path: Path to config file
        
    Returns:
        Training arguments as argparse.Namespace
    """
    loader = ConfigLoader(config_path)
    return loader.get_training_config(wandb_config)


def get_overfit_args(transcripts_path: Optional[str] = None, config_path: str = "config.yaml") -> argparse.Namespace:
    """
    Get overfit test arguments from config.
    
    Args:
        transcripts_path: Optional override for transcripts path
        config_path: Path to config file
        
    Returns:
        Overfit test arguments as argparse.Namespace
    """
    loader = ConfigLoader(config_path)
    return loader.get_overfit_config(transcripts_path)