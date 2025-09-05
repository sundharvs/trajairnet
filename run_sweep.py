#!/usr/bin/env python3
"""
Wandb Sweep Runner for Belief-Aware TrajAirNet

This script runs hyperparameter sweeps using Weights & Biases.
"""

import wandb
from train_with_beliefs import train

def sweep_train():
    """Wrapper function for wandb sweeps."""
    # Initialize wandb run (this gets called by the sweep agent)
    wandb.init(project="belief-state-integration")
    
    # Run training with wandb config
    train(use_wandb_config=True)
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    # This will be called by: wandb agent <sweep_id>
    sweep_train()