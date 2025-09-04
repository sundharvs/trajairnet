#!/usr/bin/env python3
"""
Start a Wandb Sweep for Belief-Aware TrajAirNet

Usage:
    python start_sweep.py [--config sweep_config.yaml] [--project trajairnet-beliefs]
"""

import argparse
import wandb
import yaml

def main():
    parser = argparse.ArgumentParser(description='Start a wandb sweep')
    parser.add_argument('--config', type=str, default='sweep_config.yaml',
                       help='Path to sweep configuration file')
    parser.add_argument('--project', type=str, default='trajairnet-beliefs',
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    # Load sweep configuration
    with open(args.config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    
    print(f"Created sweep: {sweep_id}")
    print(f"Run the following command to start an agent:")
    print(f"wandb agent {sweep_id}")
    
    return sweep_id

if __name__ == "__main__":
    main()