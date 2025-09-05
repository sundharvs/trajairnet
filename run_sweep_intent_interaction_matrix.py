#!/usr/bin/env python3
import wandb
import yaml
from train_sweep import train_with_config
import sys

def run_sweep():
    """
    Called by wandb agent. Pulls config from sweep and runs training.
    """
    wandb.init()
    sweep_config = dict(wandb.config)  # sweep overrides only
    train_with_config(sweep_config)


def main():
    """
    Creates a new sweep.
    """
    with open('sweep_config_intent_interaction_matrix.yaml', 'r') as f:
        sweep_yaml = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_yaml, project="trajairnet-intent-interaction-matrix")
    print(f"Created sweep {sweep_id}")
    print(f"Start agents with: wandb agent {sweep_id}")


if __name__ == "__main__":
    if any(arg.startswith("--") for arg in sys.argv[1:]):
        run_sweep()
    else:
        main()
