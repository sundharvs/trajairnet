# Wandb Hyperparameter Sweeps for Belief-Aware TrajAirNet

## Quick Start

1. **Start a sweep:**
   ```bash
   python start_sweep.py
   ```
   This will output a sweep ID.

2. **Run sweep agents:**
   ```bash
   wandb agent <sweep_id>
   ```
   Run this command on one or more machines to start hyperparameter search.

## Configuration

The sweep tunes these belief-specific hyperparameters:
- `lr`: Learning rate (0.0005 to 0.005)
- `belief_embed_dim`: Belief embedding dimension (32, 64, 128, 256)  
- `belief_integration_mode`: How to integrate beliefs ('concatenate', 'add', 'gated')
- `total_epochs`: Training epochs (10, 20, 30, 50)

## Files

- `sweep_config.yaml`: Sweep configuration
- `start_sweep.py`: Creates a new sweep
- `run_sweep.py`: Runs training for sweep (called by wandb agent)
- `train_with_beliefs.py`: Main training script (modified for sweeps)

## Custom Configuration

Edit `sweep_config.yaml` to modify:
- Search method (`bayes`, `grid`, `random`)
- Parameter ranges
- Early termination settings
- Optimization metric

## Example Commands

```bash
# Start sweep with custom config
python start_sweep.py --config my_config.yaml --project my-project

# Run multiple agents in parallel
wandb agent <sweep_id> &
wandb agent <sweep_id> &
wandb agent <sweep_id> &
```