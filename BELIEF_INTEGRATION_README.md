# Belief-Aware Trajectory Prediction System

This document describes the implementation of a dynamic belief state system that integrates pilot intent information from radio communications into TrajAirNet for enhanced socially-aware trajectory prediction.

## Overview

The belief-aware system moves beyond discrete intent categories to dynamic belief states that evolve over time as new radio communications are received. This addresses key limitations of previous approaches:

1. **Multi-intent radio calls** - Single calls containing sequential intents (e.g., "downwind for touch and go")
2. **Belief evolution** - Updates rather than replacements (e.g., "going around" changes landing plan)
3. **Temporal continuity** - Maintaining aircraft state over time

## Architecture

### Core Components

1. **BeliefState** - Represents pilot intent as a sequence of future actions
2. **BeliefManager** - Centralized system for tracking and updating pilot beliefs
3. **LLMBeliefUpdater** - GPT-4o + Gemma-3 pipeline for belief updates
4. **BeliefEncoder** - Neural network encoder for variable-length belief sequences
5. **BeliefAwareGAT** - Enhanced GAT with belief-aware attention
6. **BeliefAwareTrajAirNet** - Complete trajectory prediction model

### Data Flow

```
Radio Call → GPT-4o (clean transcript) → Gemma-3 (update belief) → BeliefState
    ↓
BeliefState → BeliefEncoder → belief_embedding
    ↓
spatial_features + belief_embedding → BeliefAwareGAT → enhanced_features
    ↓
enhanced_features → CVAE → predicted_trajectory
```

## Key Features

### Expanded Intent Vocabulary (33 categories)
- Traffic pattern operations: upwind, crosswind, downwind, base, final (×2 runways)
- Runway operations: takeoff, landing, touch-and-go (×2 runways)
- Pattern entry/exit: 45°, straight-in, teardrop, departures, go-around
- Ground operations: taxi, hold short, clear runway
- Meta states: unknown, insufficient info, other

### Dynamic Belief Updates
- **Radio Call Listener** - Processes new communications and updates beliefs
- **Temporal Tracking** - Maintains belief history for each aircraft
- **Graceful Degradation** - Functions without belief data when unavailable

### Neural Architecture Enhancements
- **Variable-Length Encoding** - LSTM-based encoder for belief sequences
- **Attention Integration** - Belief information modifies GAT attention weights
- **Multiple Integration Modes** - Concatenation, addition, or gated fusion

## File Structure

```
trajairnet/
├── model/
│   ├── belief_states.py           # Core belief state classes and encoder
│   ├── belief_manager.py          # Centralized belief tracking system
│   ├── llm_belief_updater.py      # GPT-4o + Gemma-3 belief updates
│   ├── belief_aware_gat.py        # Enhanced GAT with belief awareness
│   └── belief_trajairnet.py       # Complete belief-aware model
├── train_with_beliefs.py          # Training script with belief integration
├── test_with_beliefs.py           # Testing script for evaluation
└── test_belief_pipeline.py        # End-to-end pipeline testing
```

## Usage

### Training

```bash
python train_with_beliefs.py \
  --dataset_name 7days1 \
  --belief_embed_dim 64 \
  --belief_integration_mode concatenate \
  --transcripts_path ../main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv
```

### Testing

```bash
python test_with_beliefs.py \
  --dataset_name 7days1 \
  --epoch 10 \
  --transcripts_path ../main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv
```

### Pipeline Validation

```bash
python test_belief_pipeline.py
```

## Implementation Details

### BeliefState Structure
```python
belief = BeliefState([
    'downwind_26',  # Current intent
    'base_26',      # Next planned action  
    'final_26',     # Future action
    'land_26'       # Final goal
])
```

### LLM Integration
- **GPT-4o** - Cleans and standardizes ASR transcripts
- **Gemma-3** - Updates belief sequences based on radio calls
- **Fallback Mode** - Rule-based updates when LLMs unavailable

### Attention Mechanism
```python
# Spatial attention (standard GAT)
spatial_attention = softmax(LeakyReLU(W * [h_i || h_j]))

# Belief attention
belief_attention = softmax(LeakyReLU(W * [belief_i || belief_j]))

# Combined attention  
total_attention = (1-β) * spatial_attention + β * belief_attention
```

## Advantages Over Previous Approaches

1. **Rich Semantic Information** - Full belief sequences vs. single discrete labels
2. **Temporal Evolution** - Beliefs update over time vs. static assignments
3. **Multi-Intent Support** - Sequences capture complex pilot plans
4. **Flexible Integration** - Multiple ways to combine with spatial features
5. **Graceful Degradation** - Works without belief data

## Testing and Validation

The implementation includes comprehensive testing:
- **Unit Tests** - Each component tested independently
- **Integration Tests** - End-to-end pipeline validation
- **Ablation Studies** - With/without belief information comparison
- **Real Data Compatibility** - Tested with actual trajectory sequences

## Future Enhancements

1. **Confidence Scores** - Add uncertainty quantification to belief states
2. **Multi-Modal Integration** - Combine with other data sources (weather, traffic)
3. **Online Learning** - Adapt belief models based on observed trajectories
4. **Hierarchical Beliefs** - Multi-level intent representation

## Dependencies

- PyTorch >= 1.9
- pandas, numpy, tqdm
- openai (for GPT-4o integration)
- llama-cpp-python (for Gemma-3 integration)

## Notes

- LLM components will use rule-based fallback if API keys/models unavailable
- Belief data is optional - model functions without it
- All hyperparameters are configurable via command line arguments
- Compatible with existing TrajAirNet infrastructure