"""
Belief State Management for Intent-Aware Trajectory Prediction

This module implements a dynamic belief state system that tracks pilot intentions
from radio communications and updates them over time as new information arrives.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

# Expanded intent vocabulary covering terminal airspace operations
INTENT_VOCABULARY = {
    # Traffic Pattern Operations (10)
    'upwind_8': 0,
    'upwind_26': 1, 
    'crosswind_8': 2,
    'crosswind_26': 3,
    'downwind_8': 4,
    'downwind_26': 5,
    'base_8': 6,
    'base_26': 7,
    'final_8': 8,
    'final_26': 9,
    
    # Runway Operations (4)
    'takeoff_8': 10,
    'takeoff_26': 11,
    'land_8': 12,
    'land_26': 13,
    
    # Departure Directions (4)
    'depart_north': 14,
    'depart_south': 15,
    'depart_east': 16,
    'depart_west': 17,

    # Ground Operations (1)
    'clear_of_runway_8': 18,
    'clear_of_runway_26': 19,
    
    # Meta State (1)
    'other': 20
}

# Reverse mapping for decoding
INTENT_NAMES = {v: k for k, v in INTENT_VOCABULARY.items()}
VOCAB_SIZE = len(INTENT_VOCABULARY)

# Special tokens
PAD_TOKEN = VOCAB_SIZE
UNK_TOKEN = VOCAB_SIZE + 1
TOTAL_VOCAB_SIZE = VOCAB_SIZE + 2

class BeliefState:
    """
    Represents a pilot's current belief state - a sequence of intended future actions.
    
    Example: ["downwind_26", "base_26", "final_26", "land_26", "clear_runway"]
    """
    
    def __init__(self, intent_sequence: List[str] = None, timestamp: float = None):
        """
        Args:
            intent_sequence: List of intent names in temporal order
            timestamp: Unix timestamp when this belief was created/updated
        """
        self.intent_sequence = intent_sequence or []
        self.timestamp = timestamp or datetime.now().timestamp()
        self.radio_call_history = []  # Store originating radio calls for debugging
    
    def to_indices(self) -> List[int]:
        """Convert intent names to vocabulary indices."""
        return [INTENT_VOCABULARY.get(intent, UNK_TOKEN) for intent in self.intent_sequence]
    
    @classmethod
    def from_indices(cls, indices: List[int], timestamp: float = None):
        """Create BeliefState from vocabulary indices."""
        intent_sequence = [INTENT_NAMES.get(idx, 'unknown') for idx in indices 
                          if idx < VOCAB_SIZE]
        return cls(intent_sequence, timestamp)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            'intent_sequence': self.intent_sequence,
            'timestamp': self.timestamp,
            'radio_call_history': self.radio_call_history
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize from dictionary."""
        belief = cls(data['intent_sequence'], data['timestamp'])
        belief.radio_call_history = data.get('radio_call_history', [])
        return belief
    
    def add_radio_call(self, radio_call: str):
        """Add the radio call that led to this belief state."""
        self.radio_call_history.append(radio_call)
    
    def __str__(self):
        return f"BeliefState({self.intent_sequence})"
    
    def __len__(self):
        return len(self.intent_sequence)


class BeliefEncoder(nn.Module):
    """
    Neural network encoder for variable-length belief state sequences.
    
    Converts belief state sequences into fixed-dimension embeddings suitable
    for integration with TrajAirNet's GAT layers.
    """
    
    def __init__(self, vocab_size: int = TOTAL_VOCAB_SIZE, embed_dim: int = 64, 
                 hidden_dim: int = 64, num_layers: int = 2):
        """
        Args:
            vocab_size: Size of intent vocabulary (including special tokens)
            embed_dim: Dimension of intent embeddings
            hidden_dim: Hidden dimension of LSTM encoder
            num_layers: Number of LSTM layers
        """
        super(BeliefEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Intent embedding layer
        self.intent_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        
        # Sequence encoder - bidirectional LSTM for better representation
        self.sequence_encoder = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=0.1
        )
        
        # Output projection to fixed dimension
        self.output_projection = nn.Linear(2 * hidden_dim, embed_dim)  # 2x for bidirectional
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.xavier_uniform_(self.intent_embedding.weight)
        # Set padding token embedding to zero and freeze
        with torch.no_grad():
            self.intent_embedding.weight[PAD_TOKEN].fill_(0)
    
    def forward(self, belief_sequences: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode variable-length belief sequences to fixed-dimension embeddings.
        
        Args:
            belief_sequences: [batch_size, max_seq_len] padded sequences
            sequence_lengths: [batch_size] actual sequence lengths
            
        Returns:
            belief_embeddings: [batch_size, embed_dim] fixed-dimension encodings
        """
        batch_size, max_seq_len = belief_sequences.shape
        
        # Embed intent sequences
        embedded = self.intent_embedding(belief_sequences)  # [batch, max_seq_len, embed_dim]
        
        # Pack for variable-length processing
        # Note: pack_padded_sequence requires lengths to be on CPU
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Encode with bidirectional LSTM
        lstm_out, (hidden, cell) = self.sequence_encoder(packed)
        
        # Use final hidden states from both directions
        # hidden: [2*num_layers, batch, hidden_dim] -> [batch, 2*hidden_dim]
        forward_hidden = hidden[-2]  # Final forward hidden state
        backward_hidden = hidden[-1]  # Final backward hidden state
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Project to output dimension
        belief_embeddings = self.output_projection(final_hidden)  # [batch, embed_dim]
        
        return belief_embeddings
    
    def encode_single_belief(self, belief_state: BeliefState) -> torch.Tensor:
        """
        Convenience method to encode a single belief state.
        
        Args:
            belief_state: BeliefState instance
            
        Returns:
            embedding: [1, embed_dim] tensor
        """
        if len(belief_state) == 0:
            # Return zero embedding for empty belief
            return torch.zeros(1, self.embed_dim)
        
        # Convert to indices and create tensor
        indices = belief_state.to_indices()
        belief_tensor = torch.tensor([indices], dtype=torch.long)  # [1, seq_len]
        lengths = torch.tensor([len(indices)], dtype=torch.long)   # [1]
        
        with torch.no_grad():
            embedding = self.forward(belief_tensor, lengths)
        
        return embedding


def pad_belief_sequences(belief_sequences: List[List[int]], pad_token: int = PAD_TOKEN) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length belief sequences to same length for batch processing.
    
    Args:
        belief_sequences: List of sequences as lists of indices
        pad_token: Token to use for padding
        
    Returns:
        padded_sequences: [batch_size, max_len] padded tensor
        sequence_lengths: [batch_size] actual sequence lengths
    """
    if not belief_sequences:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    
    # Get sequence lengths
    lengths = [len(seq) for seq in belief_sequences]
    max_len = max(lengths) if lengths else 0
    
    # Pad sequences
    padded = []
    for seq in belief_sequences:
        padded_seq = seq + [pad_token] * (max_len - len(seq))
        padded.append(padded_seq)
    
    return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def test_belief_encoder():
    """Test the BeliefEncoder with sample data."""
    print("Testing BeliefEncoder...")
    
    # Create sample belief states
    belief1 = BeliefState(['downwind_26', 'base_26', 'final_26', 'land_26'])
    belief2 = BeliefState(['takeoff_8', 'upwind_8', 'depart_north'])
    belief3 = BeliefState(['hold_short', 'taxi_to_runway', 'takeoff_26'])
    
    beliefs = [belief1, belief2, belief3]
    
    # Convert to indices
    sequences = [belief.to_indices() for belief in beliefs]
    print(f"Sequences: {sequences}")
    
    # Pad sequences
    padded, lengths = pad_belief_sequences(sequences)
    print(f"Padded shape: {padded.shape}")
    print(f"Lengths: {lengths}")
    
    # Test encoder
    encoder = BeliefEncoder(embed_dim=32, hidden_dim=32)
    embeddings = encoder(padded, lengths)
    
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Sample embedding: {embeddings[0][:5]}")
    
    # Test single belief encoding
    single_embedding = encoder.encode_single_belief(belief1)
    print(f"Single belief embedding shape: {single_embedding.shape}")
    
    print("BeliefEncoder test passed!")


if __name__ == "__main__":
    test_belief_encoder()