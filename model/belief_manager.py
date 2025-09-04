"""
Belief Management System for Dynamic Intent Tracking

This module implements the radio call listener and belief update system that
maintains evolving pilot intent states over time.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .belief_states import BeliefState, INTENT_VOCABULARY


@dataclass 
class RadioCall:
    """Represents a single radio communication."""
    timestamp: float
    tail_number: str
    transcript: str
    raw_audio_path: Optional[str] = None


class BeliefManager:
    """
    Centralized system for tracking and updating pilot belief states.
    
    This class manages the "radio call listener module" that:
    1. Processes incoming radio calls
    2. Updates aircraft belief states using LLMs  
    3. Maintains historical belief evolution
    4. Provides belief lookups for trajectory prediction
    """
    
    def __init__(self, storage_dir: str = "belief_data"):
        """
        Args:
            storage_dir: Directory to store belief state files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # In-memory belief tracking
        self.aircraft_beliefs: Dict[str, List[BeliefState]] = {}  # tail -> belief history
        self.radio_calls: Dict[str, List[RadioCall]] = {}         # tail -> radio call history
        
        # Configuration
        self.belief_expiry_hours = 24  # How long beliefs remain valid
        self.max_belief_sequence_length = 8  # Maximum intents in a belief sequence
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing belief states and radio calls from storage."""
        beliefs_file = self.storage_dir / "aircraft_beliefs.json"
        calls_file = self.storage_dir / "radio_calls.json"
        
        if beliefs_file.exists():
            with open(beliefs_file, 'r') as f:
                data = json.load(f)
                for tail, belief_list in data.items():
                    self.aircraft_beliefs[tail] = [
                        BeliefState.from_dict(b) for b in belief_list
                    ]
        
        if calls_file.exists():
            with open(calls_file, 'r') as f:
                data = json.load(f)
                for tail, call_list in data.items():
                    self.radio_calls[tail] = [
                        RadioCall(**call) for call in call_list
                    ]
    
    def save_data(self):
        """Persist belief states and radio calls to storage."""
        # Save beliefs
        beliefs_data = {}
        for tail, belief_list in self.aircraft_beliefs.items():
            beliefs_data[tail] = [b.to_dict() for b in belief_list]
        
        with open(self.storage_dir / "aircraft_beliefs.json", 'w') as f:
            json.dump(beliefs_data, f, indent=2)
        
        # Save radio calls
        calls_data = {}
        for tail, call_list in self.radio_calls.items():
            calls_data[tail] = [
                {
                    'timestamp': call.timestamp,
                    'tail_number': call.tail_number,
                    'transcript': call.transcript,
                    'raw_audio_path': call.raw_audio_path
                }
                for call in call_list
            ]
        
        with open(self.storage_dir / "radio_calls.json", 'w') as f:
            json.dump(calls_data, f, indent=2)
    
    def process_radio_call(self, radio_call: RadioCall, llm_belief_updater=None) -> BeliefState:
        """
        Process a new radio call and update aircraft belief state.
        
        Args:
            radio_call: The radio communication to process
            llm_belief_updater: Function to update beliefs using LLMs
            
        Returns:
            Updated belief state for the aircraft
        """
        tail = radio_call.tail_number
        
        # Store the radio call
        if tail not in self.radio_calls:
            self.radio_calls[tail] = []
        self.radio_calls[tail].append(radio_call)
        
        # Get previous belief state
        previous_belief = self.get_most_recent_belief(tail, radio_call.timestamp)
        
        new_belief = llm_belief_updater(radio_call, previous_belief)
        
        # Store updated belief
        if tail not in self.aircraft_beliefs:
            self.aircraft_beliefs[tail] = []
        self.aircraft_beliefs[tail].append(new_belief)
        
        return new_belief
    
    def get_most_recent_belief(self, tail_number: str, before_timestamp: float) -> Optional[BeliefState]:
        """
        Get the most recent belief state for an aircraft before a given time.
        
        Args:
            tail_number: Aircraft identifier
            before_timestamp: Only consider beliefs before this time
            
        Returns:
            Most recent belief state or None if not found
        """
        if tail_number not in self.aircraft_beliefs:
            return None
        
        beliefs = self.aircraft_beliefs[tail_number]
        valid_beliefs = [b for b in beliefs if b.timestamp < before_timestamp]
        
        if not valid_beliefs:
            return None
        
        # Return most recent belief
        return max(valid_beliefs, key=lambda b: b.timestamp)
    
    def get_belief_at_sequence_time(self, tail_numbers: List[str], sequence_timestamp: float) -> List[Optional[BeliefState]]:
        """
        Get belief states for multiple aircraft at a specific sequence time.
        
        This is the main interface for the trajectory prediction module.
        
        Args:
            tail_numbers: List of aircraft identifiers in the sequence
            sequence_timestamp: Timestamp of the trajectory sequence
            
        Returns:
            List of belief states (or None) for each aircraft
        """
        beliefs = []
        for tail in tail_numbers:
            belief = self.get_most_recent_belief(tail, sequence_timestamp)
            beliefs.append(belief)
        return beliefs
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the belief tracking system."""
        stats = {
            'total_aircraft': len(self.aircraft_beliefs),
            'total_radio_calls': sum(len(calls) for calls in self.radio_calls.values()),
            'total_beliefs': sum(len(beliefs) for beliefs in self.aircraft_beliefs.values()),
            'aircraft_with_beliefs': len([t for t, beliefs in self.aircraft_beliefs.items() if beliefs]),
            'average_beliefs_per_aircraft': 0,
            'average_belief_length': 0
        }
        
        if stats['total_aircraft'] > 0:
            stats['average_beliefs_per_aircraft'] = stats['total_beliefs'] / stats['total_aircraft']
        
        # Calculate average belief sequence length
        all_beliefs = [b for beliefs in self.aircraft_beliefs.values() for b in beliefs]
        if all_beliefs:
            stats['average_belief_length'] = sum(len(b) for b in all_beliefs) / len(all_beliefs)
        
        return stats
    
    def cleanup_old_beliefs(self, current_timestamp: float):
        """Remove beliefs older than the expiry threshold."""
        expiry_threshold = current_timestamp - (self.belief_expiry_hours * 3600)
        
        for tail in list(self.aircraft_beliefs.keys()):
            self.aircraft_beliefs[tail] = [
                b for b in self.aircraft_beliefs[tail] 
                if b.timestamp > expiry_threshold
            ]
            
            # Remove aircraft with no remaining beliefs
            if not self.aircraft_beliefs[tail]:
                del self.aircraft_beliefs[tail]


class LLMBeliefUpdater:
    """
    Interface for LLM-based belief updating.
    
    This class will handle the GPT-4o transcript processing and Gemma-3 belief updates.
    Implementation will be added in the next step.
    """
    
    def __init__(self):
        self.gpt4_client = None  # Will be initialized with OpenAI client
        self.gemma_client = None  # Will be initialized with Gemma client
    
    def update_belief(self, radio_call: RadioCall, previous_belief: Optional[BeliefState]) -> BeliefState:
        """
        Update belief state using LLM reasoning.
        
        Args:
            radio_call: New radio communication
            previous_belief: Previous belief state
            
        Returns:
            Updated belief state
        """
        # TODO: Implement LLM-based belief updating
        # 1. Use GPT-4o to clean/standardize the radio call transcript
        # 2. Use Gemma-3 to update the belief sequence based on transcript + previous belief
        
        # Placeholder implementation
        return BeliefState(['unknown'], radio_call.timestamp)


def test_belief_manager():
    """Test the BeliefManager with sample data."""
    print("Testing BeliefManager...")
    
    # Create manager
    manager = BeliefManager("test_beliefs")
    
    # Create sample radio calls
    calls = [
        RadioCall(1000.0, "N135PL", "Butler county traffic Cherokee 135PL entering left downwind runway 8"),
        RadioCall(1030.0, "N135PL", "Cherokee 135PL turning left base runway 8"),
        RadioCall(1060.0, "N135PL", "Cherokee 135PL final runway 8"),
        RadioCall(1090.0, "N135PL", "Cherokee 135PL going around runway 8"),
    ]
    
    # Process calls
    for call in calls:
        belief = manager.process_radio_call(call)
        print(f"Call: {call.transcript[:50]}...")
        print(f"Belief: {belief}")
        print()
    
    # Test belief retrieval
    belief_at_1050 = manager.get_most_recent_belief("N135PL", 1050.0)
    print(f"Belief at timestamp 1050: {belief_at_1050}")
    
    # Test statistics
    stats = manager.get_statistics()
    print(f"Statistics: {stats}")
    
    # Test batch belief retrieval
    beliefs = manager.get_belief_at_sequence_time(["N135PL", "N456AB"], 1080.0)
    print(f"Beliefs at sequence time: {beliefs}")
    
    print("BeliefManager test passed!")


if __name__ == "__main__":
    test_belief_manager()