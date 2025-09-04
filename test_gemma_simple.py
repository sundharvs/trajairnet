#!/usr/bin/env python3
"""
Simple test of Gemma-3 belief extraction on transcripts.
"""

import pandas as pd
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.llm_belief_updater import LLMBeliefUpdater
from model.belief_manager import RadioCall

def main():
    # Load transcripts
    df = pd.read_csv("../main_pipeline/1_transcribe_radio_calls/transcripts.csv")
    print(f"Loaded {len(df)} transcripts\n")
    
    # Create updater
    updater = LLMBeliefUpdater()
    
    # Test on random 10 transcripts
    for i in range(min(10, len(df))):
        row = df.sample(n=1).iloc[0]
        
        # Create radio call
        radio_call = RadioCall(
            timestamp=1000.0 + i,  # dummy timestamp
            tail_number=row['speaker_tail'],
            transcript=row['whisper_transcript'].strip()
        )
        
        # Get belief
        belief = updater.update_belief(radio_call, None)
        
        print(f"#{i+1} {radio_call.tail_number}: \"{radio_call.transcript}\"")
        print(f"   → {belief.intent_sequence}\n")

if __name__ == "__main__":
    main()