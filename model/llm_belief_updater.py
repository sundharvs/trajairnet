"""
LLM-Based Belief Update System

This module implements the GPT-4o + Gemma-3 pipeline for updating pilot
belief states based on radio communications.
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import re

# Configuration flags - set these to control behavior
CLEAN_TRANSCRIPTS = False  # Set to False to skip transcript cleaning with GPT
USE_GPT5 = True  # Set to True to use GPT-5 for belief updating instead of Gemma

# LLM imports (will need to be available in environment)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    
from .belief_states import BeliefState, INTENT_VOCABULARY, INTENT_NAMES
from .belief_manager import RadioCall


class LLMBeliefUpdater:
    """
    LLM-based belief state updater with configurable transcript cleaning
    and model selection (GPT-5 or Gemma-3).
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 gemma_model_path: Optional[str] = None,
                 clean_transcripts: bool = True,
                 use_gpt5: bool = False):
        """
        Initialize LLM clients.
        
        Args:
            openai_api_key: OpenAI API key (or use environment variable)
            gemma_model_path: Path to Gemma model file
            clean_transcripts: Whether to clean transcripts with GPT-4o/GPT-5
            use_gpt5: If True, use GPT-5 for belief updating. If False, use Gemma-3
        """
        self.gpt_client = None
        self.gemma_client = None
        
        # Initialize OpenAI client (needed for transcript cleaning and/or GPT-5 belief updating)
        if HAS_OPENAI and (CLEAN_TRANSCRIPTS or USE_GPT5):
            try:
                self.gpt_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
        
        # Initialize Gemma client for belief reasoning (only if not using GPT-5)
        if not USE_GPT5:
            try:
                self.gemma_client = OpenAI(base_url="http://localhost:8080/v1", api_key='-')
            except Exception as e:
                print(f"Warning: Could not initialize Gemma client: {e}")
    
    def clean_transcript_gpt4o(self, raw_transcript: str) -> str:
        """
        Use GPT-4o/GPT-5 to clean and standardize radio call transcript.
        
        Args:
            raw_transcript: Raw ASR output
            
        Returns:
            Cleaned and standardized transcript
        """
        if not CLEAN_TRANSCRIPTS or not self.gpt_client:
            return raw_transcript  # Return as-is if cleaning disabled or no client
        
        system_prompt = """You are an expert in aviation radio communications. Your task is to clean and standardize pilot radio calls from automatic speech recognition (ASR) output.

Clean the transcript by:
1. Correcting obvious ASR errors for aviation terminology
2. Standardizing aircraft identifiers (e.g., "N135PL", "Cherokee 135PL")  
3. Standardizing airport/runway references ("runway 8", "runway 26")
4. Standardizing traffic pattern terminology ("downwind", "base", "final")
5. Preserving the original meaning and intent

Return only the cleaned transcript, nothing else."""

        user_prompt = f"Clean this radio call transcript: {raw_transcript}"
        
        try:
            model = "gpt-5" if USE_GPT5 else "gpt-4o-mini"  # Use GPT-5 or mini for cost efficiency
            response = self.gpt_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            cleaned_transcript = response.choices[0].message.content.strip()
            return cleaned_transcript
            
        except Exception as e:
            print(f"Warning: GPT-4o transcript cleaning failed: {e}")
            return raw_transcript
    
    def update_belief_gemma(self, radio_call: RadioCall, previous_belief: Optional[BeliefState]) -> BeliefState:
        """
        Use Gemma-3 to update belief state based on radio call and previous belief.
        
        Args:
            radio_call: Radio communication (with cleaned transcript)
            previous_belief: Previous belief state
            
        Returns:
            Updated belief state
        """
        if not self.gemma_client:
            raise RuntimeError("Gemma client not available. Please provide gemma_model_path or install llama-cpp-python.")
        
        # Prepare prompt with intent vocabulary and previous belief
        intent_options = self._format_intent_options()
        previous_belief_str = self._format_previous_belief(previous_belief)
        
        prompt = f"""You will translate radio communications on the common traffic advisory frequency (CTAF) of Butler county airport, which is untowered. Pilots announce their intentions in natural language over the radio, structured around the traffic pattern. In the closed traffic pattern, pilots take off from a runway, turn on to a crosswind leg, then turn onto a downwind leg, then turn onto a base leg, then turn onto a final leg, and then land on the runway. There are two traffic patterns at this airport, one for runway 8 and one for runway 26.


PATTERN OPERATIONS:
- Standard patterns are LEFT traffic (left turns)
- Common sequences: takeoff → upwind → crosswind → downwind → base → final → land
- Touch-and-go: land → immediate takeoff without stopping

Your task is to turn each radio call into a sequence of intents.

Available Intent Options:
{intent_options}

Previous Belief State: {previous_belief_str}

New Radio Call: "{radio_call.transcript}"

Examples:
"downwind for touch and go" = downwind_X, base_X, final_X, land_X, takeoff_X
"going around" = Replace landing with takeoff (e.g., final_26, land_26 → final_26, takeoff_26)
"touch and go" = Sequence includes both land and immediate takeoff (final_X, land_X, takeoff_X)
"departing runway 8 for left closed traffic" = takeoff_8, crosswind_8, downwind_8, base_8, final_8, land_8
"departing the pattern to the north" = depart_north
"clear of the active" = final_26, land_26 → clear_of_runway_26 or final_8, land_8 → clear_of_runway_8

Output ONLY comma-separated intent names from the options above
If unclear, insufficient information, or other intent (taxiing, crossing runways etc.), use "other"

Intent sequence:"""

        try:
            response = self.gemma_client.chat.completions.create(
                model = 'gemma',
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response into intent sequence
            intent_sequence = self._parse_gemma_response(response_text)
            
            belief = BeliefState(intent_sequence, radio_call.timestamp)
            belief.add_radio_call(radio_call.transcript)
            
            return belief
            
        except Exception as e:
            raise RuntimeError(f"Gemma belief update failed: {e}")
    
    def _format_intent_options(self) -> str:
        """Format intent vocabulary for prompt."""
        options_by_category = {
            "Traffic Pattern": ["upwind_8", "upwind_26", "crosswind_8", "crosswind_26", 
                              "downwind_8", "downwind_26", "base_8", "base_26", "final_8", "final_26"],
            "Runway Operations": ["takeoff_8", "takeoff_26", "land_8", "land_26"],
            "Departure Directions": ["depart_north", "depart_south", "depart_east", "depart_west"],
            "Meta": ["unknown"]
        }
        
        formatted = ""
        for category, intents in options_by_category.items():
            formatted += f"{category}: {', '.join(intents)}\n"
        
        return formatted
    
    def _format_previous_belief(self, previous_belief: Optional[BeliefState]) -> str:
        """Format previous belief for prompt."""
        if not previous_belief or not previous_belief.intent_sequence:
            return "None (first radio call from this aircraft)"
        
        return ", ".join(previous_belief.intent_sequence)
    
    def _parse_gemma_response(self, response_text: str) -> List[str]:
        """Parse Gemma response into list of intent names."""
        # Clean the response
        response_text = response_text.strip()
        
        # Split by comma and clean each intent
        intents = [intent.strip() for intent in response_text.split(',')]
        
        # Validate intents against vocabulary
        valid_intents = []
        for intent in intents:
            if intent in INTENT_VOCABULARY:
                valid_intents.append(intent)
            else:
                # Try to find close matches or default to unknown
                valid_intents.append('unknown')
        
        return valid_intents if valid_intents else ['unknown']
    
    def update_belief_gpt5(self, radio_call: RadioCall, previous_belief: Optional[BeliefState]) -> BeliefState:
        """
        Use GPT-5 to update belief state based on radio call and previous belief.
        
        Args:
            radio_call: Radio communication (with cleaned transcript)
            previous_belief: Previous belief state
            
        Returns:
            Updated belief state
        """
        if not self.gpt_client:
            raise RuntimeError("OpenAI client not available. Please set OPENAI_API_KEY environment variable.")
        
        # Prepare prompt with intent vocabulary and previous belief
        intent_options = self._format_intent_options()
        previous_belief_str = self._format_previous_belief(previous_belief)
        
        prompt = f"""You will translate radio communications on the common traffic advisory frequency (CTAF) of Butler county airport, which is untowered. Pilots announce their intentions in natural language over the radio, structured around the traffic pattern. In the closed traffic pattern, pilots take off from a runway, turn on to a crosswind leg, then turn onto a downwind leg, then turn onto a base leg, then turn onto a final leg, and then land on the runway. There are two traffic patterns at this airport, one for runway 8 and one for runway 26.


PATTERN OPERATIONS:
- Standard patterns are LEFT traffic (left turns)
- Common sequences: takeoff → upwind → crosswind → downwind → base → final → land
- Touch-and-go: land → immediate takeoff without stopping

Your task is to turn each radio call into a sequence of intents.

Available Intent Options:
{intent_options}

Previous Belief State: {previous_belief_str}

New Radio Call: "{radio_call.transcript}"

Examples:
"downwind for touch and go" = downwind_X, base_X, final_X, land_X, takeoff_X
"going around" = Replace landing with takeoff (e.g., final_26, land_26 → final_26, takeoff_26)
"touch and go" = Sequence includes both land and immediate takeoff (final_X, land_X, takeoff_X)
"departing runway 8 for left closed traffic" = takeoff_8, crosswind_8, downwind_8, base_8, final_8, land_8
"departing the pattern to the north" = depart_north
"clear of the active" = final_26, land_26 → clear_of_runway_26 or final_8, land_8 → clear_of_runway_8

Output ONLY comma-separated intent names from the options above
If unclear, insufficient information, or other intent (taxiing, crossing runways etc.), use "other"

Intent sequence:"""

        try:
            response = self.gpt_client.chat.completions.create(
                model='gpt-5',
                messages=[
                    {"role": "system", "content": "You are an expert in aviation traffic pattern analysis."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response into intent sequence
            intent_sequence = self._parse_gemma_response(response_text)
            
            belief = BeliefState(intent_sequence, radio_call.timestamp)
            belief.add_radio_call(radio_call.transcript)
            
            return belief
            
        except Exception as e:
            raise RuntimeError(f"GPT-5 belief update failed: {e}")
    
    def update_belief(self, radio_call: RadioCall, previous_belief: Optional[BeliefState]) -> BeliefState:
        """
        Main entry point for belief updating.
        
        Args:
            radio_call: Radio communication
            previous_belief: Previous belief state
            
        Returns:
            Updated belief state
        """
        # Step 1: Clean transcript (if enabled)
        if CLEAN_TRANSCRIPTS:
            cleaned_transcript = self.clean_transcript_gpt4o(radio_call.transcript)
        else:
            cleaned_transcript = radio_call.transcript
        
        # Create cleaned radio call
        cleaned_call = RadioCall(
            radio_call.timestamp,
            radio_call.tail_number,
            cleaned_transcript,
            radio_call.raw_audio_path
        )
        
        # Step 2: Update belief using selected model
        if USE_GPT5:
            updated_belief = self.update_belief_gpt5(cleaned_call, previous_belief)
        else:
            updated_belief = self.update_belief_gemma(cleaned_call, previous_belief)
        
        return updated_belief


def test_llm_belief_updater():
    """Test the LLM belief updater with sample data."""
    print("Testing LLMBeliefUpdater...")
    
    # Create updater (will use fallback mode if LLMs not available)
    updater = LLMBeliefUpdater()
    
    # Test cases
    test_calls = [
        RadioCall(1000.0, "N135PL", "Butler county traffic Cherokee 135PL entering left downwind runway 8"),
        RadioCall(1030.0, "N135PL", "Cherokee 135PL turning left base runway 8"),
        RadioCall(1060.0, "N135PL", "Cherokee 135PL final runway 8 touch and go"),
        RadioCall(1090.0, "N135PL", "Cherokee 135PL going around runway 8"),
    ]
    
    previous_belief = None
    
    for call in test_calls:
        belief = updater.update_belief(call, previous_belief)
        print(f"Call: {call.transcript}")
        print(f"Previous: {previous_belief.intent_sequence if previous_belief else None}")
        print(f"Updated:  {belief.intent_sequence}")
        print()
        previous_belief = belief
    
    print("LLMBeliefUpdater test completed!")


if __name__ == "__main__":
    test_llm_belief_updater()
