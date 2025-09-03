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

# LLM imports (will need to be available in environment)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    
try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

from belief_states import BeliefState, INTENT_VOCABULARY, INTENT_NAMES
from belief_manager import RadioCall


class LLMBeliefUpdater:
    """
    LLM-based belief state updater using GPT-4o for transcript processing
    and Gemma-3 for belief reasoning.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 gemma_model_path: Optional[str] = None):
        """
        Initialize LLM clients.
        
        Args:
            openai_api_key: OpenAI API key (or use environment variable)
            gemma_model_path: Path to Gemma model file
        """
        self.gpt_client = None
        self.gemma_client = None
        
        # Initialize OpenAI client for transcript processing
        if HAS_OPENAI:
            try:
                self.gpt_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
        
        # Initialize Gemma client for belief reasoning
        if HAS_LLAMA and gemma_model_path:
            try:
                self.gemma_client = Llama.from_pretrained(
                    repo_id="google/gemma-3-27b-it-qat-q4_0-gguf",
                    filename="gemma-3-27b-it-q4_0.gguf",
                    n_gpu_layers=-1,
                    n_ctx=4096,
                    verbose=False
                )
            except Exception as e:
                print(f"Warning: Could not initialize Gemma client: {e}")
    
    def clean_transcript_gpt4o(self, raw_transcript: str) -> str:
        """
        Use GPT-4o to clean and standardize radio call transcript.
        
        Args:
            raw_transcript: Raw ASR output
            
        Returns:
            Cleaned and standardized transcript
        """
        if not self.gpt_client:
            return raw_transcript  # Return as-is if no client
        
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
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for cost efficiency
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=200
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
            return self._fallback_belief_update(radio_call, previous_belief)
        
        # Prepare prompt with intent vocabulary and previous belief
        intent_options = self._format_intent_options()
        previous_belief_str = self._format_previous_belief(previous_belief)
        
        prompt = f"""You are an expert pilot assistant tracking aircraft intent sequences in terminal airspace. 

Based on a pilot's radio call, predict their sequence of future intentions from the available options.

Available Intent Options:
{intent_options}

Previous Belief State: {previous_belief_str}

New Radio Call: "{radio_call.transcript}"

Rules:
1. Radio calls can contain multiple sequential intents (e.g., "downwind for touch and go" = downwind, base, final, land, takeoff)
2. New radio calls can update/override previous beliefs (e.g., "going around" changes landing to go_around)
3. Consider standard traffic pattern sequences (downwind→base→final→land)
4. Output ONLY a comma-separated sequence of intent names from the options above
5. If unclear, use "unknown" 

Intent sequence:"""

        try:
            response = self.gemma_client.create_chat_completion(
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                temperature=0.2,
                max_tokens=100
            )
            
            response_text = response['choices'][0]['message']['content'].strip()
            
            # Parse the response into intent sequence
            intent_sequence = self._parse_gemma_response(response_text)
            
            belief = BeliefState(intent_sequence, radio_call.timestamp)
            belief.add_radio_call(radio_call.transcript)
            
            return belief
            
        except Exception as e:
            print(f"Warning: Gemma belief update failed: {e}")
            return self._fallback_belief_update(radio_call, previous_belief)
    
    def _format_intent_options(self) -> str:
        """Format intent vocabulary for prompt."""
        options_by_category = {
            "Traffic Pattern": ["upwind_8", "upwind_26", "crosswind_8", "crosswind_26", 
                              "downwind_8", "downwind_26", "base_8", "base_26", "final_8", "final_26"],
            "Runway Operations": ["takeoff_8", "takeoff_26", "land_8", "land_26", 
                                "touch_and_go_8", "touch_and_go_26"],
            "Pattern Entry/Exit": ["pattern_entry_45", "pattern_entry_straight_in", "pattern_entry_teardrop",
                                 "depart_north", "depart_south", "depart_east", "depart_west", "go_around"],
            "Ground Operations": ["taxi_to_runway", "taxi_to_ramp", "hold_short", "clear_runway", "engine_start"],
            "Meta": ["unknown", "insufficient_info", "other"]
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
    
    def _fallback_belief_update(self, radio_call: RadioCall, previous_belief: Optional[BeliefState]) -> BeliefState:
        """
        Fallback belief update when LLMs are not available.
        Uses simple rule-based approach.
        """
        transcript = radio_call.transcript.lower()
        intent_sequence = []
        
        # Enhanced pattern matching
        patterns = {
            # Traffic pattern entries
            r'entering.*downwind.*8': ['downwind_8', 'base_8', 'final_8', 'land_8'],
            r'entering.*downwind.*26': ['downwind_26', 'base_26', 'final_26', 'land_26'],
            r'downwind.*8': ['downwind_8', 'base_8', 'final_8', 'land_8'],
            r'downwind.*26': ['downwind_26', 'base_26', 'final_26', 'land_26'],
            
            # Pattern segments
            r'base.*8': ['base_8', 'final_8', 'land_8'],
            r'base.*26': ['base_26', 'final_26', 'land_26'],
            r'final.*8': ['final_8', 'land_8'],
            r'final.*26': ['final_26', 'land_26'],
            
            # Operations
            r'takeoff.*8': ['takeoff_8'],
            r'takeoff.*26': ['takeoff_26'],
            r'touch.*go.*8': ['land_8', 'takeoff_8'],
            r'touch.*go.*26': ['land_26', 'takeoff_26'],
            r'going.*around|go.*around': ['go_around'],
            
            # Ground operations
            r'taxi.*runway': ['taxi_to_runway'],
            r'hold.*short': ['hold_short'],
            r'clear.*runway': ['clear_runway'],
        }
        
        # Apply pattern matching
        for pattern, sequence in patterns.items():
            if re.search(pattern, transcript):
                intent_sequence = sequence
                break
        
        # Default to unknown if no patterns match
        if not intent_sequence:
            intent_sequence = ['unknown']
        
        belief = BeliefState(intent_sequence, radio_call.timestamp)
        belief.add_radio_call(radio_call.transcript)
        
        return belief
    
    def update_belief(self, radio_call: RadioCall, previous_belief: Optional[BeliefState]) -> BeliefState:
        """
        Main entry point for belief updating.
        
        Args:
            radio_call: Radio communication
            previous_belief: Previous belief state
            
        Returns:
            Updated belief state
        """
        # Step 1: Clean transcript with GPT-4o
        cleaned_transcript = self.clean_transcript_gpt4o(radio_call.transcript)
        
        # Create cleaned radio call
        cleaned_call = RadioCall(
            radio_call.timestamp,
            radio_call.tail_number,
            cleaned_transcript,
            radio_call.raw_audio_path
        )
        
        # Step 2: Update belief with Gemma-3
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