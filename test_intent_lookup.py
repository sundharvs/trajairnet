#!/usr/bin/env python3
"""
Test script for IntentLookup class to verify correct intent retrieval
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import tempfile
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.utils import IntentLookup, TrajectoryDataset

def create_test_intent_data():
    """Create sample intent data for testing"""
    test_data = [
        # Aircraft N123AB - multiple radio calls
        {"filename": "call1.wav", "start_time": "2022-06-10 10:00:00.000000", "speaker_tail": "N123AB", "Goal Category": 1},
        {"filename": "call2.wav", "start_time": "2022-06-10 10:05:00.000000", "speaker_tail": "N123AB", "Goal Category": 3},
        {"filename": "call3.wav", "start_time": "2022-06-10 10:10:00.000000", "speaker_tail": "N123AB", "Goal Category": 12},
        
        # Aircraft N456CD - single radio call
        {"filename": "call4.wav", "start_time": "2022-06-10 10:07:30.000000", "speaker_tail": "N456CD", "Goal Category": 5},
        
        # Aircraft N789EF - radio calls with different formats
        {"filename": "call5.wav", "start_time": "2022-06-10 10:02:15.123456", "speaker_tail": "N789EF", "Goal Category": 8},
        {"filename": "call6.wav", "start_time": "2022-06-10 10:08:45.654321", "speaker_tail": "N789EF", "Goal Category": 14},
        
        # Unknown/missing tail
        {"filename": "call7.wav", "start_time": "2022-06-10 10:03:00.000000", "speaker_tail": "Unknown", "Goal Category": 2},
        {"filename": "call8.wav", "start_time": "2022-06-10 10:04:00.000000", "speaker_tail": None, "Goal Category": 4},
    ]
    
    return pd.DataFrame(test_data)

def test_intent_lookup():
    """Test IntentLookup functionality"""
    print("Testing IntentLookup class...")
    
    # Create temporary CSV file with test data
    test_df = create_test_intent_data()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_df.to_csv(f.name, index=False)
        temp_csv_path = f.name
    
    try:
        # Initialize IntentLookup
        intent_lookup = IntentLookup(temp_csv_path)
        
        # Test 1: Verify data loading
        print("\nTest 1: Data Loading")
        print(f"Number of aircraft loaded: {len(intent_lookup.intent_data)}")
        expected_aircraft = {"N123AB", "N456CD", "N789EF"}
        loaded_aircraft = set(intent_lookup.intent_data.keys())
        print(f"Expected aircraft: {expected_aircraft}")
        print(f"Loaded aircraft: {loaded_aircraft}")
        assert loaded_aircraft == expected_aircraft, f"Expected {expected_aircraft}, got {loaded_aircraft}"
        print("✓ Data loading test passed")
        
        # Test 2: Verify timestamps are properly converted
        print("\nTest 2: Timestamp Conversion")
        n123ab_calls = intent_lookup.intent_data["N123AB"]
        print(f"N123AB calls: {len(n123ab_calls)}")
        
        # Check that timestamps are in ascending order and reasonable
        timestamps = [call[0] for call in n123ab_calls]
        print(f"Timestamps (Unix): {timestamps}")
        
        # Convert back to datetime for verification
        dt_timestamps = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]
        print(f"Timestamps (UTC): {dt_timestamps}")
        
        assert timestamps == sorted(timestamps), "Timestamps should be sorted"
        assert all(1654000000 < ts < 1656000000 for ts in timestamps), "Timestamps should be reasonable (June 2022)"
        print("✓ Timestamp conversion test passed")
        
        # Test 3: Most recent intent retrieval - exact timestamp match
        print("\nTest 3: Exact Timestamp Match")
        # Use exact timestamp from the data (2022-06-10 10:05:00 = intent 3)
        # Note: timestamps are processed as UTC by IntentLookup
        exact_timestamp = timestamps[1]  # Second call timestamp (intent 3)
        intent = intent_lookup.get_most_recent_intent("N123AB", exact_timestamp)
        print(f"Intent at exact timestamp {exact_timestamp}: {intent}")
        print(f"Expected intent from data: {n123ab_calls[1][1]}")
        assert intent == 3, f"Expected intent 3, got {intent}"
        print("✓ Exact timestamp match test passed")
        
        # Test 4: Most recent intent retrieval - between timestamps
        print("\nTest 4: Between Timestamps")
        # Use timestamp between second and third call (should get intent from second call)
        between_timestamp = (timestamps[1] + timestamps[2]) / 2
        intent = intent_lookup.get_most_recent_intent("N123AB", between_timestamp)
        print(f"Intent between timestamps {between_timestamp}: {intent}")
        expected_intent = n123ab_calls[1][1]  # Should get second call's intent
        assert intent == expected_intent, f"Expected intent {expected_intent}, got {intent}"
        print("✓ Between timestamps test passed")
        
        # Test 5: Most recent intent retrieval - after all calls
        print("\nTest 5: After All Calls")
        # Use timestamp after all calls (should get last intent)
        after_timestamp = timestamps[-1] + 300  # 5 minutes after last call
        intent = intent_lookup.get_most_recent_intent("N123AB", after_timestamp)
        print(f"Intent after all calls {after_timestamp}: {intent}")
        expected_intent = n123ab_calls[-1][1]  # Should get last call's intent
        assert intent == expected_intent, f"Expected intent {expected_intent}, got {intent}"
        print("✓ After all calls test passed")
        
        # Test 6: Before any radio calls
        print("\nTest 6: Before Any Radio Calls")
        # Use timestamp before any calls (should get default 15)
        before_timestamp = timestamps[0] - 300  # 5 minutes before first call
        intent = intent_lookup.get_most_recent_intent("N123AB", before_timestamp)
        print(f"Intent before any calls {before_timestamp}: {intent}")
        assert intent == 15, f"Expected intent 15 (default), got {intent}"
        print("✓ Before radio calls test passed")
        
        # Test 7: Unknown aircraft
        print("\nTest 7: Unknown Aircraft")
        unknown_timestamp = datetime(2022, 6, 10, 10, 5, 0, tzinfo=timezone.utc).timestamp()
        intent = intent_lookup.get_most_recent_intent("N999ZZ", unknown_timestamp)
        print(f"Intent for unknown aircraft: {intent}")
        assert intent == 15, f"Expected intent 15 (default), got {intent}"
        print("✓ Unknown aircraft test passed")
        
        # Test 8: Numeric tail conversion
        print("\nTest 8: Numeric Tail Conversion")
        # Test base-36 conversion (simulate what happens in trajectory data)
        test_tail = "N123AB"
        # Convert to base-36 numeric representation and back
        numeric_tail = int(test_tail, 36)  # This is how it might be stored
        converted_tail = intent_lookup._convert_tail_to_string(numeric_tail)
        print(f"Original: {test_tail}, Numeric: {numeric_tail}, Converted: {converted_tail}")
        
        # Test with actual intent lookup using numeric tail
        test_timestamp = datetime(2022, 6, 10, 10, 7, 30, tzinfo=timezone.utc).timestamp()
        intent_numeric = intent_lookup.get_most_recent_intent(numeric_tail, test_timestamp)
        intent_string = intent_lookup.get_most_recent_intent(test_tail, test_timestamp)
        print(f"Intent with numeric tail: {intent_numeric}, with string tail: {intent_string}")
        assert intent_numeric == intent_string, f"Numeric and string tail should give same result"
        print("✓ Numeric tail conversion test passed")
        
        # Test 9: Time delta simulation (like in test.py)
        print("\nTest 9: Time Delta Simulation")
        # Simulate what happens in test.py
        radio_call_time = timestamps[1]  # Second radio call
        pred_start_time = radio_call_time + 90  # 1.5 minutes later
        
        # Get intent
        intent = intent_lookup.get_most_recent_intent("N123AB", pred_start_time)
        
        # Calculate time delta (should be reasonable now)
        time_delta = pred_start_time - radio_call_time
        
        print(f"Radio call time: {datetime.fromtimestamp(radio_call_time, tz=timezone.utc)}")
        print(f"Prediction start: {datetime.fromtimestamp(pred_start_time, tz=timezone.utc)}")
        print(f"Time delta: {time_delta} seconds ({time_delta/60:.1f} minutes)")
        print(f"Retrieved intent: {intent}")
        
        assert 0 < time_delta < 3600, f"Time delta should be reasonable (0-3600 seconds), got {time_delta}"
        expected_intent = n123ab_calls[1][1]
        assert intent == expected_intent, f"Should retrieve intent {expected_intent}"
        print("✓ Time delta simulation test passed")
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("IntentLookup is working correctly")
        print("="*50)
        
    finally:
        # Clean up temporary file
        os.unlink(temp_csv_path)

def test_real_data():
    """Test with actual project data if available"""
    print("\nTesting with real project data...")
    
    real_csv_path = "/home/ssangeetha3/git/ctaf-intent-inference/main_pipeline/2_categorize_radio_calls/transcripts_with_goals.csv"
    
    if not os.path.exists(real_csv_path):
        print(f"Real data file not found at {real_csv_path}")
        return
    
    try:
        # Load real data
        intent_lookup = IntentLookup(real_csv_path)
        print(f"Loaded real data for {len(intent_lookup.intent_data)} aircraft")
        
        # Show some sample data
        if len(intent_lookup.intent_data) > 0:
            sample_aircraft = list(intent_lookup.intent_data.keys())[0]
            sample_calls = intent_lookup.intent_data[sample_aircraft][:3]  # First 3 calls
            
            print(f"\nSample aircraft: {sample_aircraft}")
            print("Sample radio calls:")
            for timestamp, intent in sample_calls:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                print(f"  {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC -> Intent {intent}")
            
            # Test retrieval
            if len(sample_calls) > 0:
                test_timestamp = sample_calls[0][0] + 30  # 30 seconds after first call
                retrieved_intent = intent_lookup.get_most_recent_intent(sample_aircraft, test_timestamp)
                expected_intent = sample_calls[0][1]
                
                print(f"\nTest: 30 seconds after first call")
                print(f"Expected intent: {expected_intent}, Retrieved: {retrieved_intent}")
                assert retrieved_intent == expected_intent, "Should retrieve first call's intent"
                print("✓ Real data test passed")
    
    except Exception as e:
        print(f"Error testing real data: {e}")

def analyze_time_delta_distribution(dataset_path=None):
    """
    Comprehensive analysis of time deltas between radio calls and trajectory predictions.
    This function analyzes the entire dataset to understand the distribution of time deltas.
    """
    print("=" * 80)
    print("TIME DELTA DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    if dataset_path is None:
        dataset_path = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/7daysJune/processed_data/"
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset
    dataset_test = TrajectoryDataset(dataset_path + "train", obs_len=11, pred_len=120, step=10, skip=1, delim=' ')
    intent_lookup = dataset_test.intent_lookup
    
    print(f"Dataset loaded: {len(dataset_test)} sequences")
    print(f"Radio call data for {len(intent_lookup.intent_data)} aircraft")
    
    # Calculate all time deltas
    print("\nCalculating time deltas for all predictions...")
    time_deltas = []
    same_day_deltas = []
    cross_day_deltas = []
    
    for seq_idx in tqdm(range(len(dataset_test)), desc="Processing sequences"):
        batch_data = dataset_test[seq_idx]
        timestamp_data = batch_data[6]  # obs_timestamp
        tail_data = batch_data[7]       # obs_tail
        
        for agent in range(timestamp_data.shape[0]):
            agent_tail = tail_data[agent, 0, -1].item()
            pred_timestamp = timestamp_data[agent, 0, -1].item()
            
            tail_str = intent_lookup._convert_tail_to_string(agent_tail)
            
            if tail_str in intent_lookup.intent_data:
                radio_call_timestamp = None
                
                # Find most recent radio call before prediction (same logic as test.py)
                for timestamp, intent in intent_lookup.intent_data[tail_str]:
                    if timestamp <= pred_timestamp:
                        radio_call_timestamp = timestamp
                    else:
                        break
                
                if radio_call_timestamp is not None:
                    time_delta = pred_timestamp - radio_call_timestamp
                    time_deltas.append(time_delta)
                    
                    # Categorize by same-day vs cross-day
                    pred_date = datetime.fromtimestamp(pred_timestamp, tz=timezone.utc).date()
                    radio_date = datetime.fromtimestamp(radio_call_timestamp, tz=timezone.utc).date()
                    
                    if pred_date == radio_date:
                        same_day_deltas.append(time_delta)
                    else:
                        cross_day_deltas.append(time_delta)
    
    # Convert to numpy arrays
    time_deltas = np.array(time_deltas)
    same_day_deltas = np.array(same_day_deltas)
    cross_day_deltas = np.array(cross_day_deltas)
    
    print(f"\nAnalysis complete:")
    print(f"  Total predictions with radio data: {len(time_deltas):,}")
    print(f"  Same-day predictions: {len(same_day_deltas):,} ({len(same_day_deltas)/len(time_deltas)*100:.1f}%)")
    print(f"  Cross-day predictions: {len(cross_day_deltas):,} ({len(cross_day_deltas)/len(time_deltas)*100:.1f}%)")
    
    # Generate comprehensive statistics
    def calculate_stats(data, name):
        if len(data) == 0:
            return {name: {'count': 0, 'message': 'No data'}}
        
        stats = {
            'count': len(data),
            'mean_seconds': float(np.mean(data)),
            'mean_minutes': float(np.mean(data) / 60),
            'mean_hours': float(np.mean(data) / 3600),
            'median_seconds': float(np.median(data)),
            'median_minutes': float(np.median(data) / 60),
            'median_hours': float(np.median(data) / 3600),
            'std_seconds': float(np.std(data)),
            'min_seconds': float(np.min(data)),
            'max_seconds': float(np.max(data)),
            'percentiles': {}
        }
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats['percentiles'][f'p{p:02d}'] = float(np.percentile(data, p))
        
        # Distribution by time ranges
        ranges = [
            (60, 'under_1_minute'),
            (300, 'under_5_minutes'), 
            (600, 'under_10_minutes'),
            (1800, 'under_30_minutes'),
            (3600, 'under_1_hour'),
            (21600, 'under_6_hours'),
            (86400, 'under_24_hours')
        ]
        
        stats['distribution'] = {}
        for threshold, label in ranges:
            count = int(np.sum(data < threshold))
            percentage = count / len(data) * 100
            stats['distribution'][label] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        return {name: stats}
    
    # Calculate statistics for all categories
    all_stats = {}
    all_stats.update(calculate_stats(time_deltas, 'all_deltas'))
    all_stats.update(calculate_stats(same_day_deltas, 'same_day_deltas'))
    all_stats.update(calculate_stats(cross_day_deltas, 'cross_day_deltas'))
    
    # Add dataset overview
    summary = {
        'dataset_overview': {
            'total_sequences': len(dataset_test),
            'total_predictions_with_radio_data': len(time_deltas),
            'same_day_predictions': len(same_day_deltas),
            'cross_day_predictions': len(cross_day_deltas),
            'percentage_same_day': round(len(same_day_deltas) / len(time_deltas) * 100, 1),
            'percentage_cross_day': round(len(cross_day_deltas) / len(time_deltas) * 100, 1),
            'analysis_timestamp': datetime.now(tz=timezone.utc).isoformat()
        }
    }
    
    final_stats = {**summary, **all_stats}
    
    # Save statistics to JSON
    stats_file = 'time_delta_analysis_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if len(same_day_deltas) > 0:
        print(f"\nSAME-DAY PREDICTIONS (realistic use case):")
        print(f"  Count: {len(same_day_deltas):,}")
        print(f"  Mean: {np.mean(same_day_deltas):.0f} seconds ({np.mean(same_day_deltas)/60:.1f} minutes)")
        print(f"  Median: {np.median(same_day_deltas):.0f} seconds ({np.median(same_day_deltas)/60:.1f} minutes)")
        print(f"  95th percentile: {np.percentile(same_day_deltas, 95):.0f} seconds ({np.percentile(same_day_deltas, 95)/60:.1f} minutes)")
    
    if len(cross_day_deltas) > 0:
        print(f"\nCROSS-DAY PREDICTIONS (problematic):")
        print(f"  Count: {len(cross_day_deltas):,}")
        print(f"  Mean: {np.mean(cross_day_deltas):.0f} seconds ({np.mean(cross_day_deltas)/3600:.1f} hours)")
        print(f"  Median: {np.median(cross_day_deltas):.0f} seconds ({np.median(cross_day_deltas)/3600:.1f} hours)")
    
    print(f"\nOverall dataset:")
    print(f"  Mean: {np.mean(time_deltas):.0f} seconds ({np.mean(time_deltas)/3600:.1f} hours)")
    print(f"  Median: {np.median(time_deltas):.0f} seconds ({np.median(time_deltas)/3600:.1f} hours)")
    
    # Generate plots
    print(f"\nGenerating visualization plots...")
    create_time_delta_plots(time_deltas, same_day_deltas, cross_day_deltas)
    
    print(f"\nAnalysis complete!")
    print(f"  Statistics saved to: {stats_file}")
    print(f"  Plots saved to: time_delta_analysis_plots.png")
    
    return final_stats

def create_time_delta_plots(time_deltas, same_day_deltas, cross_day_deltas):
    """Create comprehensive visualization plots for time delta analysis"""
    
    # Enable interactive mode for zooming
    plt.ion()
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Time Delta Analysis: Radio Call to Trajectory Prediction', fontsize=16)
    
    # Plot 1: All time deltas histogram (log scale)
    axes[0,0].hist(time_deltas / 3600, bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Time Delta (hours)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title(f'All Time Deltas (n={len(time_deltas):,})')
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Same-day only histogram
    if len(same_day_deltas) > 0:
        axes[0,1].hist(same_day_deltas / 60, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].set_xlabel('Time Delta (minutes)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title(f'Same-Day Deltas Only (n={len(same_day_deltas):,})')
        axes[0,1].grid(True, alpha=0.3)
    else:
        axes[0,1].text(0.5, 0.5, 'No same-day data', ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Same-Day Deltas (No Data)')
    
    # Plot 3: Cross-day histogram  
    if len(cross_day_deltas) > 0:
        axes[0,2].hist(cross_day_deltas / 3600, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0,2].set_xlabel('Time Delta (hours)')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title(f'Cross-Day Deltas (n={len(cross_day_deltas):,})')
        axes[0,2].grid(True, alpha=0.3)
    else:
        axes[0,2].text(0.5, 0.5, 'No cross-day data', ha='center', va='center', transform=axes[0,2].transAxes)
        axes[0,2].set_title('Cross-Day Deltas (No Data)')
    
    # Plot 4: Box plot comparison
    box_data = []
    box_labels = []
    if len(same_day_deltas) > 0:
        box_data.append(same_day_deltas / 60)
        box_labels.append(f'Same Day\n(minutes)\nn={len(same_day_deltas):,}')
    if len(cross_day_deltas) > 0:
        box_data.append(cross_day_deltas / 3600) 
        box_labels.append(f'Cross Day\n(hours)\nn={len(cross_day_deltas):,}')
    
    if box_data:
        axes[1,0].boxplot(box_data, labels=box_labels)
        axes[1,0].set_ylabel('Time Delta')
        axes[1,0].set_title('Distribution Comparison')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'No data for comparison', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Box Plot Comparison (No Data)')
    
    # Plot 5: Cumulative distribution (all data)
    sorted_all = np.sort(time_deltas)
    cumulative_pct = np.arange(1, len(sorted_all)+1) / len(sorted_all) * 100
    axes[1,1].plot(sorted_all / 3600, cumulative_pct)
    axes[1,1].set_xlabel('Time Delta (hours)')
    axes[1,1].set_ylabel('Cumulative Percentage')
    axes[1,1].set_title('Cumulative Distribution (All Data)')
    axes[1,1].set_xscale('log')
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Same-day cumulative (if available)
    if len(same_day_deltas) > 0:
        sorted_same = np.sort(same_day_deltas)
        cumulative_same = np.arange(1, len(sorted_same)+1) / len(sorted_same) * 100
        axes[1,2].plot(sorted_same / 60, cumulative_same, color='green', linewidth=2)
        axes[1,2].set_xlabel('Time Delta (minutes)')
        axes[1,2].set_ylabel('Cumulative Percentage')
        axes[1,2].set_title('Cumulative Distribution (Same-Day Only)')
        axes[1,2].grid(True, alpha=0.3)
        
        # Add some key percentiles as vertical lines
        for p in [50, 90, 95]:
            val = np.percentile(same_day_deltas, p) / 60
            axes[1,2].axvline(val, color='red', linestyle='--', alpha=0.7)
            axes[1,2].text(val, p, f'{p}%', rotation=90, va='bottom', ha='right')
    else:
        axes[1,2].text(0.5, 0.5, 'No same-day data\nfor cumulative plot', ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('Same-Day Cumulative (No Data)')
    
    plt.tight_layout()
    plt.savefig('time_delta_analysis_plots.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60)
    print("INTERACTIVE PLOT CONTROLS:")
    print("- Use mouse wheel to zoom in/out")
    print("- Click and drag to pan")
    print("- Use toolbar buttons for zoom, pan, home, etc.")
    print("- Right-click for context menu options")
    print("="*60)
    
    plt.show(block=True)  # Keep plot window open for interaction

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-time-deltas":
        # Run comprehensive time delta analysis
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_time_delta_distribution(dataset_path)
    else:
        # Run original intent lookup tests
        test_intent_lookup()
        test_real_data()
