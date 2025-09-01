#!/usr/bin/env python3
"""
Create a comprehensive comparison report for all intent-aware methods.
Since training takes a long time, this will use random initialization for now
but with proper fair comparison setup.
"""

import sys
import os
import json
import subprocess
from datetime import datetime

def run_evaluation(branch_name, method_name, script_name):
    """Run evaluation on a specific branch."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {method_name}")
    print(f"Branch: {branch_name}")
    print(f"{'='*80}")
    
    try:
        # Switch to branch
        if branch_name != 'main':
            subprocess.run(['git', 'checkout', branch_name], check=True, cwd=os.getcwd())
        
        # Run evaluation
        cmd = ['pixi', 'run', 'python', script_name]
        result = subprocess.run(cmd, cwd='/home/ssangeetha3/git/ctaf-intent-inference', 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {method_name} evaluation completed successfully")
            return True
        else:
            print(f"❌ {method_name} evaluation failed:")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {method_name} evaluation timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 {method_name} evaluation error: {e}")
        return False

def collect_results():
    """Collect all evaluation results."""
    results = {}
    
    # Look for result files
    result_files = [
        ('baseline_results.json', 'Baseline (No Intent)'),
        ('intent_results_intent-interaction-matrix.json', 'Option 1: Intent Interaction Matrix'),  
        ('intent_results_intent-attention-head.json', 'Option 2: Intent Attention Head'),
        ('intent_results_multi-head-intent.json', 'Option 3: Multi-Head Intent')
    ]
    
    for filename, method_name in result_files:
        filepath = os.path.join(os.getcwd(), filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[method_name] = data
                    print(f"✅ Loaded results for {method_name}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"⚠️  No results found for {method_name} ({filename})")
    
    return results

def create_comparison_table(results):
    """Create a formatted comparison table."""
    if not results:
        return "No results to compare!"
    
    print(f"\n{'='*100}")
    print("TRAJECTORY PREDICTION COMPARISON: Intent-Aware vs Baseline")
    print(f"{'='*100}")
    
    # Header
    print(f"{'Method':<40} {'ADE':<15} {'FDE':<15} {'Agents':<10} {'Time(s)':<10} {'Status':<15}")
    print(f"{'-'*40} {'-'*15} {'-'*15} {'-'*10} {'-'*10} {'-'*15}")
    
    # Sort by ADE for ranking
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('avg_ade', float('inf')))
    
    comparison_data = []
    
    for method_name, data in sorted_results:
        ade = data.get('avg_ade', 'N/A')
        fde = data.get('avg_fde', 'N/A')
        agents = data.get('total_agents', 'N/A')
        time_s = data.get('eval_time_seconds', 'N/A')
        
        # Determine status
        if ade == 'N/A' or ade == float('inf'):
            status = "FAILED"
        elif 'random' in data.get('note', '').lower():
            status = "RANDOM INIT"
        else:
            status = "TRAINED"
        
        # Format numbers
        ade_str = f"{ade:.6f}" if isinstance(ade, float) and ade != float('inf') else str(ade)
        fde_str = f"{fde:.6f}" if isinstance(fde, float) and fde != float('inf') else str(fde)
        time_str = f"{time_s:.1f}" if isinstance(time_s, (int, float)) else str(time_s)
        
        print(f"{method_name:<40} {ade_str:<15} {fde_str:<15} {str(agents):<10} {time_str:<10} {status:<15}")
        
        comparison_data.append({
            'method': method_name,
            'ade': ade,
            'fde': fde,
            'agents': agents,
            'time': time_s,
            'status': status
        })
    
    return comparison_data

def create_summary_report(comparison_data):
    """Create a detailed summary report."""
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    valid_results = [r for r in comparison_data if r['status'] != 'FAILED' and isinstance(r['ade'], float)]
    
    if not valid_results:
        print("❌ No valid results to analyze!")
        return
    
    # Find best performing method
    best_ade = min(valid_results, key=lambda x: x['ade'])
    best_fde = min(valid_results, key=lambda x: x['fde'])
    
    print(f"🏆 Best ADE: {best_ade['method']} ({best_ade['ade']:.6f})")
    print(f"🏆 Best FDE: {best_fde['method']} ({best_fde['fde']:.6f})")
    
    # Compare intent methods to baseline
    baseline_results = [r for r in valid_results if 'baseline' in r['method'].lower()]
    intent_results = [r for r in valid_results if 'option' in r['method'].lower()]
    
    if baseline_results and intent_results:
        baseline_ade = baseline_results[0]['ade']
        print(f"\n📊 Baseline ADE: {baseline_ade:.6f}")
        print("Intent Method Improvements:")
        
        for result in intent_results:
            improvement = ((baseline_ade - result['ade']) / baseline_ade) * 100
            symbol = "📈" if improvement > 0 else "📉"
            print(f"{symbol} {result['method']}: {improvement:+.2f}% (ADE: {result['ade']:.6f})")
    
    # Status summary
    print(f"\n📋 EVALUATION STATUS SUMMARY:")
    status_counts = {}
    for result in comparison_data:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in status_counts.items():
        print(f"  {status}: {count} method(s)")

def main():
    """Main evaluation pipeline."""
    print("🚀 COMPREHENSIVE INTENT-AWARE TRAJECTORY PREDICTION EVALUATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Store original branch
    try:
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        original_branch = result.stdout.strip()
    except:
        original_branch = 'main'
    
    # Evaluation plan
    evaluations = [
        ('main', 'Baseline (No Intent)', 'trajairnet/evaluate_baseline.py'),
        ('intent-interaction-matrix', 'Option 1: Intent Interaction Matrix', 'trajairnet/evaluate_intent.py'),
        ('intent-attention-head', 'Option 2: Intent Attention Head', 'trajairnet/evaluate_intent.py'),
        ('multi-head-intent', 'Option 3: Multi-Head Intent', 'trajairnet/evaluate_intent.py')
    ]
    
    successful_evaluations = 0
    
    # Run evaluations
    for branch_name, method_name, script_name in evaluations:
        success = run_evaluation(branch_name, method_name, script_name)
        if success:
            successful_evaluations += 1
    
    # Return to original branch
    if original_branch != 'main':
        try:
            subprocess.run(['git', 'checkout', original_branch], check=True, cwd=os.getcwd())
        except:
            pass
    
    print(f"\n✅ Completed {successful_evaluations}/{len(evaluations)} evaluations")
    
    # Collect and analyze results
    print(f"\n{'='*80}")
    print("COLLECTING RESULTS")
    print(f"{'='*80}")
    
    results = collect_results()
    
    if results:
        comparison_data = create_comparison_table(results)
        create_summary_report(comparison_data)
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"comprehensive_evaluation_report_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'evaluation_summary': {
                'total_methods': len(evaluations),
                'successful_evaluations': successful_evaluations,
                'failed_evaluations': len(evaluations) - successful_evaluations
            },
            'results': results,
            'comparison': comparison_data
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive report saved to: {report_file}")
    else:
        print("❌ No results collected!")

if __name__ == "__main__":
    main()