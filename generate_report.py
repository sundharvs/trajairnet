#!/usr/bin/env python3
"""
Generate comprehensive comparison report for all intent-aware methods.
"""

import json
import os
from datetime import datetime

def load_results():
    """Load all evaluation results."""
    results = {}
    result_files = [
        ('baseline_results.json', 'Baseline (No Intent)'),
        ('intent_results_intent-interaction-matrix.json', 'Option 1: Intent Interaction Matrix'),  
        ('intent_results_intent-attention-head.json', 'Option 2: Intent Attention Head'),
        ('intent_results_multi-head-intent.json', 'Option 3: Multi-Head Intent')
    ]
    
    for filename, method_name in result_files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                results[method_name] = data
                print(f"✅ Loaded results for {method_name}")
        else:
            print(f"⚠️  No results found for {method_name}")
    
    return results

def create_comparison_table(results):
    """Create a formatted comparison table."""
    if not results:
        return "No results to compare!"
    
    print("\n" + "="*100)
    print("TRAJECTORY PREDICTION COMPARISON: Intent-Aware vs Baseline")
    print("="*100)
    
    # Header
    print(f"{'Method':<40} {'ADE':<15} {'FDE':<15} {'Agents':<10} {'Time(s)':<10} {'Status':<15}")
    print("-"*40 + " " + "-"*15 + " " + "-"*15 + " " + "-"*10 + " " + "-"*10 + " " + "-"*15)
    
    # Sort by ADE for ranking
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('avg_ade', float('inf')))
    
    comparison_data = []
    
    for method_name, data in sorted_results:
        ade = data.get('avg_ade', 'N/A')
        fde = data.get('avg_fde', 'N/A')
        agents = data.get('total_agents', 'N/A')
        time_s = data.get('eval_time_seconds', 'N/A')
        
        status = "RANDOM INIT"  # All using random initialization
        
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
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    valid_results = [r for r in comparison_data if r['status'] != 'FAILED' and isinstance(r['ade'], float)]
    
    if not valid_results:
        print("❌ No valid results to analyze!")
        return
    
    # Find best performing method
    best_ade = min(valid_results, key=lambda x: x['ade'])
    best_fde = min(valid_results, key=lambda x: x['fde'])
    
    print(f"🏆 Best ADE: {best_ade['method']} ({best_ade['ade']:.6f})")
    print(f"🏆 Best FDE: {best_fde['method']} ({best_fde['ade']:.6f})")
    
    # Compare intent methods to baseline
    baseline_results = [r for r in valid_results if 'baseline' in r['method'].lower()]
    intent_results = [r for r in valid_results if 'option' in r['method'].lower()]
    
    if baseline_results and intent_results:
        baseline_ade = baseline_results[0]['ade']
        print(f"\n📊 Baseline ADE: {baseline_ade:.6f}")
        print("Intent Method Performance:")
        
        for result in intent_results:
            improvement = ((baseline_ade - result['ade']) / baseline_ade) * 100
            symbol = "📈" if improvement > 0 else "📉"
            print(f"{symbol} {result['method']}: {improvement:+.2f}% (ADE: {result['ade']:.6f})")
    
    # Status summary
    print(f"\n📋 EVALUATION STATUS SUMMARY:")
    print("  All methods evaluated with random initialization (no training performed)")
    print("  Results demonstrate architectural differences between intent integration methods")

def main():
    """Main execution."""
    print("🚀 COMPREHENSIVE INTENT-AWARE TRAJECTORY PREDICTION EVALUATION REPORT")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = load_results()
    
    if results:
        comparison_data = create_comparison_table(results)
        create_summary_report(comparison_data)
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"comprehensive_evaluation_report_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'evaluation_summary': {
                'total_methods': len(results),
                'successful_evaluations': len(results),
                'failed_evaluations': 0
            },
            'results': results,
            'comparison': comparison_data,
            'note': 'All evaluations performed with random initialization for architectural comparison'
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive report saved to: {report_file}")
    else:
        print("❌ No results collected!")

if __name__ == "__main__":
    main()