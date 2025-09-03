"""
Phase 1A Results Analysis
Comprehensive examination of empirical temporal analysis findings
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any

def load_and_analyze_phase1a_results():
    """
    Load and comprehensively analyze Phase 1A empirical findings
    """
    print("🔍 Loading Phase 1A Empirical Analysis Results")
    print("=" * 60)
    
    # Load results
    with open('results/phase1a_empirical_analysis.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print_data_summary(results)
    print_temporal_measurements(results)
    print_discovered_patterns(results)
    print_phase1b_recommendations(results)
    create_visualizations(results)
    
    return results

def print_data_summary(results: Dict[str, Any]):
    """Print basic data summary"""
    summary = results['data_summary']
    print("\n📊 DATA SUMMARY")
    print("-" * 30)
    print(f"Total Transactions: {summary['total_transactions']:,}")
    print(f"Fraud Rate: {summary['fraud_rate']:.3%}")
    print(f"Optimal Window Size: {summary['analysis_window_size']}")
    
def print_temporal_measurements(results: Dict[str, Any]):
    """Print detailed temporal measurements"""
    measurements = results['temporal_measurements']
    
    print("\n⏰ TEMPORAL MEASUREMENTS")
    print("-" * 30)
    
    # Fraud Rate Evolution
    fraud_evo = measurements['fraud_rate_evolution']
    print(f"\n🎯 Fraud Rate Analysis:")
    print(f"   Mean Rate: {fraud_evo['mean_rate']:.3%}")
    print(f"   Rate Variance: {fraud_evo['rate_variance']:.6f}")
    print(f"   Rate Stability: {fraud_evo['rate_stability']:.3f}")
    print(f"   Temporal Trend: {fraud_evo['rate_trend']:.3f}")
    print(f"   Windows Analyzed: {len(fraud_evo['rates'])}")
    
    # Feature Stability
    feature_stability = measurements['feature_stability_evolution']
    print(f"\n📈 Feature Stability Analysis:")
    print(f"   Overall Stability: {feature_stability['overall_stability']:.3f}")
    
    print(f"\n   Per-Feature Stability:")
    for feature, metrics in feature_stability['by_feature'].items():
        print(f"      {feature}: {metrics['stability_score']:.3f} "
              f"(mean_distance: {metrics['mean_distance']:.3f})")
    
    # Temporal Correlations
    temp_corr = measurements['temporal_correlations']
    print(f"\n🔗 Temporal Dependencies:")
    print(f"   Overall Temporal Strength: {temp_corr['overall_temporal_strength']:.3f}")
    
    print(f"\n   Fraud Target Autocorrelations:")
    fraud_autocorr = temp_corr['by_variable']['fraud_target']
    print(f"      Max Correlation: {fraud_autocorr['max_correlation']:.3f}")
    print(f"      Mean Correlation: {fraud_autocorr['mean_correlation']:.3f}")
    print(f"      Correlations by lag: {[f'{c:.3f}' for c in fraud_autocorr['correlations'][:6]]}")
    
    # Transaction Patterns
    volume_patterns = measurements['transaction_volume_patterns']
    print(f"\n📊 Volume Patterns:")
    print(f"   Volume Stability: {volume_patterns['volume_stability']:.3f}")
    print(f"   Fraud Volume Stability: {volume_patterns['fraud_volume_stability']:.3f}")
    
    # Distribution Shifts
    if 'distribution_shifts' in measurements and 'insufficient_data' not in measurements['distribution_shifts']:
        print(f"\n📉 Distribution Shifts:")
        for feature, shift_data in measurements['distribution_shifts'].items():
            shift_rate = shift_data['shift_rate']
            print(f"   {feature}: {shift_rate:.2%} significant shifts "
                  f"(mean KS: {shift_data['mean_ks_stat']:.3f})")

def print_discovered_patterns(results: Dict[str, Any]):
    """Print discovered temporal patterns"""
    patterns = results['discovered_patterns']
    
    print("\n🔍 DISCOVERED TEMPORAL PATTERNS")
    print("-" * 30)
    
    if 'insufficient_data_for_clustering' in patterns:
        print("❌ Insufficient data for pattern clustering")
        return
    
    print(f"Number of Distinct Patterns: {patterns['num_clusters']}")
    print(f"Clustering Quality Score: {patterns['clustering_quality']:.3f}")
    
    print(f"\n📋 Pattern Characteristics:")
    for cluster_name, cluster_info in patterns['clusters'].items():
        print(f"\n   {cluster_name.upper()}:")
        print(f"      Size: {cluster_info['size']} time periods")
        print(f"      Fraud Rate: {cluster_info['fraud_rate_mean']:.3%} ± {cluster_info['fraud_rate_std']:.3%}")
        print(f"      Stability: {cluster_info['stability_mean']:.3f}")
        print(f"      Temporal Strength: {cluster_info['temporal_strength_mean']:.3f}")
        print(f"      Type: {cluster_info['characteristics']}")

def print_phase1b_recommendations(results: Dict[str, Any]):
    """Print recommendations for Phase 1B development"""
    recs = results['recommendations_for_phase1b']
    
    print("\n🎯 PHASE 1B DEVELOPMENT RECOMMENDATIONS")
    print("-" * 30)
    
    if 'error' in recs:
        print(f"❌ {recs['error']}")
        return
    
    print(f"\n📍 Key Insights:")
    for insight in recs['phase1b_focus']:
        print(f"   • {insight}")
    
    print(f"\n🛠️  Suggested Metrics:")
    for metric in recs['suggested_metrics']:
        print(f"   • {metric}")

def create_visualizations(results: Dict[str, Any]):
    """Create visualizations of temporal analysis"""
    print("\n📊 Creating Visualizations...")
    
    measurements = results['temporal_measurements']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Phase 1A: Empirical Temporal Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Fraud Rate Evolution
    fraud_rates = measurements['fraud_rate_evolution']['rates']
    time_points = measurements['fraud_rate_evolution']['time_points']
    
    axes[0, 0].plot(time_points, fraud_rates, 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].axhline(y=np.mean(fraud_rates), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(fraud_rates):.3%}')
    axes[0, 0].set_title('Fraud Rate Evolution Over Time')
    axes[0, 0].set_xlabel('Time (Transaction Index)')
    axes[0, 0].set_ylabel('Fraud Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Feature Stability Scores
    feature_names = []
    stability_scores = []
    
    for feature, metrics in measurements['feature_stability_evolution']['by_feature'].items():
        feature_names.append(feature)
        stability_scores.append(metrics['stability_score'])
    
    axes[0, 1].bar(feature_names, stability_scores, color='green', alpha=0.7)
    axes[0, 1].set_title('Feature Stability Scores')
    axes[0, 1].set_xlabel('Features')
    axes[0, 1].set_ylabel('Stability Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Temporal Autocorrelations
    fraud_autocorr = measurements['temporal_correlations']['by_variable']['fraud_target']['correlations']
    lags = [1, 5, 10, 25, 50, 100]
    
    axes[1, 0].bar(range(len(fraud_autocorr)), fraud_autocorr, color='orange', alpha=0.7)
    axes[1, 0].set_title('Fraud Temporal Autocorrelations')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Absolute Correlation')
    axes[1, 0].set_xticks(range(len(lags)))
    axes[1, 0].set_xticklabels(lags)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Discovered Pattern Clusters (if available)
    if 'discovered_patterns' in results and 'clusters' in results['discovered_patterns']:
        clusters = results['discovered_patterns']['clusters']
        
        cluster_names = []
        fraud_rates_cluster = []
        stability_scores_cluster = []
        cluster_sizes = []
        
        for cluster_name, cluster_info in clusters.items():
            cluster_names.append(cluster_name.replace('cluster_', 'Pattern '))
            fraud_rates_cluster.append(cluster_info['fraud_rate_mean'])
            stability_scores_cluster.append(cluster_info['stability_mean'])
            cluster_sizes.append(cluster_info['size'])
        
        # Scatter plot: Fraud Rate vs Stability, sized by cluster size
        scatter = axes[1, 1].scatter(fraud_rates_cluster, stability_scores_cluster, 
                                   s=[size*50 for size in cluster_sizes], 
                                   alpha=0.7, c=range(len(cluster_names)), cmap='viridis')
        
        # Add cluster labels
        for i, name in enumerate(cluster_names):
            axes[1, 1].annotate(name, (fraud_rates_cluster[i], stability_scores_cluster[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 1].set_title('Discovered Temporal Pattern Clusters')
        axes[1, 1].set_xlabel('Mean Fraud Rate')
        axes[1, 1].set_ylabel('Mean Stability Score')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No clusters discovered\n(insufficient temporal variation)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Pattern Clustering Results')
    
    plt.tight_layout()
    plt.savefig('figures/phase1a_temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved to: figures/phase1a_temporal_analysis.png")

def interpret_results_for_research(results: Dict[str, Any]):
    """
    Provide research interpretation of the empirical findings
    """
    print("\n🧠 RESEARCH INTERPRETATION")
    print("=" * 60)
    
    measurements = results['temporal_measurements']
    patterns = results['discovered_patterns']
    
    # Key metrics for interpretation
    temporal_strength = measurements['temporal_correlations']['overall_temporal_strength']
    fraud_stability = measurements['fraud_rate_evolution']['rate_stability']
    feature_stability = measurements['feature_stability_evolution']['overall_stability']
    num_patterns = patterns.get('num_clusters', 0)
    
    print(f"\n🔍 KEY RESEARCH FINDINGS:")
    
    # Temporal Dependencies
    if temporal_strength > 0.3:
        print(f"   ✅ STRONG temporal dependencies ({temporal_strength:.3f}) - TGNs may provide value")
    elif temporal_strength > 0.1:
        print(f"   ⚠️  MODERATE temporal dependencies ({temporal_strength:.3f}) - mixed evidence for TGNs")
    else:
        print(f"   ❌ WEAK temporal dependencies ({temporal_strength:.3f}) - statistical methods likely sufficient")
    
    # Pattern Diversity
    if num_patterns >= 3:
        print(f"   ✅ Multiple temporal patterns ({num_patterns}) - adaptive selection justified")
    elif num_patterns == 2:
        print(f"   ⚠️  Limited temporal patterns ({num_patterns}) - simple switching may suffice")
    else:
        print(f"   ❌ No distinct patterns found - single model approach recommended")
    
    # Stability Analysis
    if fraud_stability > 0.7 and feature_stability > 0.7:
        print(f"   ✅ High stability - statistical methods should perform well")
    elif fraud_stability < 0.5 or feature_stability < 0.5:
        print(f"   ✅ Low stability - complex models may be needed")
    else:
        print(f"   ⚠️  Mixed stability - adaptive approach has merit")
    
    print(f"\n📊 RESEARCH IMPLICATIONS:")
    
    # Determine research direction
    if temporal_strength < 0.2 and num_patterns <= 2:
        print(f"   🎯 Focus: Prove statistical methods sufficient for this fraud type")
        print(f"   🎯 Contribution: Evidence-based model selection for stable fraud patterns")
    elif temporal_strength > 0.3 and num_patterns >= 3:
        print(f"   🎯 Focus: Build full adaptive framework with temporal complexity analysis")
        print(f"   🎯 Contribution: Novel temporal pattern-based model selection")
    else:
        print(f"   🎯 Focus: Identify specific scenarios where complexity helps")
        print(f"   🎯 Contribution: Selective adaptive model switching")
    
    # Window size insights
    optimal_window = results['data_summary']['analysis_window_size']
    if optimal_window <= 500:
        print(f"   📏 Small optimal window ({optimal_window}) suggests local pattern analysis sufficient")
    else:
        print(f"   📏 Large optimal window ({optimal_window}) suggests need for broader temporal context")

def generate_phase1b_development_plan(results: Dict[str, Any]):
    """
    Generate specific development plan for Phase 1B based on empirical findings
    """
    print("\n📋 PHASE 1B DEVELOPMENT PLAN")
    print("=" * 60)
    
    measurements = results['temporal_measurements']
    patterns = results['discovered_patterns']
    
    temporal_strength = measurements['temporal_correlations']['overall_temporal_strength']
    fraud_stability = measurements['fraud_rate_evolution']['rate_stability']
    num_patterns = patterns.get('num_clusters', 0)
    
    print(f"\n🎯 METRIC DEVELOPMENT PRIORITIES:")
    
    # TSI Development
    if fraud_stability < 0.7:
        print(f"   🔸 TSI (Temporal Stability Index): HIGH PRIORITY")
        print(f"     • Fraud stability = {fraud_stability:.3f} shows significant variation")
        print(f"     • Focus on fraud rate consistency measurement")
    else:
        print(f"   🔸 TSI (Temporal Stability Index): LOW PRIORITY")
        print(f"     • Fraud stability = {fraud_stability:.3f} is already high")
    
    # SWUS Development  
    if temporal_strength > 0.2:
        print(f"   🔸 SWUS (Sliding Window Utility Score): HIGH PRIORITY")
        print(f"     • Temporal strength = {temporal_strength:.3f} justifies sliding window analysis")
        print(f"     • Focus on cost-benefit optimization")
    else:
        print(f"   🔸 SWUS (Sliding Window Utility Score): LOW PRIORITY")
        print(f"     • Temporal strength = {temporal_strength:.3f} suggests limited sliding window value")
    
    # ACT Development
    if num_patterns >= 3:
        print(f"   🔸 ACT (Adaptive Complexity Threshold): HIGH PRIORITY")
        print(f"     • {num_patterns} distinct patterns found - decision boundaries needed")
        print(f"     • Focus on cluster-based threshold determination")
    else:
        print(f"   🔸 ACT (Adaptive Complexity Threshold): MEDIUM PRIORITY")
        print(f"     • {num_patterns} patterns found - simple thresholding may suffice")
    
    print(f"\n🛠️  NEXT IMPLEMENTATION STEPS:")
    
    if temporal_strength < 0.2:
        print(f"   1. Build evidence that statistical methods are sufficient")
        print(f"   2. Create framework for detecting rare temporal complexity spikes")
        print(f"   3. Focus on computational efficiency optimization")
    elif temporal_strength > 0.3:
        print(f"   1. Implement full TSI, SWUS, ACT metric suite")
        print(f"   2. Build meta-learning model selector")
        print(f"   3. Create comprehensive validation framework")
    else:
        print(f"   1. Focus on selective complexity detection")
        print(f"   2. Build lightweight switching mechanisms")
        print(f"   3. Emphasize interpretability and confidence measures")

if __name__ == "__main__":
    # Comprehensive results analysis
    results = load_and_analyze_phase1a_results()
    
    # Research interpretation
    interpret_results_for_research(results)
    
    # Development planning
    generate_phase1b_development_plan(results)
    
    print(f"\n✅ PHASE 1A ANALYSIS COMPLETE")
    print("=" * 60)
    print("Ready to proceed with Phase 1B metric development based on empirical findings!")
