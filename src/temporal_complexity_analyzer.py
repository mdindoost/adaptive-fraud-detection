"""
Empirical Temporal Analyzer for Fraud Detection
Phase 1A: Pure Measurement Framework - No Assumptions, No Arbitrary Weights

Research Objective: Build systematic measurement tools to empirically discover
temporal patterns in fraud data before building complexity metrics.

Methodology: Framework-first approach with validation checkpoints to avoid
confirmation bias and ensure reproducible, generalizable measurements.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class EmpiricalTemporalAnalyzer:
    """
    Core measurement framework for temporal fraud pattern analysis
    
    Design Principles:
    1. Measure first, theorize second
    2. No arbitrary weights or assumptions
    3. Validate measurements make sense before proceeding
    4. Enable systematic pattern discovery
    """
    
    def __init__(self, data_path: str = "data/processed_data.pkl"):
        """
        Initialize with basic data validation only
        """
        try:
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            self.X_train = data_dict['X_train']
            self.y_train = data_dict['y_train'] 
            self.X_val = data_dict['X_val']
            self.y_val = data_dict['y_val']
            
            # Combine for temporal analysis (maintaining chronological order)
            self.data = pd.concat([
                pd.DataFrame(self.X_train).assign(target=self.y_train, split='train'),
                pd.DataFrame(self.X_val).assign(target=self.y_val, split='val')
            ], ignore_index=True)
            
            # Add time index for temporal analysis
            self.data['time_index'] = range(len(self.data))
            
            self.validate_data_structure()
            print(f"✓ Data loaded: {len(self.data)} transactions, {self.data['target'].mean():.3f} fraud rate")
            
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")
    
    def validate_data_structure(self) -> Dict[str, Any]:
        """
        Basic sanity checks on data structure
        """
        validations = {
            'has_fraud_cases': self.data['target'].sum() > 0,
            'fraud_rate_reasonable': 0.001 < self.data['target'].mean() < 0.2,
            'sufficient_data': len(self.data) > 1000,
            'no_missing_targets': not self.data['target'].isnull().any(),
            'temporal_order_maintained': len(self.data) == len(self.data.drop_duplicates())
        }
        
        failed_checks = [k for k, v in validations.items() if not v]
        if failed_checks:
            raise ValueError(f"Data validation failed: {failed_checks}")
        
        print("✓ Data structure validation passed")
        return validations
    
    def measure_basic_temporal_properties(self, window_size: int = 1000) -> Dict[str, Any]:
        """
        Measure fundamental temporal properties without interpretation
        
        Returns raw measurements that will inform metric development
        """
        print(f"📊 Measuring temporal properties with window_size={window_size}...")
        
        properties = {
            'fraud_rate_evolution': self._calculate_rolling_fraud_rates(window_size),
            'feature_stability_evolution': self._calculate_feature_stability_over_time(window_size),
            'transaction_volume_patterns': self._analyze_transaction_volumes(window_size),
            'temporal_correlations': self._measure_temporal_correlations(),
            'distribution_shifts': self._measure_distribution_shifts_over_time(window_size)
        }
        
        return properties
    
    def _calculate_rolling_fraud_rates(self, window_size: int) -> Dict[str, Any]:
        """
        Track how fraud rate evolves over time
        """
        fraud_rates = []
        time_points = []
        
        for i in range(0, len(self.data) - window_size + 1, window_size // 2):
            window = self.data.iloc[i:i + window_size]
            fraud_rate = window['target'].mean()
            fraud_rates.append(fraud_rate)
            time_points.append(i + window_size // 2)
        
        fraud_rates = np.array(fraud_rates)
        
        return {
            'rates': fraud_rates,
            'time_points': time_points,
            'mean_rate': fraud_rates.mean(),
            'rate_variance': fraud_rates.var(),
            'rate_trend': np.corrcoef(time_points, fraud_rates)[0, 1] if len(fraud_rates) > 1 else 0,
            'rate_stability': 1.0 / (1.0 + fraud_rates.std())  # Higher = more stable
        }
    
    def _calculate_feature_stability_over_time(self, window_size: int) -> Dict[str, Any]:
        """
        Measure how feature distributions change over time
        """
        feature_cols = [col for col in self.data.columns if col not in ['target', 'split', 'time_index']]
        
        stability_metrics = {}
        
        for feature in feature_cols:
            distribution_distances = []
            time_points = []
            
            windows = []
            for i in range(0, len(self.data) - window_size + 1, window_size // 2):
                window = self.data.iloc[i:i + window_size]
                windows.append(window[feature].values)
                time_points.append(i + window_size // 2)
            
            # Calculate distribution distances between consecutive windows
            for j in range(1, len(windows)):
                distance = wasserstein_distance(windows[j-1], windows[j])
                distribution_distances.append(distance)
            
            distribution_distances = np.array(distribution_distances)
            
            stability_metrics[feature] = {
                'distribution_distances': distribution_distances,
                'mean_distance': distribution_distances.mean(),
                'distance_variance': distribution_distances.var(),
                'stability_score': 1.0 / (1.0 + distribution_distances.mean())
            }
        
        # Overall feature stability
        overall_stability = np.mean([metrics['stability_score'] for metrics in stability_metrics.values()])
        
        return {
            'by_feature': stability_metrics,
            'overall_stability': overall_stability,
            'time_points': time_points[1:]  # One less due to consecutive comparison
        }
    
    def _analyze_transaction_volumes(self, window_size: int) -> Dict[str, Any]:
        """
        Analyze transaction volume patterns over time
        """
        volumes = []
        fraud_volumes = []
        time_points = []
        
        for i in range(0, len(self.data) - window_size + 1, window_size // 2):
            window = self.data.iloc[i:i + window_size]
            volumes.append(len(window))
            fraud_volumes.append(window['target'].sum())
            time_points.append(i + window_size // 2)
        
        volumes = np.array(volumes)
        fraud_volumes = np.array(fraud_volumes)
        
        return {
            'total_volumes': volumes,
            'fraud_volumes': fraud_volumes,
            'volume_stability': 1.0 / (1.0 + volumes.std()),
            'fraud_volume_stability': 1.0 / (1.0 + fraud_volumes.std()),
            'time_points': time_points
        }
    
    def _measure_temporal_correlations(self) -> Dict[str, Any]:
        """
        Measure temporal dependencies in fraud patterns
        """
        feature_cols = [col for col in self.data.columns if col not in ['target', 'split', 'time_index']]
        
        autocorrelations = {}
        
        # Measure autocorrelation for fraud indicator
        fraud_autocorr = []
        for lag in [1, 5, 10, 25, 50, 100]:
            if lag < len(self.data):
                corr = np.corrcoef(self.data['target'][:-lag], self.data['target'][lag:])[0, 1]
                fraud_autocorr.append(abs(corr))
            else:
                fraud_autocorr.append(0)
        
        autocorrelations['fraud_target'] = {
            'correlations': fraud_autocorr,
            'max_correlation': max(fraud_autocorr),
            'mean_correlation': np.mean(fraud_autocorr)
        }
        
        # Measure autocorrelation for features
        for feature in feature_cols:
            feature_autocorr = []
            for lag in [1, 5, 10, 25, 50, 100]:
                if lag < len(self.data):
                    corr = np.corrcoef(self.data[feature][:-lag], self.data[feature][lag:])[0, 1]
                    feature_autocorr.append(abs(corr))
                else:
                    feature_autocorr.append(0)
            
            autocorrelations[feature] = {
                'correlations': feature_autocorr,
                'max_correlation': max(feature_autocorr),
                'mean_correlation': np.mean(feature_autocorr)
            }
        
        # Overall temporal dependency strength
        overall_temporal_strength = np.mean([
            metrics['max_correlation'] for metrics in autocorrelations.values()
        ])
        
        return {
            'by_variable': autocorrelations,
            'overall_temporal_strength': overall_temporal_strength
        }
    
    def _measure_distribution_shifts_over_time(self, window_size: int) -> Dict[str, Any]:
        """
        Quantify how data distributions change over time using statistical tests
        """
        feature_cols = [col for col in self.data.columns if col not in ['target', 'split', 'time_index']]
        
        shift_results = {}
        
        # Create temporal windows
        windows = []
        for i in range(0, len(self.data) - window_size + 1, window_size):
            window = self.data.iloc[i:i + window_size]
            windows.append(window)
        
        if len(windows) < 2:
            return {'insufficient_data': True}
        
        # Compare each consecutive pair of windows
        for feature in feature_cols:
            ks_statistics = []
            p_values = []
            
            for i in range(len(windows) - 1):
                window1 = windows[i][feature].dropna()
                window2 = windows[i + 1][feature].dropna()
                
                if len(window1) > 10 and len(window2) > 10:
                    ks_stat, p_val = ks_2samp(window1, window2)
                    ks_statistics.append(ks_stat)
                    p_values.append(p_val)
            
            if ks_statistics:
                shift_results[feature] = {
                    'ks_statistics': ks_statistics,
                    'p_values': p_values,
                    'mean_ks_stat': np.mean(ks_statistics),
                    'significant_shifts': sum(p < 0.05 for p in p_values),
                    'shift_rate': sum(p < 0.05 for p in p_values) / len(p_values)
                }
        
        return shift_results
    
    def validate_measurements_sanity(self, measurements: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check that measurements pass basic sanity tests before proceeding
        
        Critical validation step to catch measurement errors early
        """
        print("🔍 Validating measurement sanity...")
        
        sanity_checks = {}
        
        # Check fraud rate measurements
        fraud_rates = measurements['fraud_rate_evolution']['rates']
        sanity_checks['fraud_rates_positive'] = all(r >= 0 for r in fraud_rates)
        sanity_checks['fraud_rates_reasonable'] = all(r <= 1.0 for r in fraud_rates)
        sanity_checks['fraud_rates_not_all_zero'] = any(r > 0 for r in fraud_rates)
        
        # Check stability measurements
        overall_stability = measurements['feature_stability_evolution']['overall_stability']
        sanity_checks['stability_in_range'] = 0 <= overall_stability <= 1
        
        # Check temporal correlations
        temporal_strength = measurements['temporal_correlations']['overall_temporal_strength']
        sanity_checks['temporal_strength_reasonable'] = 0 <= temporal_strength <= 1
        
        # Check for sufficient variation
        fraud_variance = measurements['fraud_rate_evolution']['rate_variance']
        sanity_checks['sufficient_variation'] = fraud_variance > 0
        
        # Overall validation
        all_passed = all(sanity_checks.values())
        sanity_checks['all_sanity_checks_passed'] = all_passed
        
        if all_passed:
            print("✓ All sanity checks passed - measurements are reasonable")
        else:
            failed_checks = [k for k, v in sanity_checks.items() if not v]
            print(f"⚠️  Sanity check failures: {failed_checks}")
        
        return sanity_checks
    
    def test_multiple_window_sizes(self, window_sizes: List[int] = [100, 500, 1000, 2000]) -> Dict[int, Any]:
        """
        Test framework with multiple window sizes to find optimal parameters
        
        This helps determine data-driven window size rather than arbitrary choice
        """
        print("🔄 Testing multiple window sizes for optimal parameters...")
        
        results = {}
        
        for window_size in window_sizes:
            if window_size > len(self.data) // 4:  # Skip if window too large
                continue
                
            print(f"   Testing window_size={window_size}")
            
            try:
                measurements = self.measure_basic_temporal_properties(window_size)
                sanity_check = self.validate_measurements_sanity(measurements)
                
                # Calculate efficiency metrics
                analysis_complexity = self._estimate_analysis_complexity(window_size)
                pattern_resolution = self._estimate_pattern_resolution(measurements)
                
                results[window_size] = {
                    'measurements': measurements,
                    'sanity_passed': sanity_check['all_sanity_checks_passed'],
                    'analysis_complexity': analysis_complexity,
                    'pattern_resolution': pattern_resolution,
                    'efficiency_score': pattern_resolution / analysis_complexity if analysis_complexity > 0 else 0
                }
                
            except Exception as e:
                print(f"   ⚠️ Window size {window_size} failed: {e}")
                results[window_size] = {'error': str(e)}
        
        # Find optimal window size
        valid_results = {k: v for k, v in results.items() if 'error' not in v and v['sanity_passed']}
        
        if valid_results:
            optimal_window = max(valid_results.keys(), key=lambda k: valid_results[k]['efficiency_score'])
            print(f"✓ Optimal window size: {optimal_window}")
            results['optimal_window_size'] = optimal_window
        else:
            print("⚠️ No valid window sizes found")
            results['optimal_window_size'] = None
        
        return results
    
    def _estimate_analysis_complexity(self, window_size: int) -> float:
        """
        Estimate computational complexity of analysis with this window size
        """
        num_windows = len(self.data) // (window_size // 2)
        return num_windows * np.log(window_size)  # O(n log k) complexity estimate
    
    def _estimate_pattern_resolution(self, measurements: Dict[str, Any]) -> float:
        """
        Estimate how well this window size captures temporal patterns
        """
        # Higher variance in measurements suggests better pattern resolution
        fraud_rate_variance = measurements['fraud_rate_evolution']['rate_variance']
        stability_variance = np.var([
            metrics['mean_distance'] 
            for metrics in measurements['feature_stability_evolution']['by_feature'].values()
        ])
        
        return fraud_rate_variance + stability_variance
    
    def generate_empirical_analysis_report(self, optimal_window_size: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive empirical analysis using optimal parameters
        
        This is the foundation for building complexity metrics in Phase 1B
        """
        if optimal_window_size is None:
            window_test_results = self.test_multiple_window_sizes()
            optimal_window_size = window_test_results.get('optimal_window_size', 1000)
        
        print(f"📋 Generating empirical analysis report with window_size={optimal_window_size}")
        
        measurements = self.measure_basic_temporal_properties(optimal_window_size)
        sanity_check = self.validate_measurements_sanity(measurements)
        
        if not sanity_check['all_sanity_checks_passed']:
            raise ValueError("Measurements failed sanity checks - cannot proceed")
        
        # Empirical pattern discovery
        empirical_patterns = self._discover_natural_patterns(measurements)
        
        report = {
            'data_summary': {
                'total_transactions': len(self.data),
                'fraud_rate': self.data['target'].mean(),
                'analysis_window_size': optimal_window_size
            },
            'temporal_measurements': measurements,
            'sanity_validation': sanity_check,
            'discovered_patterns': empirical_patterns,
            'recommendations_for_phase1b': self._generate_phase1b_recommendations(empirical_patterns)
        }
        
        return report
    
    def _discover_natural_patterns(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use clustering to discover natural temporal patterns without assumptions
        """
        fraud_rates = measurements['fraud_rate_evolution']['rates']
        stability_scores = [
            metrics['stability_score'] 
            for metrics in measurements['feature_stability_evolution']['by_feature'].values()
        ]
        
        # Create pattern features for clustering
        pattern_features = []
        time_points = measurements['fraud_rate_evolution']['time_points']
        
        for i, time_point in enumerate(time_points):
            if i < len(fraud_rates):
                features = [
                    fraud_rates[i],
                    measurements['fraud_rate_evolution']['rate_stability'],
                    measurements['feature_stability_evolution']['overall_stability'],
                    measurements['temporal_correlations']['overall_temporal_strength']
                ]
                pattern_features.append(features)
        
        if len(pattern_features) < 3:
            return {'insufficient_data_for_clustering': True}
        
        # Cluster temporal patterns
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(pattern_features)
        
        # Try different numbers of clusters
        best_clusters = 2
        best_score = -1
        
        for n_clusters in range(2, min(5, len(pattern_features))):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scaled_features)
                
                # Simple silhouette-like score
                score = self._calculate_clustering_score(scaled_features, labels)
                if score > best_score:
                    best_score = score
                    best_clusters = n_clusters
            except:
                continue
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Characterize clusters
        clusters = {}
        for cluster_id in range(best_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = np.array(pattern_features)[cluster_mask]
            
            clusters[f'cluster_{cluster_id}'] = {
                'size': int(cluster_mask.sum()),
                'fraud_rate_mean': float(cluster_data[:, 0].mean()),
                'fraud_rate_std': float(cluster_data[:, 0].std()),
                'stability_mean': float(cluster_data[:, 1].mean()),
                'temporal_strength_mean': float(cluster_data[:, 3].mean()),
                'characteristics': self._characterize_cluster(cluster_data)
            }
        
        return {
            'num_clusters': best_clusters,
            'clusters': clusters,
            'clustering_quality': best_score
        }
    
    def _calculate_clustering_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Simple clustering quality score"""
        if len(np.unique(labels)) < 2:
            return 0
        
        intra_distances = []
        inter_distances = []
        
        for label in np.unique(labels):
            cluster_points = features[labels == label]
            if len(cluster_points) > 1:
                # Intra-cluster distance
                intra_dist = np.mean([
                    np.linalg.norm(cluster_points[i] - cluster_points[j])
                    for i in range(len(cluster_points))
                    for j in range(i+1, len(cluster_points))
                ])
                intra_distances.append(intra_dist)
        
        # Inter-cluster distance
        centroids = np.array([features[labels == label].mean(axis=0) for label in np.unique(labels)])
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                inter_distances.append(np.linalg.norm(centroids[i] - centroids[j]))
        
        if not intra_distances or not inter_distances:
            return 0
        
        return np.mean(inter_distances) / (np.mean(intra_distances) + 1e-6)
    
    def _characterize_cluster(self, cluster_data: np.ndarray) -> str:
        """Provide interpretable characterization of cluster"""
        fraud_rate = cluster_data[:, 0].mean()
        stability = cluster_data[:, 1].mean()
        temporal_strength = cluster_data[:, 3].mean()
        
        if fraud_rate < 0.02 and stability > 0.7:
            return "stable_low_fraud"
        elif fraud_rate > 0.05 and stability < 0.5:
            return "unstable_high_fraud"
        elif temporal_strength > 0.3:
            return "strong_temporal_dependencies"
        elif stability > 0.8:
            return "highly_stable"
        else:
            return "mixed_patterns"
    
    def _generate_phase1b_recommendations(self, patterns: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate recommendations for Phase 1B metric development based on empirical findings
        """
        if 'insufficient_data_for_clustering' in patterns:
            return {'error': 'Insufficient data for pattern analysis'}
        
        recommendations = []
        
        # Check if distinct temporal patterns exist
        if patterns['num_clusters'] >= 3:
            recommendations.append("Multiple distinct temporal patterns found - complexity metrics justified")
        else:
            recommendations.append("Limited temporal pattern diversity - focus on stability metrics")
        
        # Check temporal dependency strength
        cluster_data = patterns['clusters']
        high_temporal_clusters = sum(1 for c in cluster_data.values() if c['temporal_strength_mean'] > 0.2)
        
        if high_temporal_clusters > 0:
            recommendations.append("Strong temporal dependencies detected - sliding window analysis needed")
        else:
            recommendations.append("Weak temporal dependencies - statistical methods may suffice")
        
        # Check stability patterns  
        stable_clusters = sum(1 for c in cluster_data.values() if c['stability_mean'] > 0.7)
        
        if stable_clusters > 0:
            recommendations.append("Stable periods identified - adaptive switching opportunities exist")
        else:
            recommendations.append("Low stability throughout - complex models may be consistently needed")
        
        return {
            'phase1b_focus': recommendations,
            'suggested_metrics': [
                'TSI based on stability patterns found',
                'SWUS based on temporal dependency strength',
                'ACT based on cluster characteristics'
            ]
        }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Starting Empirical Temporal Analysis - Phase 1A")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = EmpiricalTemporalAnalyzer()
        
        # Test multiple window sizes to find optimal parameters
        window_results = analyzer.test_multiple_window_sizes([500, 1000, 2000])
        
        # Generate comprehensive empirical analysis
        analysis_report = analyzer.generate_empirical_analysis_report()
        
        print("\n📊 EMPIRICAL ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Data: {analysis_report['data_summary']['total_transactions']} transactions")
        print(f"Fraud Rate: {analysis_report['data_summary']['fraud_rate']:.3f}")
        print(f"Optimal Window: {analysis_report['data_summary']['analysis_window_size']}")
        print(f"Patterns Found: {analysis_report['discovered_patterns'].get('num_clusters', 'N/A')}")
        
        print("\n🎯 PHASE 1B RECOMMENDATIONS:")
        for rec in analysis_report['recommendations_for_phase1b']['phase1b_focus']:
            print(f"   • {rec}")
        
        # Save results for Phase 1B
        with open('results/phase1a_empirical_analysis.pkl', 'wb') as f:
            import pickle
            pickle.dump(analysis_report, f)
        
        print("\n✅ Phase 1A Foundation Complete - Ready for Phase 1B Metric Development")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
