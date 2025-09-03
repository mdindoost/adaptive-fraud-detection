"""
Multi-Dataset Temporal Analysis Framework
Cross-Domain Validation for Temporal Complexity in Fraud Detection

Purpose: Validate temporal complexity findings across different fraud domains
to avoid single-dataset bias in research conclusions.

Datasets to analyze:
1. IEEE-CIS (current) - Credit card transactions, 3.5% fraud rate
2. PaySim - Mobile money transactions, synthetic data
3. European Credit Card (MLG-ULB) - European transactions, 0.172% fraud rate  
4. Credit Card 2023 - Recent European data, 550k+ transactions

Research Goal: Determine if temporal complexity patterns are universal
or domain-specific across different fraud detection scenarios.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import our existing temporal analyzer
from temporal_complexity_analyzer import EmpiricalTemporalAnalyzer

class MultiDatasetTemporalAnalyzer:
    """
    Framework for analyzing temporal patterns across multiple fraud datasets
    
    Purpose: Validate whether temporal complexity findings from IEEE-CIS 
    generalize across different fraud detection domains and contexts
    """
    
    def __init__(self, datasets_dir: str = "data/multi_datasets"):
        """Initialize multi-dataset analyzer"""
        self.datasets_dir = datasets_dir
        os.makedirs(datasets_dir, exist_ok=True)
        
        self.dataset_configs = {
            'ieee_cis': {
                'name': 'IEEE-CIS Credit Card Fraud',
                'source': 'existing',  # Already have this
                'path': 'data/processed_data.pkl',
                'fraud_type': 'credit_card',
                'temporal_unit': 'transaction_sequence',
                'description': 'Credit card transactions, 3.5% fraud rate'
            },
            'paysim': {
                'name': 'PaySim Mobile Money Fraud',
                'source': 'kaggle',
                'dataset_id': 'ealaxi/paysim1',
                'fraud_type': 'mobile_money',
                'temporal_unit': 'hour_steps',
                'description': 'Mobile money simulation, 30 days (744 hours)'
            },
            'european_cc': {
                'name': 'European Credit Card Fraud',
                'source': 'kaggle',
                'dataset_id': 'mlg-ulb/creditcardfraud', 
                'fraud_type': 'credit_card',
                'temporal_unit': 'seconds_elapsed',
                'description': 'European transactions, 2 days, 0.172% fraud rate'
            },
            'cc_2023': {
                'name': 'Credit Card Fraud 2023',
                'source': 'kaggle',
                'dataset_id': 'nelgiriyewithana/credit-card-fraud-detection-dataset-2023',
                'fraud_type': 'credit_card', 
                'temporal_unit': 'transaction_sequence',
                'description': 'European 2023 data, 550k+ transactions'
            }
        }
        
        self.analysis_results = {}
        
    def setup_kaggle_datasets(self):
        """
        Instructions for setting up Kaggle datasets
        """
        print("📋 KAGGLE DATASET SETUP INSTRUCTIONS")
        print("=" * 60)
        print("To download datasets automatically, you need:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Get Kaggle API token from https://www.kaggle.com/account")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print()
        
        datasets_info = """
Available datasets for analysis:
• PaySim (ealaxi/paysim1): Mobile money fraud simulation
• European CC (mlg-ulb/creditcardfraud): European credit card fraud  
• CC 2023 (nelgiriyewithana/credit-card-fraud-detection-dataset-2023): Recent European data

Run: python -c "import kaggle; print('Kaggle API ready!')" to test setup
        """
        print(datasets_info)
        
    def download_datasets(self, datasets_to_download: List[str] = ['paysim', 'european_cc']):
        """
        Download specified datasets from Kaggle
        """
        try:
            import kaggle
        except ImportError:
            print("❌ Kaggle API not installed. Run: pip install kaggle")
            return False
        
        print(f"📥 Downloading {len(datasets_to_download)} datasets...")
        
        for dataset_key in datasets_to_download:
            if dataset_key not in self.dataset_configs:
                print(f"⚠️ Unknown dataset: {dataset_key}")
                continue
                
            config = self.dataset_configs[dataset_key]
            if config['source'] != 'kaggle':
                continue
                
            dataset_id = config['dataset_id']
            download_path = f"{self.datasets_dir}/{dataset_key}"
            
            try:
                print(f"   Downloading {config['name']}...")
                kaggle.api.dataset_download_files(
                    dataset_id, 
                    path=download_path, 
                    unzip=True
                )
                print(f"   ✅ {config['name']} downloaded")
                
            except Exception as e:
                print(f"   ❌ Failed to download {config['name']}: {e}")
                print(f"   💡 Try manually downloading from: https://www.kaggle.com/datasets/{dataset_id}")
        
        return True
    
    def standardize_dataset_format(self, dataset_key: str) -> pd.DataFrame:
        """
        Standardize different datasets into common format for temporal analysis
        
        Common format:
        - 'target': fraud label (0/1)
        - 'time_index': temporal ordering 
        - 'amount': transaction amount (if available)
        - Additional features as needed
        """
        config = self.dataset_configs[dataset_key]
        
        if dataset_key == 'ieee_cis':
            return self._standardize_ieee_cis()
        elif dataset_key == 'paysim':
            return self._standardize_paysim()
        elif dataset_key == 'european_cc':
            return self._standardize_european_cc()
        elif dataset_key == 'cc_2023':
            return self._standardize_cc_2023()
        else:
            raise ValueError(f"Unknown dataset: {dataset_key}")
    
    def _standardize_ieee_cis(self) -> pd.DataFrame:
        """Load existing IEEE-CIS data in standard format"""
        try:
            with open('data/processed_data.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            
            # Combine train and validation
            X_combined = pd.concat([
                pd.DataFrame(data_dict['X_train']),
                pd.DataFrame(data_dict['X_val'])
            ])
            y_combined = np.concatenate([data_dict['y_train'], data_dict['y_val']])
            
            df = X_combined.copy()
            df['target'] = y_combined
            df['time_index'] = range(len(df))
            df['amount'] = df.get('TransactionAmt', 1.0)  # Use TransactionAmt or default
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load IEEE-CIS data: {e}")
    
    def _standardize_paysim(self) -> pd.DataFrame:
        """Standardize PaySim mobile money data"""
        try:
            # PaySim file is typically named PS_*.csv
            paysim_path = f"{self.datasets_dir}/paysim"
            csv_files = [f for f in os.listdir(paysim_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in PaySim directory")
            
            df = pd.read_csv(os.path.join(paysim_path, csv_files[0]))
            
            # PaySim columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, 
            # nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
            
            # Encode transaction types as numerical features
            transaction_types = pd.get_dummies(df['type'], prefix='type')
            
            standardized = pd.DataFrame({
                'target': df['isFraud'],
                'time_index': df['step'],  # Hour steps (1-744)
                'amount': df['amount'],
                'old_balance_orig': df['oldbalanceOrg'],
                'new_balance_orig': df['newbalanceOrig'],
                'old_balance_dest': df['oldbalanceDest'],
                'new_balance_dest': df['newbalanceDest']
            })
            
            # Add encoded transaction type features
            for col in transaction_types.columns[:5]:  # Limit features for consistency
                standardized[col] = transaction_types[col]
            
            # Sort by time for proper temporal analysis
            standardized = standardized.sort_values('time_index').reset_index(drop=True)
            
            print(f"✅ PaySim standardized: {len(standardized)} transactions, {standardized['target'].mean():.3%} fraud rate")
            return standardized
            
        except Exception as e:
            raise ValueError(f"Failed to standardize PaySim data: {e}")
    
    def _standardize_european_cc(self) -> pd.DataFrame:
        """Standardize European Credit Card data"""
        try:
            european_path = f"{self.datasets_dir}/european_cc"
            df = pd.read_csv(os.path.join(european_path, 'creditcard.csv'))
            
            # European CC columns: Time, V1-V28 (PCA features), Amount, Class
            
            # Create feature columns - use a subset of PCA features for consistency
            feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
            
            standardized = pd.DataFrame({
                'target': df['Class'],
                'time_index': range(len(df)),  # Sequential index since Time is elapsed seconds
                'amount': df['Amount'],
                'time_elapsed': df['Time']  # Original time feature
            })
            
            # Add selected PCA features
            for col in feature_cols:
                if col in df.columns:
                    standardized[col] = df[col]
            
            print(f"✅ European CC standardized: {len(standardized)} transactions, {standardized['target'].mean():.3%} fraud rate")
            return standardized
            
        except Exception as e:
            raise ValueError(f"Failed to standardize European CC data: {e}")
    
    def _standardize_cc_2023(self) -> pd.DataFrame:
        """Standardize Credit Card 2023 data"""
        try:
            cc_2023_path = f"{self.datasets_dir}/cc_2023"
            csv_files = [f for f in os.listdir(cc_2023_path) if f.endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in CC 2023 directory")
            
            df = pd.read_csv(os.path.join(cc_2023_path, csv_files[0]))
            
            # CC 2023 typically has: V1-V28, Amount, Class (similar to European CC)
            
            feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
            
            standardized = pd.DataFrame({
                'target': df['Class'],
                'time_index': range(len(df)),
                'amount': df['Amount']
            })
            
            # Add available features
            for col in feature_cols:
                if col in df.columns:
                    standardized[col] = df[col]
            
            print(f"✅ CC 2023 standardized: {len(standardized)} transactions, {standardized['target'].mean():.3%} fraud rate")
            return standardized
            
        except Exception as e:
            raise ValueError(f"Failed to standardize CC 2023 data: {e}")
    
    def analyze_dataset_temporal_complexity(self, dataset_key: str) -> Dict[str, Any]:
        """
        Apply temporal complexity analysis to a specific dataset
        """
        print(f"\\n🔍 ANALYZING {self.dataset_configs[dataset_key]['name']}")
        print("-" * 50)
        
        try:
            # Load and standardize data
            df = self.standardize_dataset_format(dataset_key)
            
            # Create temporary analyzer for this dataset
            analyzer = DatasetSpecificAnalyzer(df, dataset_key)
            
            # Run the same temporal analysis as Phase 1A
            measurements = analyzer.measure_basic_temporal_properties()
            sanity_check = analyzer.validate_measurements_sanity(measurements)
            
            if not sanity_check['all_sanity_checks_passed']:
                print(f"⚠️ Sanity checks failed for {dataset_key}")
            
            # Discover patterns
            patterns = analyzer._discover_natural_patterns(measurements)
            
            analysis_result = {
                'dataset_key': dataset_key,
                'dataset_name': self.dataset_configs[dataset_key]['name'],
                'fraud_type': self.dataset_configs[dataset_key]['fraud_type'],
                'data_summary': {
                    'total_transactions': len(df),
                    'fraud_rate': df['target'].mean(),
                    'temporal_span': df['time_index'].nunique()
                },
                'temporal_measurements': measurements,
                'sanity_validation': sanity_check,
                'discovered_patterns': patterns,
                'key_metrics': {
                    'temporal_strength': measurements['temporal_correlations']['overall_temporal_strength'],
                    'fraud_stability': measurements['fraud_rate_evolution']['rate_stability'],
                    'feature_stability': measurements['feature_stability_evolution']['overall_stability'],
                    'num_patterns': patterns.get('num_clusters', 0)
                }
            }
            
            print(f"✅ Analysis complete for {dataset_key}")
            return analysis_result
            
        except Exception as e:
            print(f"❌ Analysis failed for {dataset_key}: {e}")
            return {'error': str(e), 'dataset_key': dataset_key}
    
    def run_cross_domain_analysis(self, datasets: List[str] = ['ieee_cis', 'paysim', 'european_cc']) -> Dict[str, Any]:
        """
        Run temporal complexity analysis across multiple datasets
        """
        print("🚀 MULTI-DATASET TEMPORAL COMPLEXITY ANALYSIS")
        print("=" * 60)
        print(f"Analyzing {len(datasets)} datasets for temporal complexity validation\\n")
        
        results = {}
        
        # Analyze each dataset
        for dataset_key in datasets:
            if dataset_key not in self.dataset_configs:
                print(f"⚠️ Unknown dataset: {dataset_key}")
                continue
                
            try:
                analysis_result = self.analyze_dataset_temporal_complexity(dataset_key)
                results[dataset_key] = analysis_result
                
            except Exception as e:
                print(f"❌ Failed to analyze {dataset_key}: {e}")
                results[dataset_key] = {'error': str(e)}
        
        # Comparative analysis
        comparative_results = self.generate_cross_dataset_comparison(results)
        
        # Save results
        full_results = {
            'individual_analyses': results,
            'comparative_analysis': comparative_results,
            'datasets_analyzed': datasets,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open('results/multi_dataset_analysis.pkl', 'wb') as f:
            pickle.dump(full_results, f)
        
        print("\\n✅ MULTI-DATASET ANALYSIS COMPLETE")
        print("Results saved to: results/multi_dataset_analysis.pkl")
        
        return full_results
    
    def generate_cross_dataset_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare temporal complexity findings across datasets
        """
        print("\\n📊 GENERATING CROSS-DATASET COMPARISON")
        print("-" * 50)
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if len(valid_results) < 2:
            return {'error': 'Need at least 2 valid datasets for comparison'}
        
        # Extract key metrics for comparison
        comparison_data = []
        for dataset_key, result in valid_results.items():
            metrics = result['key_metrics']
            comparison_data.append({
                'dataset': dataset_key,
                'fraud_type': result['fraud_type'],
                'fraud_rate': result['data_summary']['fraud_rate'],
                'temporal_strength': metrics['temporal_strength'],
                'fraud_stability': metrics['fraud_stability'],
                'feature_stability': metrics['feature_stability'],
                'num_patterns': metrics['num_patterns'],
                'total_transactions': result['data_summary']['total_transactions']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Generate insights
        insights = self._generate_comparative_insights(comparison_df)
        
        return {
            'comparison_table': comparison_df.to_dict('records'),
            'insights': insights,
            'summary_statistics': {
                'temporal_strength_range': [comparison_df['temporal_strength'].min(), 
                                          comparison_df['temporal_strength'].max()],
                'fraud_stability_range': [comparison_df['fraud_stability'].min(),
                                        comparison_df['fraud_stability'].max()],
                'fraud_rate_range': [comparison_df['fraud_rate'].min(),
                                   comparison_df['fraud_rate'].max()]
            }
        }
    
    def _generate_comparative_insights(self, comparison_df: pd.DataFrame) -> List[str]:
        """Generate research insights from cross-dataset comparison"""
        insights = []
        
        # Temporal strength analysis
        temporal_strengths = comparison_df['temporal_strength']
        if temporal_strengths.max() < 0.2:
            insights.append("🔍 CONSISTENT finding: All datasets show weak temporal dependencies (<0.2)")
            insights.append("📊 Research implication: Statistical methods likely sufficient across fraud types")
        elif temporal_strengths.min() > 0.4:
            insights.append("🔍 CONSISTENT finding: All datasets show strong temporal dependencies (>0.4)")
            insights.append("📊 Research implication: TGN models justified across fraud domains")
        else:
            insights.append("🔍 MIXED finding: Temporal dependency strength varies by fraud type")
            insights.append("📊 Research implication: Domain-aware adaptive model selection needed")
        
        # Fraud type analysis
        fraud_types = comparison_df['fraud_type'].unique()
        if len(fraud_types) > 1:
            cc_temporal = comparison_df[comparison_df['fraud_type'] == 'credit_card']['temporal_strength'].mean()
            if 'mobile_money' in comparison_df['fraud_type'].values:
                mm_temporal = comparison_df[comparison_df['fraud_type'] == 'mobile_money']['temporal_strength'].mean()
                if abs(cc_temporal - mm_temporal) > 0.1:
                    insights.append(f"🔍 DOMAIN difference: Credit card ({cc_temporal:.3f}) vs Mobile money ({mm_temporal:.3f}) temporal strength")
        
        # Stability analysis
        stabilities = comparison_df['fraud_stability']
        if stabilities.min() > 0.8:
            insights.append("🔍 HIGH stability across all datasets - supports statistical method effectiveness")
        elif stabilities.max() < 0.6:
            insights.append("🔍 LOW stability across all datasets - complex models may be consistently needed")
        
        # Pattern diversity
        pattern_counts = comparison_df['num_patterns']
        if pattern_counts.min() >= 3:
            insights.append("🔍 Multiple temporal patterns found in all datasets - adaptive selection valuable")
        elif pattern_counts.max() <= 2:
            insights.append("🔍 Limited temporal pattern diversity - simple switching may suffice")
        
        return insights
    
    def print_comparative_summary(self, results: Dict[str, Any]):
        """Print executive summary of cross-dataset findings"""
        if 'comparative_analysis' not in results:
            print("❌ No comparative analysis available")
            return
        
        comparative = results['comparative_analysis']
        
        print("\\n📋 CROSS-DATASET ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Print comparison table
        if 'comparison_table' in comparative:
            df = pd.DataFrame(comparative['comparison_table'])
            print("\\n📊 Dataset Comparison:")
            print(f"{'Dataset':<15} {'Fraud Type':<12} {'Fraud Rate':<10} {'Temporal Strength':<16} {'Stability':<10} {'Patterns':<8}")
            print("-" * 80)
            
            for _, row in df.iterrows():
                print(f"{row['dataset']:<15} {row['fraud_type']:<12} {row['fraud_rate']:<10.3%} "
                      f"{row['temporal_strength']:<16.3f} {row['fraud_stability']:<10.3f} {row['num_patterns']:<8}")
        
        # Print insights
        if 'insights' in comparative:
            print("\\n🎯 KEY RESEARCH INSIGHTS:")
            for insight in comparative['insights']:
                print(f"   • {insight}")
        
        # Print recommendations
        print("\\n🛠️ RESEARCH DIRECTION RECOMMENDATIONS:")
        
        summary_stats = comparative.get('summary_statistics', {})
        temporal_range = summary_stats.get('temporal_strength_range', [0, 1])
        
        if temporal_range[1] < 0.2:
            print("   • Focus: Evidence-based framework proving statistical sufficiency")
            print("   • Contribution: Cross-domain validation of computational efficiency approach")
        elif temporal_range[0] > 0.3:
            print("   • Focus: Full adaptive temporal complexity framework")
            print("   • Contribution: Universal temporal pattern-based model selection")
        else:
            print("   • Focus: Domain-aware adaptive model selection")
            print("   • Contribution: Context-sensitive temporal complexity detection")


class DatasetSpecificAnalyzer(EmpiricalTemporalAnalyzer):
    """
    Adapter class to apply temporal analysis to different dataset formats
    """
    
    def __init__(self, df: pd.DataFrame, dataset_key: str):
        """Initialize with standardized dataframe"""
        self.data = df
        self.dataset_key = dataset_key
        
        # Ensure required columns exist
        if 'target' not in df.columns:
            raise ValueError("Standardized data must have 'target' column")
        if 'time_index' not in df.columns:
            raise ValueError("Standardized data must have 'time_index' column")
        
        # Set time_index as index for temporal analysis
        self.data = self.data.sort_values('time_index').reset_index(drop=True)
        
        print(f"📊 Dataset loaded: {len(self.data)} transactions, {self.data['target'].mean():.3%} fraud rate")


# Usage and testing framework
if __name__ == "__main__":
    print("🚀 Multi-Dataset Temporal Analysis Framework")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MultiDatasetTemporalAnalyzer()
    
    print("\\n1️⃣ Setup Phase:")
    analyzer.setup_kaggle_datasets()
    
    print("\\n2️⃣ Would you like to:")
    print("   A) Download datasets and run full analysis")
    print("   B) Run analysis on IEEE-CIS only (test framework)")
    print("   C) Setup instructions only")
    
    # For now, demonstrate with IEEE-CIS only
    print("\\n🧪 DEMO: Running analysis on IEEE-CIS dataset only")
    try:
        # Test framework with existing data
        results = analyzer.run_cross_domain_analysis(['ieee_cis'])
        analyzer.print_comparative_summary(results)
        
        print("\\n✅ Framework ready for multi-dataset analysis!")
        print("\\nTo analyze multiple datasets:")
        print("1. Run: analyzer.download_datasets(['paysim', 'european_cc'])")
        print("2. Run: analyzer.run_cross_domain_analysis(['ieee_cis', 'paysim', 'european_cc'])")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\\n🔧 Framework is ready - set up Kaggle API to proceed with full analysis")
