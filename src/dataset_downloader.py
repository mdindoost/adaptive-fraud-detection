"""
Dataset Downloader and Multi-Domain Analysis Runner
Simple script to download fraud datasets and run cross-domain validation
"""

import os
import sys
from multi_dataset_temporal_analyzer import MultiDatasetTemporalAnalyzer

def setup_kaggle_api():
    """Guide user through Kaggle API setup"""
    print("🔧 KAGGLE API SETUP")
    print("=" * 40)
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✅ Kaggle package installed")
    except ImportError:
        print("❌ Kaggle package not installed")
        print("Run: pip install kaggle")
        return False
    
    # Check for API credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print("✅ Kaggle credentials found")
        try:
            kaggle.api.authenticate()
            print("✅ Kaggle API authentication successful")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    else:
        print("❌ Kaggle credentials not found")
        print("Setup steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it at: ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_and_analyze(datasets=['paysim', 'european_cc'], skip_download=False):
    """Download datasets and run temporal analysis"""
    
    print("🚀 CROSS-DOMAIN FRAUD DETECTION ANALYSIS")
    print("=" * 50)
    
    analyzer = MultiDatasetTemporalAnalyzer()
    
    if not skip_download:
        print("\\n📥 DOWNLOADING DATASETS...")
        
        # Check Kaggle setup
        if not setup_kaggle_api():
            print("\\n❌ Kaggle API setup required. Please set up credentials first.")
            print("\\nAlternatively, manually download datasets:")
            print("• PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1")
            print("• European CC: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            return False
        
        # Download datasets
        success = analyzer.download_datasets(datasets)
        if not success:
            print("❌ Dataset download failed")
            return False
    
    print("\\n🔍 RUNNING TEMPORAL COMPLEXITY ANALYSIS...")
    
    # Always include IEEE-CIS in analysis
    datasets_to_analyze = ['ieee_cis'] + datasets
    
    try:
        # Run cross-domain analysis
        results = analyzer.run_cross_domain_analysis(datasets_to_analyze)
        
        print("\\n📊 ANALYSIS RESULTS")
        print("=" * 50)
        analyzer.print_comparative_summary(results)
        
        # Generate research insights
        print("\\n🧠 RESEARCH IMPLICATIONS")
        print("=" * 50)
        generate_research_conclusions(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def generate_research_conclusions(results):
    """Generate research conclusions from multi-dataset analysis"""
    
    comparative = results.get('comparative_analysis', {})
    insights = comparative.get('insights', [])
    
    if not insights:
        print("❌ No comparative insights available")
        return
    
    print("Based on cross-dataset temporal complexity analysis:\\n")
    
    # Categorize findings
    consistent_weak = any("CONSISTENT finding: All datasets show weak temporal dependencies" in insight for insight in insights)
    consistent_strong = any("CONSISTENT finding: All datasets show strong temporal dependencies" in insight for insight in insights)
    mixed_findings = any("MIXED finding" in insight for insight in insights)
    
    if consistent_weak:
        print("🎯 RESEARCH CONCLUSION: Statistical Method Sufficiency")
        print("   • Empirical evidence across multiple fraud domains")
        print("   • Temporal complexity rarely justifies computational overhead")
        print("   • Focus: Computational efficiency optimization")
        print("   • Contribution: Evidence-based model selection framework")
        
    elif consistent_strong:
        print("🎯 RESEARCH CONCLUSION: Universal Temporal Complexity") 
        print("   • Strong temporal dependencies across fraud domains")
        print("   • TGN/temporal modeling consistently valuable")
        print("   • Focus: Advanced temporal pattern detection")
        print("   • Contribution: Universal temporal complexity framework")
        
    elif mixed_findings:
        print("🎯 RESEARCH CONCLUSION: Domain-Aware Adaptive Selection")
        print("   • Temporal complexity varies by fraud domain")
        print("   • Adaptive selection based on domain characteristics")
        print("   • Focus: Context-sensitive temporal analysis")
        print("   • Contribution: Domain-aware model selection framework")
    
    else:
        print("🎯 RESEARCH CONCLUSION: Inconclusive - Need Additional Data")
        print("   • Current datasets insufficient for clear conclusions")
        print("   • Recommendation: Analyze additional fraud domains")
    
    print("\\n📋 NEXT STEPS:")
    if consistent_weak or mixed_findings:
        print("   1. Develop fraud-rate-regime based model selection")
        print("   2. Focus on computational efficiency optimization") 
        print("   3. Build evidence for statistical method sufficiency")
    else:
        print("   1. Implement full temporal complexity framework")
        print("   2. Develop adaptive TGN model selection")
        print("   3. Create temporal pattern characterization metrics")

def main():
    """Main execution function with options"""
    
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
    else:
        print("🚀 MULTI-DATASET FRAUD DETECTION ANALYSIS")
        print("=" * 50)
        print("Options:")
        print("  setup    - Setup Kaggle API credentials")
        print("  test     - Test with IEEE-CIS dataset only")  
        print("  download - Download datasets and run full analysis")
        print("  analyze  - Run analysis (skip download)")
        print("")
        option = input("Choose option [test/download/analyze/setup]: ").lower()
    
    if option == 'setup':
        setup_kaggle_api()
        
    elif option == 'test':
        print("🧪 TESTING FRAMEWORK WITH IEEE-CIS ONLY")
        analyzer = MultiDatasetTemporalAnalyzer()
        try:
            results = analyzer.run_cross_domain_analysis(['ieee_cis'])
            analyzer.print_comparative_summary(results)
            print("\\n✅ Framework test successful!")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    elif option == 'download':
        download_and_analyze(['paysim', 'european_cc'], skip_download=False)
        
    elif option == 'analyze':
        download_and_analyze(['paysim', 'european_cc'], skip_download=True)
        
    else:
        print("❌ Invalid option. Use: setup, test, download, or analyze")

if __name__ == "__main__":
    main()
