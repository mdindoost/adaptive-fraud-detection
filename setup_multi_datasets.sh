#!/bin/bash

# Multi-Dataset Temporal Analysis Setup Script
# Sets up environment for cross-domain fraud detection validation

echo "🚀 Setting up Multi-Dataset Temporal Analysis"
echo "=============================================="

# Check Python environment
echo "🐍 Checking Python environment..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

# Install required packages
echo "📦 Installing required packages..."
pip install pandas numpy scikit-learn scipy matplotlib seaborn kaggle

if [ $? -ne 0 ]; then
    echo "❌ Package installation failed"
    exit 1
else
    echo "✅ Packages installed successfully"
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/multi_datasets
mkdir -p results
mkdir -p figures

# Check if kaggle is working
echo "🔍 Testing Kaggle API..."
python -c "import kaggle; print('✅ Kaggle API ready')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Kaggle API not configured"
    echo "📋 Follow these steps to set up Kaggle:"
    echo "   1. Go to https://www.kaggle.com/account"
    echo "   2. Click 'Create New API Token'"
    echo "   3. Download kaggle.json"
    echo "   4. Run: mkdir -p ~/.kaggle"
    echo "   5. Run: cp kaggle.json ~/.kaggle/"
    echo "   6. Run: chmod 600 ~/.kaggle/kaggle.json"
else
    echo "✅ Kaggle API configured"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Save the framework files:"
echo "      - src/multi_dataset_temporal_analyzer.py"
echo "      - src/dataset_downloader.py"
echo ""
echo "   2. Test the framework:"
echo "      python src/dataset_downloader.py test"
echo ""
echo "   3. Download datasets and run full analysis:"
echo "      python src/dataset_downloader.py download"
echo ""
echo "📊 This will analyze temporal complexity across:"
echo "   • IEEE-CIS (your current data)"
echo "   • PaySim (mobile money fraud)"
echo "   • European Credit Card (different temporal characteristics)"
echo ""
echo "🔬 Research goal: Validate if temporal complexity findings"
echo "   generalize across fraud detection domains"
