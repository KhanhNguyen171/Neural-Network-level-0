# Notebooks Directory

Complete notebook-based workflow for MNIST CNN training and analysis.

## 📚 Structure

```
notebooks/
├── 00_data_exploration.ipynb      # Dataset exploration and statistics
├── 01_data_preprocessing.ipynb    # Data normalization and augmentation
├── 02_baseline_model.ipynb        # Model setup and training setup
├── 03_training_analysis.ipynb     # Training progress analysis
├── 04_error_analysis.ipynb        # Error analysis and confusion matrix
├── 05_visualization.ipynb         # Results visualization
│
├── experiments/
│   ├── exp_001_learning_rate.ipynb       # LR ablation study
│   ├── exp_002_batch_size.ipynb          # Batch size study
│   └── exp_003_regularization.ipynb      # Weight decay study
│
├── reports/
│   ├── confusion_matrix.ipynb     # Detailed confusion matrix analysis
│   ├── feature_maps.ipynb         # CNN feature visualization
│   └── model_comparison.ipynb     # Config comparison report
│
└── assets/
    ├── figures/                   # Generated plots
    └── tables/                    # Generated tables (CSV)
```

## 🚀 Quick Start

### 1. Start Jupyter Lab

```bash
# Using Docker
docker-compose up jupyter

# Or local
jupyter lab
```

### 2. Run Notebooks in Order

1. **00_data_exploration.ipynb** - Understand your data
2. **01_data_preprocessing.ipynb** - Prepare data
3. **02_baseline_model.ipynb** - Set up model
4. **03_training_analysis.ipynb** - Monitor training
5. **04_error_analysis.ipynb** - Analyze errors
6. **05_visualization.ipynb** - Visualize results

### 3. Explore Experiments

- `experiments/exp_001_learning_rate.ipynb` - Find optimal LR
- `experiments/exp_002_batch_size.ipynb` - Find optimal batch size
- `experiments/exp_003_regularization.ipynb` - Find optimal regularization

### 4. Generate Reports

- `reports/confusion_matrix.ipynb` - Detailed error analysis
- `reports/feature_maps.ipynb` - What model learned
- `reports/model_comparison.ipynb` - Compare configurations

## 📖 Notebook Descriptions

### Data Pipeline (00-05)

#### 00_data_exploration.ipynb
- Load and inspect MNIST dataset
- Class distribution analysis
- Data quality checks
- Sample visualizations
- Statistics summary

**Outputs**: 
- `assets/figures/class_distribution.png`
- `assets/figures/sample_digits.png`

#### 01_data_preprocessing.ipynb
- Normalization to [0,1]
- Data augmentation techniques
- Processing statistics
- Quality validation

**Outputs**:
- `assets/figures/data_augmentation.png`
- `assets/tables/preprocessing_stats.csv`

#### 02_baseline_model.ipynb
- Model architecture overview
- Configuration setup
- DataLoader creation
- Training commands reference

**Outputs**:
- Training ready environment
- Command reference

#### 03_training_analysis.ipynb
- Load checkpoint results
- Training history visualization
- Loss curves plotting
- Accuracy analysis
- Performance metrics

**Outputs**:
- `assets/figures/training_curves.png`
- `assets/tables/training_history.csv`
- `assets/tables/training_metrics.csv`

#### 04_error_analysis.ipynb
- Load and evaluate model
- Confusion matrix generation
- Per-class metrics
- Error distribution
- Misclassification patterns

**Outputs**:
- `assets/figures/confusion_matrix.png`
- Classification report
- Error statistics

#### 05_visualization.ipynb
- Prediction visualizations
- Confidence distributions
- Results summary plots
- Final statistics

**Outputs**:
- `assets/figures/predictions_sample.png`
- `assets/figures/confidence_distribution.png`
- `assets/tables/visualization_summary.csv`

### Experiments (experiments/)

#### exp_001_learning_rate.ipynb
- Test: [0.0001, 0.0005, 0.001, 0.005, 0.01]
- Metrics: Accuracy, Training Time
- Find: Optimal learning rate

**Outputs**:
- `assets/figures/exp_001_learning_rate.png`
- `assets/tables/exp_001_learning_rate.csv`

#### exp_002_batch_size.ipynb
- Test: [8, 16, 32, 64, 128]
- Metrics: Accuracy, Training Time, Convergence
- Find: Optimal batch size

**Outputs**:
- `assets/figures/exp_002_batch_size.png`
- `assets/tables/exp_002_batch_size.csv`

#### exp_003_regularization.ipynb
- Test: [0.0, 0.0001, 0.0005, 0.001, 0.01] weight decay
- Metrics: Train/Val Accuracy, Overfitting Gap
- Find: Optimal regularization

**Outputs**:
- `assets/figures/exp_003_regularization.png`
- `assets/tables/exp_003_regularization.csv`

### Reports (reports/)

#### confusion_matrix.ipynb
- Load trained model
- Generate predictions
- Confusion matrix heatmap
- Per-class accuracy
- Common misclassifications

**Outputs**:
- Detailed confusion matrix analysis
- Per-digit metrics
- Error patterns

#### feature_maps.ipynb
- Extract layer activations
- Visualize learned features
- Layer-wise analysis
- Feature interpretation

**Outputs**:
- Layer feature visualizations
- Activation patterns

#### model_comparison.ipynb
- Compare all 4 configurations
- Performance metrics comparison
- Training time analysis
- Recommendations

**Outputs**:
- `assets/figures/model_comparison.png`
- Configuration comparison table

## 🔄 Workflow

### Training Workflow
```
00_data_exploration
    ↓
01_data_preprocessing
    ↓
02_baseline_model → Run: python -m scripts.train
    ↓
03_training_analysis
    ↓
04_error_analysis
    ↓
05_visualization
```

### Analysis Workflow
```
04_error_analysis
    ↓
reports/confusion_matrix.ipynb
    ↓
reports/feature_maps.ipynb
    ↓
reports/model_comparison.ipynb
```

### Experimentation Workflow
```
experiments/exp_001_learning_rate.ipynb
    ↓
experiments/exp_002_batch_size.ipynb
    ↓
experiments/exp_003_regularization.ipynb
```

## 📊 Key Outputs

### Generated Figures
```
notebooks/assets/figures/
├── class_distribution.png
├── sample_digits.png
├── data_augmentation.png
├── training_curves.png
├── confusion_matrix.png
├── predictions_sample.png
├── confidence_distribution.png
├── exp_001_learning_rate.png
├── exp_002_batch_size.png
├── exp_003_regularization.png
├── feature_maps.png
└── model_comparison.png
```

### Generated Tables
```
notebooks/assets/tables/
├── preprocessing_stats.csv
├── training_history.csv
├── training_metrics.csv
├── visualization_summary.csv
├── exp_001_learning_rate.csv
├── exp_002_batch_size.csv
├── exp_003_regularization.csv
└── confusion_matrix_report.csv
```

## 💡 Tips

### Before Running Notebooks
1. Ensure data exists: `ls data/processed/`
2. Check model checkpoint: `ls artifacts/checkpoints/`
3. Have trained model before analysis notebooks

### Kernel Management
```python
# Restart kernel between major sections
from IPython.display import clear_output
import importlib
import sys
```

### Memory Management
```python
# For large datasets
del large_object
import gc
gc.collect()
```

### Plotting Tips
```python
# Save high-res figures
plt.savefig('path/to/figure.png', dpi=150, bbox_inches='tight')

# Display in notebook
%matplotlib inline
```

## 🐛 Troubleshooting

### Import Errors
```python
# Add project root to path
import sys
sys.path.append('.')
```

### Missing Data
```bash
# Download data
python -m scripts.download_data

# Check structure
ls -la data/processed/
```

### Model Not Loading
```bash
# Verify checkpoint exists
ls artifacts/checkpoints/

# Train model first
python -m scripts.train --config configs/experiments/debug.yaml
```

### CUDA Errors
```python
# Use CPU instead
import torch
device = torch.device('cpu')
```

## 📝 Configuration Reference

### Training Configs Used
- **Debug**: 2 epochs, BS=8, LR=0.001
- **Small**: 10 epochs, BS=128, LR=0.001
- **Baseline**: 20 epochs, BS=64, LR=0.001
- **Large**: 30 epochs, BS=32, LR=0.0005

## 🎓 Learning Path

**Beginner**: 00 → 01 → 02 → 03 → 05  
**Intermediate**: Add 04 and reports/  
**Advanced**: Add experiments/ and deep analysis  

## 📚 Related Documentation

- [Data Pipeline](../data/notebooks/README.md)
- [Scripts Documentation](../scripts/README.md)
- [Configuration Guide](../docs/VERSION_UPDATE.md)
- [Docker Setup](../Docker/README.md)

## ✅ Checklist

- [ ] Data downloaded and verified
- [ ] Model trained (basic config)
- [ ] Evaluation complete
- [ ] Notebooks run successfully
- [ ] Visualizations generated
- [ ] Reports reviewed
- [ ] Experiments planned

---

**Last Updated**: 2026-06-14  
**Notebook Version**: 1.0.0  
**Status**: ✅ All notebooks complete and tested
