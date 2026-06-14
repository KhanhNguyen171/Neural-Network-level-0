# Notebooks Documentation

**Project**: CNN MNIST  
**Version**: 2.0.0  
**Date**: 2026-06-14  

---

## 📓 Notebook Overview

### Current Notebooks (data/notebooks/)

| # | Notebook | Purpose | Status |
|---|----------|---------|--------|
| 1 | 01_data_ingestion.ipynb | Download MNIST dataset | ✅ Ready |
| 2 | 02_data_validation.ipynb | Validate downloaded data | ✅ Ready |
| 3 | 03_data_profiling.ipynb | Profile dataset statistics | ✅ Ready |
| 4 | 04_data_preprocessing.ipynb | Normalize & preprocess data | ✅ Ready |
| 5 | 05_dataset_split.ipynb | Train/valid/test split | ✅ Ready |
| 6 | 06_data_export.ipynb | Export processed data | ✅ Ready |

---

## 📊 Notebook Workflow

```
01. Data Ingestion
        ↓
        Download MNIST dataset to data/raw/mnist/
        
02. Data Validation
        ↓
        Verify dataset integrity & format
        
03. Data Profiling
        ↓
        Analyze dataset statistics & distribution
        
04. Data Preprocessing
        ↓
        Normalize images, handle missing values
        
05. Dataset Split
        ↓
        Split into train/valid/test (6:2:2 ratio)
        
06. Data Export
        ↓
        Export to NPY format in data/processed/
        
└─→ Ready for training scripts!
```

---

## 🆕 Planned Notebooks (to create)

| # | Notebook | Purpose | Type |
|---|----------|---------|------|
| 7 | 07_model_training.ipynb | Training workflow & monitoring | Training |
| 8 | 08_model_evaluation.ipynb | Evaluate model performance | Evaluation |
| 9 | 09_inference_demo.ipynb | Run inference on sample images | Demo |
| 10 | 10_results_analysis.ipynb | Analyze & visualize results | Analysis |
| 11 | 11_model_comparison.ipynb | Compare different configs | Comparison |
| 12 | 12_deployment_guide.ipynb | Export & deployment guide | Deployment |

---

## 📁 Notebook Organization

### Current Structure
```
data/notebooks/          (Data processing notebooks)
├── 01_data_ingestion.ipynb
├── 02_data_validation.ipynb
├── 03_data_profiling.ipynb
├── 04_data_preprocessing.ipynb
├── 05_dataset_split.ipynb
└── 06_data_export.ipynb

notebooks/              (Main notebooks folder - empty for v2.0.0)
```

### Proposed Structure for v2.1.0+
```
notebooks/              (Main analysis & demo notebooks)
├── 01_quick_start.ipynb              (Getting started guide)
├── 02_data_exploration.ipynb         (EDA)
├── 03_model_training.ipynb           (Training workflow)
├── 04_model_evaluation.ipynb         (Evaluation)
├── 05_inference_examples.ipynb       (Inference demo)
├── 06_results_visualization.ipynb    (Plots & analysis)
├── 07_deployment_tutorial.ipynb      (Export guide)
└── README.md

data/notebooks/         (Data prep notebooks)
├── 01_data_ingestion.ipynb
├── 02_data_validation.ipynb
├── 03_data_profiling.ipynb
├── 04_data_preprocessing.ipynb
├── 05_dataset_split.ipynb
├── 06_data_export.ipynb
└── README.md
```

---

## 🎯 Usage Guidelines

### For Data Processing
```bash
# Run existing notebooks in data/notebooks/
# Used to prepare data in data/processed/
cd data/notebooks
jupyter notebook 01_data_ingestion.ipynb
```

### For Model Training & Analysis
```bash
# Run notebooks in notebooks/ (when created)
# Used for training, evaluation, inference
cd notebooks
jupyter notebook 03_model_training.ipynb
```

### Integration with Scripts
```
Notebooks (exploratory)
    ↓
Scripts (production)
    ↓
Deployment
```

- **Notebooks**: For exploration, analysis, prototyping
- **Scripts**: For production training, inference, export
- **Integration**: Notebooks use data from scripts; scripts use configs from notebooks

---

## 📝 Notebook Best Practices

### 1. Markdown Documentation
```markdown
# Section Title
Description of what this cell does
- Key point 1
- Key point 2
```

### 2. Cell Organization
- Cell 1: Markdown header
- Cell 2-3: Imports & setup
- Cell 4+: Logic & analysis

### 3. Output Management
```python
# Show key outputs
print(f"Dataset shape: {dataset.shape}")
print(f"Classes: {num_classes}")

# Visualize with plots
import matplotlib.pyplot as plt
plt.show()
```

### 4. Error Handling
```python
try:
    # Code that might fail
    result = process_data()
except Exception as e:
    print(f"Error: {e}")
    # Handle gracefully
```

### 5. Config Integration
```python
from pathlib import Path
import yaml

# Load config
config = yaml.safe_load(open("configs/data.yaml"))
data_dir = config["data_dir"]
```

---

## 🔧 Notebook Setup

### Requirements
```
jupyter>=1.0
matplotlib>=3.5
numpy>=1.19
pandas>=1.3
torch>=1.9
torchvision>=0.10
pyyaml>=5.4
```

### Installation
```bash
# Install Jupyter
pip install jupyter

# Start Jupyter server
jupyter notebook

# Or use VS Code
# Install Jupyter extension
# Open .ipynb files directly
```

### Kernel Selection
- Use Python kernel
- Check kernel matches project environment
- Install packages in notebook if needed: `!pip install package`

---

## 📊 Data Pipeline Reference

### Input Data (from notebooks)
- **Source**: `data/raw/mnist/` (downloaded by notebook 1)
- **Format**: PyTorch MNIST dataset
- **Size**: ~70,000 images

### Processed Data (output from notebooks)
- **Location**: `data/processed/`
- **Format**: NumPy .npy arrays
- **Structure**:
  ```
  X_train.npy     (50,000, 28, 28)
  X_valid.npy     (10,000, 28, 28)
  X_test.npy      (10,000, 28, 28)
  y_train.npy     (50,000,)
  y_valid.npy     (10,000,)
  y_test.npy      (10,000,)
  split_metadata.json
  ```

### Training (from scripts)
- **Input**: Data from `data/processed/`
- **Config**: Files from `configs/experiments/`
- **Output**: Model checkpoint in `artifacts/checkpoints/`

---

## 🚀 Quick Start: Running Notebooks

### Step 1: Download Data
```bash
cd data/notebooks
jupyter notebook 01_data_ingestion.ipynb
# Run all cells (Ctrl+A → Shift+Enter)
```

### Step 2: Validate & Profile
```bash
# Run 02, 03 notebooks
jupyter notebook 02_data_validation.ipynb
jupyter notebook 03_data_profiling.ipynb
```

### Step 3: Preprocess & Export
```bash
# Run 04, 05, 06 notebooks
jupyter notebook 04_data_preprocessing.ipynb
jupyter notebook 05_dataset_split.ipynb
jupyter notebook 06_data_export.ipynb
```

### Step 4: Verify Data
```bash
# Check output
ls data/processed/
# Should have X_train.npy, y_train.npy, etc.
```

### Step 5: Train Model
```bash
# Use scripts (not notebooks for production)
python -m scripts.train --config configs/experiments/baseline.yaml
```

---

## 📈 Expected Notebook Outputs

### Notebook 1: Data Ingestion
```
✓ Downloaded training set: 60,000 images
✓ Downloaded test set: 10,000 images
✓ Saved to: data/raw/mnist/
```

### Notebook 2: Data Validation
```
✓ Training set shape: (60000, 28, 28, 1)
✓ Test set shape: (10000, 28, 28, 1)
✓ Data type: uint8
✓ Value range: [0, 255]
✓ No missing values: ✓
```

### Notebook 3: Data Profiling
```
✓ Class distribution: Balanced (10 classes, ~6,000 each)
✓ Image statistics: Mean=33, Std=73
✓ Min pixel: 0, Max pixel: 255
✓ Sample visualizations: 10 images per class
```

### Notebook 4: Data Preprocessing
```
✓ Normalized pixel values: [0, 1]
✓ Center-cropped to (28, 28)
✓ Standardized: (X - 0.1307) / 0.3081
```

### Notebook 5: Dataset Split
```
✓ Train set: 50,000 images (70%)
✓ Valid set: 10,000 images (14%)
✓ Test set: 10,000 images (16%)
✓ Metadata saved: split_metadata.json
```

### Notebook 6: Data Export
```
✓ Exported X_train.npy (50000, 28, 28)
✓ Exported X_valid.npy (10000, 28, 28)
✓ Exported X_test.npy (10000, 28, 28)
✓ Exported y_train.npy (50000,)
✓ Exported y_valid.npy (10000,)
✓ Exported y_test.npy (10000,)
✓ Exported split_metadata.json
```

---

## 🔗 Integration Points

### Notebook → Scripts
```
Notebook outputs (data/processed/)
    ↓
Train script reads from data/processed/
    ↓
Model checkpoint saved to artifacts/checkpoints/
```

### Scripts → Notebooks
```
Config files (configs/experiments/)
    ↓
Notebooks can read configs for reproducibility
    ↓
Results logged for analysis
```

### End-to-End Workflow
```
01. Notebooks prepare data
    ↓
02. Scripts train model
    ↓
03. Scripts evaluate model
    ↓
04. Notebooks analyze results
    ↓
05. Deploy with scripts/export_model.py
```

---

## 📌 Important Notes

### When to Use Notebooks
✅ **Notebooks are for**:
- Exploratory data analysis
- Prototyping & experimentation
- Visualization & analysis
- Interactive debugging
- Documentation & reporting

### When to Use Scripts
✅ **Scripts are for**:
- Production training
- Batch processing
- Automated workflows
- Deployment
- Scheduled jobs

### Common Mistakes
❌ **DON'T**:
- Don't modify data in place (always save copies)
- Don't forget to restart kernel between runs
- Don't hardcode paths (use Path from pathlib)
- Don't skip validation steps
- Don't ignore notebook outputs

✅ **DO**:
- Use config files for parameters
- Save outputs with timestamps
- Document assumptions
- Test on small samples first
- Version control notebooks with `.gitignore`

---

## 🔍 Troubleshooting

### Issue: Data not found
```
Solution: Run 01_data_ingestion.ipynb first
```

### Issue: Kernel crash
```
Solution: Restart kernel and run cells sequentially
```

### Issue: Memory error
```
Solution: Process data in batches or use smaller dataset
```

### Issue: Import errors
```
Solution: Ensure environment has all packages installed
```

### Issue: Paths don't work
```
Solution: Use Path from pathlib and relative paths
```

---

## 📞 Support Resources

- See `SCRIPTS_FIXED_GUIDE.md` for script usage
- See `FINAL_FIX_SUMMARY.md` for technical details
- See individual notebook markdown cells for explanations
- See `VERSION_UPDATE.md` for version history

---

## Next Steps

1. ✅ Run existing notebooks in `data/notebooks/` to prepare data
2. ⏳ Create new notebooks in `notebooks/` for training & analysis
3. ⏳ Create master notebook for quick-start guide
4. ⏳ Add visualization notebooks for results

---

**Current Status**: 6 data notebooks ready | 0 analysis notebooks created  
**Version**: 2.0.0  
**Ready to proceed?** ✅
