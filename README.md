# DDI - Drug-Drug Interaction Classification with BiSHop

This repository contains code for training a BiSHop (Bi-Directional Cellular Learning for Tabular Data with Generalized Sparse Modern Hopfield Model) classifier for drug-drug interaction classification using cross-validation.

## Overview

The `train_bishop_cv.py` script performs 3-fold stratified cross-validation on drug-drug interaction data using the BiSHop model. It includes feature selection, data preprocessing, model training, and comprehensive evaluation metrics.

## Prerequisites

### System Requirements
- Python 3.10
- CUDA-capable GPU (recommended) or CPU
- Sufficient RAM for loading large datasets

### Data Files

Download the required data files from the Google Drive link:
- **Data Link**: https://drive.google.com/file/d/1Q5RhMkibuW0QIx-ywa4eMVKCrfPVmpwp/view?usp=drive_link

The dataset should include:
- `train_extracted.csv`: Training dataset
- `validation_extracted.csv`: Validation dataset  
- `list_of_all_features_ascending_order.txt`: List of all features in ascending order (used for feature selection)

## Installation

### 1. Create Conda Environment

```bash
conda create -n BiSHop python=3.10
conda activate BiSHop
```

### 2. Install PyTorch

Install PyTorch according to your CUDA version. For CUDA 12.1:

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

### 3. Install Dependencies

Install the required packages from the BiSHop requirements file:

```bash
pip install -r BiSHop/requirements.txt
```

### 4. Install Additional Dependencies

The script uses `polars` for efficient CSV loading, which may not be in the requirements file:

```bash
pip install polars
```

## Data Setup

### 1. Download Data

Download the data files from the Google Drive link provided above and extract them to your desired location.

### 2. Update File Paths

Edit `train_bishop_cv.py` and update the following paths in the `main()` function (around lines 196-198):

```python
train_path = "/path/to/your/data/train_extracted.csv"
valid_path = "/path/to/your/data/validation_extracted.csv"
features_file = "/path/to/your/data/list_of_all_features_ascending_order.txt"
```

**Example for Windows:**
```python
train_path = "D:/data/drugbank/data_splits/train_extracted.csv"
valid_path = "D:/data/drugbank/data_splits/validation_extracted.csv"
features_file = "D:/data/drugbank/list_of_all_features_ascending_order.txt"
```

**Example for Linux/Mac:**
```python
train_path = "/mnt/data/drugbank/data_splits/train_extracted.csv"
valid_path = "/mnt/data/drugbank/data_splits/validation_extracted.csv"
features_file = "/mnt/data/drugbank/list_of_all_features_ascending_order.txt"
```

## Configuration

You can modify the following parameters in the `main()` function of `train_bishop_cv.py`:

- **`batch_size`** (default: 32): Batch size for training
- **`num_epochs`** (default: 30): Number of training epochs per fold
- **`n_folds`** (default: 3): Number of cross-validation folds
- **`num_features_to_drop`** (default: 3506): Number of features to drop (6 + 3500)

## Running the Script

### Basic Usage

Activate the conda environment and run the script:

```bash
conda activate BiSHop
python train_bishop_cv.py
```

### Expected Output

The script will:
1. Load and display device information (CPU/GPU)
2. Load the feature list and drop specified features
3. Load and preprocess the training and validation datasets
4. Perform 3-fold stratified cross-validation
5. For each fold:
   - Train the BiSHop model
   - Evaluate on the validation set
   - Print detailed metrics (accuracy, precision, recall, F1-score)
6. Print a summary of cross-validation results with mean and standard deviation

### Output Metrics

For each fold, the script reports:
- **Classification Report**: Per-class precision, recall, and F1-score
- **Accuracy**: Overall classification accuracy
- **Weighted Metrics**: Precision, recall, and F1-score (weighted by class support)
- **Macro Metrics**: Precision, recall, and F1-score (unweighted mean per class)

At the end, a cross-validation summary shows the mean and standard deviation across all folds.

## Model Architecture

The script uses the BiSHop model with the following default configuration:
- Embedding dimension: 32
- Output dimension: 24
- Patch dimension: 8
- Factor: 10
- Encoder layers: 3
- Decoder layers: 4
- Attention heads: 4
- Model dimension: 256
- Feed-forward dimension: 512
- Dropout: 0.2

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Ensure all data file paths are correctly set in the script
   - Verify that the CSV files and feature list file exist at the specified paths

2. **CUDA Out of Memory**
   - Reduce `batch_size` in the configuration
   - Reduce the number of features (increase `num_features_to_drop`)
   - Use CPU instead of GPU (will be slower)

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r BiSHop/requirements.txt`
   - Install polars: `pip install polars`
   - Verify that the BiSHop directory is in the correct location

4. **Memory Issues with Large Datasets**
   - The script uses polars for efficient memory management
   - If issues persist, consider processing data in chunks or using a machine with more RAM

## Project Structure

```
DDI/
├── BiSHop/              # BiSHop model implementation
│   ├── models/          # Model definitions
│   ├── data/            # Data loading utilities
│   ├── exp/             # Experiment configurations
│   ├── utils/           # Utility functions
│   └── requirements.txt # Python dependencies
├── train_bishop_cv.py   # Main training script
└── README.md           # This file
```

## References

- **BiSHop Paper**: [BiSHop: Bi-Directional Cellular Learning for Tabular Data with Generalized Sparse Modern Hopfield Model](https://arxiv.org/abs/2404.03830)
- **BiSHop Repository**: [MAGICS-LAB/Bi-SHop](https://github.com/MAGICS-LAB/Bi-SHop)

## License

See the LICENSE file in the repository for license information.
