# TDDI with Uncertainty Estimation

This repository contains a complete implementation of TDDI for drug-drug interaction prediction with uncertainty estimation using ensemble methods.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/HienKha/tddi.git
cd tddi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python run_TDDI.py \
    --train_path data_splits/train_extracted.csv \
    --test_path data_splits/test_extracted.csv \
    --valid_path data_splits/validation_extracted.csv \
    --n_folds 3 \
    --num_epochs 300 \
    --patience 120 \
    --output_dir results
```

### With Feature Selection

If you have a list of features to drop:

```bash
python run_TDDI.py \
    --train_path data_splits/train_extracted.csv \
    --test_path data_splits/test_extracted.csv \
    --valid_path data_splits/validation_extracted.csv \
    --feature_list_path list_of_all_features_ascending_order.txt \
    --num_features_to_drop 6 \
    --output_dir results
```

### Command Line Arguments

- `--train_path`: Path to training CSV file (required)
- `--test_path`: Path to test CSV file (required)
- `--valid_path`: Path to validation CSV file (required)
- `--feature_list_path`: Path to file with features to drop (optional)
- `--num_features_to_drop`: Number of features to drop from the list (default: 0)
- `--n_folds`: Number of cross-validation folds (default: 3)
- `--num_epochs`: Maximum training epochs (default: 300)
- `--patience`: Early stopping patience (default: 120)
- `--output_dir`: Output directory for results (default: 'results')

## Project Structure

```
TDDI_github/
├── run_TDDI.py    # Main training script
├── models.py                 # Model classes (TDDI, FocalLoss)
├── utils.py                  # Utility classes (MemoryOptimizer, UncertaintyEstimator)
├── preprocessing.py          # Data preprocessing functions
├── training.py               # Training utilities
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.3.1+
- See `requirements.txt` for full list

## Acknowledgement
Part of the code was borrowed from TabTransformer [Link](https://github.com/lucidrains/tab-transformer-pytorch)
## License

[Add your license here]

