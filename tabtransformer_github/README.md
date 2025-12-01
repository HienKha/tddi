# TabTransformer with Uncertainty Estimation

This repository contains a complete implementation of TabTransformer for drug-drug interaction prediction with uncertainty estimation using ensemble methods.

## Features

- **TabTransformer Architecture**: State-of-the-art transformer model for tabular data
- **Uncertainty Estimation**: Ensemble-based uncertainty quantification using:
  - Entropy uncertainty
  - Variance uncertainty
  - Mutual information
  - Confidence scores
- **Cross-Validation Training**: Stratified k-fold cross-validation for robust model training
- **Focal Loss**: Handles class imbalance effectively
- **Memory Optimization**: Efficient GPU memory management

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd tabtransformer_github
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python run_tabtransformer.py \
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
python run_tabtransformer.py \
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
tabtransformer_github/
├── run_tabtransformer.py    # Main training script
├── models.py                 # Model classes (TabTransformer, FocalLoss)
├── utils.py                  # Utility classes (MemoryOptimizer, UncertaintyEstimator)
├── preprocessing.py          # Data preprocessing functions
├── training.py               # Training utilities
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Key Components

### UncertaintyEstimator

The `UncertaintyEstimator` class provides multiple uncertainty metrics:

- **Entropy Uncertainty**: Measures prediction entropy
- **Variance Uncertainty**: Measures variance across ensemble predictions
- **Mutual Information**: Measures epistemic uncertainty
- **Confidence Scores**: Normalized confidence based on entropy

### EnhancedTabTransformerWithImprovements

Main model wrapper that:
- Creates TabTransformer models
- Manages ensemble of models
- Provides uncertainty estimation via `predict_with_uncertainty()`

### Training Pipeline

1. Data loading and preprocessing
2. Feature encoding (categorical and numerical)
3. Cross-validation training with early stopping
4. Ensemble prediction with uncertainty estimation
5. Results evaluation and saving

## Output

The script generates:
- **tabtransformer_results.csv**: Comprehensive results including:
  - Test accuracy and F1 scores
  - Cross-validation metrics
  - Uncertainty statistics
  - Confidence-based performance metrics

## Model Architecture

Default hyperparameters:
- Embedding dimension: 64
- Transformer depth: 3
- Attention heads: 16
- Attention dropout: 0.4
- Feed-forward dropout: 0.2
- Learning rate: 9.45e-05
- Batch size: 256
- Focal loss gamma: 1.0

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.3.1+
- See `requirements.txt` for full list

## Citation

If you use this code, please cite the original TabTransformer paper and this implementation.

## License

[Add your license here]

