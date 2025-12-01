"""
Main script to run TabTransformer training with uncertainty estimation.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
import os
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import MemoryOptimizer, UncertaintyEstimator
from models import EnhancedTabTransformerWithImprovements, FocalLoss
from preprocessing import load_and_clean_data, preprocess_ultra_fast
from training import train_single_model

warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def load_feature_list(feature_list_path: str) -> List[str]:
    """Load list of features to drop from file."""
    with open(feature_list_path, "r", encoding="utf-8") as f:
        columns_to_drop = [line.strip() for line in f if line.strip()]
    return columns_to_drop


def main(
    train_path: str,
    test_path: str,
    valid_path: str,
    feature_list_path: Optional[str] = None,
    number_of_features_to_drop: int = 0,
    best_params: Optional[Dict] = None,
    n_folds: int = 3,
    num_epochs: int = 300,
    patience: int = 120,
    output_dir: str = "results"
):
    """
    Main training and evaluation pipeline.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        valid_path: Path to validation CSV
        feature_list_path: Path to file with features to drop (optional)
        number_of_features_to_drop: Number of features to drop from the list
        best_params: Dictionary of hyperparameters
        n_folds: Number of cross-validation folds
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        output_dir: Directory to save results
    """
    
    # Default hyperparameters
    if best_params is None:
        best_params = {
            'dim': 64,
            'depth': 3,
            'heads': 16,
            'attn_dropout': 0.4,
            'ff_dropout': 0.2,
            'mlp_hidden_mult_1': 2,
            'mlp_hidden_mult_2': 2,
            'learning_rate': 9.452571391072311e-05,
            'weight_decay': 0.0001544608907504709,
            'batch_size': 256,
            'gamma': 1.0
        }
    
    # Load feature list if provided
    columns_to_drop = []
    if feature_list_path and os.path.exists(feature_list_path):
        columns_to_drop = load_feature_list(feature_list_path)
        columns_to_drop = columns_to_drop[:number_of_features_to_drop]
        print(f"Columns to drop: {len(columns_to_drop)} columns")
    
    # Load and clean data
    print("Loading datasets...")
    train_df = load_and_clean_data(train_path, columns_to_drop)
    test_df = load_and_clean_data(test_path, columns_to_drop)
    valid_df = load_and_clean_data(valid_path, columns_to_drop)
    
    if train_df is None or test_df is None or valid_df is None:
        raise ValueError("Failed to load one or more datasets")
    
    print(f"Train size: {len(train_df):,} samples")
    print(f"Test size: {len(test_df):,} samples") 
    print(f"Valid size: {len(valid_df):,} samples")
    
    # Preprocess data
    (X_train, y_train, X_test, y_test, X_valid, y_valid, 
     categorical_cols, numerical_cols, preprocessors) = preprocess_ultra_fast(
        train_df, test_df, valid_df, target_col='class'
    )
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"X_valid: {X_valid.shape}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    
    MemoryOptimizer.cleanup_memory()
    
    # Determine model architecture
    categories = [X_train[col].nunique() for col in categorical_cols]
    num_classes = len(np.unique(y_train))
    num_continuous = len(numerical_cols)
    
    print(f"\nModel Configuration:")
    print(f"  - Number of categorical features: {len(categorical_cols)}")
    print(f"  - Categories (unique values per feature): {categories}")
    print(f"  - Number of continuous features: {num_continuous}")
    print(f"  - Number of classes: {num_classes}")
    
    # Create model wrapper
    enhanced_model = EnhancedTabTransformerWithImprovements(
        categories=categories,
        num_continuous=num_continuous,
        num_classes=num_classes,
        device=device,
        best_params=best_params
    )
    
    print(f"\nHyperparameters:")
    print(f"  - Embedding dimension: {best_params['dim']}")
    print(f"  - Transformer depth: {best_params['depth']}")
    print(f"  - Attention heads: {best_params['heads']}")
    print(f"  - Learning rate: {best_params['learning_rate']:.2e}")
    print(f"  - Batch size: {best_params['batch_size']}")
    
    # Combine train and validation for cross-validation
    X_combined = pd.concat([X_train, X_valid], ignore_index=True)
    y_combined = np.concatenate([y_train, y_valid])
    
    print(f"\nCombined training data shape: {X_combined.shape}")
    print(f"Combined target shape: {y_combined.shape}")
    
    # Cross-validation training
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X_combined, y_combined))
    
    fold_models = []
    fold_scores = []
    all_train_losses = []
    all_val_accuracies = []
    
    print(f"\nStarting {n_folds}-fold cross-validation training...")
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_fold_train = X_combined.iloc[train_idx]
        y_fold_train = y_combined[train_idx]
        X_fold_val = X_combined.iloc[val_idx]
        y_fold_val = y_combined[val_idx]
        
        print(f"\nFold {fold+1}: Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")
        
        model, val_score, train_losses, val_accs = train_single_model(
            X_fold_train, y_fold_train, X_fold_val, y_fold_val, fold+1,
            enhanced_model, best_params, categories, num_continuous, device,
            num_epochs=num_epochs, patience=patience
        )
        
        fold_models.append(model)
        fold_scores.append(val_score)
        all_train_losses.append(train_losses)
        all_val_accuracies.append(val_accs)
        
        MemoryOptimizer.cleanup_memory()
    
    enhanced_model.models = fold_models
    
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results:")
    print(f"Mean CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"Individual fold scores: {[f'{score:.4f}' for score in fold_scores]}")
    print(f"Best fold: {np.argmax(fold_scores)+1} with score: {np.max(fold_scores):.4f}")
    print(f"Worst fold: {np.argmin(fold_scores)+1} with score: {np.min(fold_scores):.4f}")
    print(f"{'='*50}")
    
    # Evaluate on test set with uncertainty estimation
    print(f"\nEvaluating on test set with uncertainty estimation...")
    test_df_for_prediction = X_test.copy()
    if y_test is not None:
        test_df_for_prediction['class'] = y_test
    
    # Use UncertaintyEstimator for predictions
    test_results = enhanced_model.predict_with_uncertainty(test_df_for_prediction)
    
    predictions = test_results['predictions']
    probabilities = test_results['probabilities']
    uncertainties = test_results['uncertainties']
    
    print(f"Predictions completed for {len(predictions)} samples")
    
    # Calculate metrics
    if y_test is not None:
        test_accuracy = accuracy_score(y_test, predictions)
        print(f"\nTest Set Performance:")
        print(f"{'='*40}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        cls_report = classification_report(y_test, predictions, output_dict=True)
        f1_macro = cls_report['macro avg']['f1-score']
        f1_weighted = cls_report['weighted avg']['f1-score']
        precision_macro = cls_report['macro avg']['precision']
        precision_weighted = cls_report['weighted avg']['precision']
        recall_macro = cls_report['macro avg']['recall']
        recall_weighted = cls_report['weighted avg']['recall']
        
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        
        # Uncertainty analysis
        print(f"\nUncertainty Statistics:")
        print(f"Mean Entropy: {np.mean(uncertainties['entropy']):.4f}")
        print(f"Mean Confidence: {np.mean(uncertainties['confidence']):.4f}")
        print(f"Mean Variance: {np.mean(uncertainties['variance']):.4f}")
        
        confidence_scores = uncertainties['confidence']
        high_conf_mask = confidence_scores > 0.8
        medium_conf_mask = (confidence_scores >= 0.5) & (confidence_scores <= 0.8)
        low_conf_mask = confidence_scores < 0.5
        
        print(f"\nConfidence Distribution:")
        print(f"High confidence (>0.8): {np.sum(high_conf_mask):,} ({np.mean(high_conf_mask)*100:.1f}%)")
        print(f"Medium confidence (0.5-0.8): {np.sum(medium_conf_mask):,} ({np.mean(medium_conf_mask)*100:.1f}%)")
        print(f"Low confidence (<0.5): {np.sum(low_conf_mask):,} ({np.mean(low_conf_mask)*100:.1f}%)")
        
        # Accuracy by confidence level
        y_test_np = np.asarray(y_test)
        preds_np = np.asarray(predictions)
        
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(y_test_np[high_conf_mask], preds_np[high_conf_mask])
            print(f"High confidence accuracy: {high_conf_acc:.4f}")
        
        if medium_conf_mask.sum() > 0:
            medium_conf_acc = accuracy_score(y_test_np[medium_conf_mask], preds_np[medium_conf_mask])
            print(f"Medium confidence accuracy: {medium_conf_acc:.4f}")
        
        if low_conf_mask.sum() > 0:
            low_conf_acc = accuracy_score(y_test_np[low_conf_mask], preds_np[low_conf_mask])
            print(f"Low confidence accuracy: {low_conf_acc:.4f}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        result_dict = {
            'num_features_dropped': number_of_features_to_drop,
            'num_classes': num_classes,
            'test_accuracy': test_accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'mean_entropy': np.mean(uncertainties['entropy']),
            'mean_confidence': np.mean(uncertainties['confidence']),
            'mean_variance': np.mean(uncertainties['variance']),
        }
        
        results_df = pd.DataFrame([result_dict])
        results_path = os.path.join(output_dir, f"tabtransformer_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
    
    print("\nTraining and evaluation completed!")
    return enhanced_model, test_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TabTransformer with uncertainty estimation')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--valid_path', type=str, required=True,
                        help='Path to validation CSV file')
    parser.add_argument('--feature_list_path', type=str, default=None,
                        help='Path to file with features to drop')
    parser.add_argument('--num_features_to_drop', type=int, default=0,
                        help='Number of features to drop')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of CV folds')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=120,
                        help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    main(
        train_path=args.train_path,
        test_path=args.test_path,
        valid_path=args.valid_path,
        feature_list_path=args.feature_list_path,
        number_of_features_to_drop=args.num_features_to_drop,
        n_folds=args.n_folds,
        num_epochs=args.num_epochs,
        patience=args.patience,
        output_dir=args.output_dir
    )

