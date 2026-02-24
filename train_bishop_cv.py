import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
import gc
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BiSHop'))

from models.model import BiSHop

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class MemoryOptimizer:        
    @staticmethod
    def cleanup_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_and_clean_data(path, columns_to_drop):
    """Load CSV using polars for memory efficiency and drop specified columns"""
    try:
        df = pl.read_csv(path)
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(existing_cols_to_drop)
        result = df.to_pandas()
        del df
        MemoryOptimizer.cleanup_memory()
        return result
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def preprocess_data(train_df, valid_df, target_col='class'):
    """Preprocess data: encode labels and separate features from target"""
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_valid = valid_df.drop(columns=[target_col], errors='ignore')
    y_valid = valid_df[target_col] if target_col in valid_df.columns else None
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    if y_valid is not None:
        y_valid_encoded = label_encoder.transform(y_valid)
    else:
        y_valid_encoded = None
    
    # All features are numerical
    print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X_train, y_train_encoded, X_valid, y_valid_encoded, label_encoder


def create_dataloader(X, y, batch_size, shuffle=True):
    """Create DataLoader for BiSHop (needs separate cat and num tensors)"""
    # All features are numerical, so x_cat is empty
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.LongTensor(y)
    
    # Create empty categorical tensor (shape: batch_size, 0)
    X_cat = torch.empty((len(X_tensor), 0), dtype=torch.long)
    X_num = X_tensor
    
    dataset = TensorDataset(X_cat, X_num, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_bishop_model(model, train_loader, val_loader, num_epochs=10, device=device):
    """Train BiSHop model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat = batch_cat.long().to(device)
            batch_num = batch_num.float().to(device)
            batch_y = batch_y.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_cat, batch_num)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
    
    return model


def evaluate_model(model, val_loader, device=device):
    """Evaluate model and return predictions and true labels"""
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_cat, batch_num, batch_y in val_loader:
            batch_cat = batch_cat.long().to(device)
            batch_num = batch_num.float().to(device)
            batch_y = batch_y.long().to(device)
            
            outputs = model(batch_cat, batch_num)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(batch_y.cpu().numpy())
    
    return np.array(all_preds), np.array(all_trues)


def print_metrics(y_true, y_pred, fold_num):
    """Print classification report and all requested metrics"""
    print(f"\n{'='*60}")
    print(f"Fold {fold_num} Results")
    print(f"{'='*60}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Calculate all metrics
    accuracy = accuracy_score(y_true, y_pred)
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nMetrics Summary:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  Weighted Precision: {weighted_precision:.4f}")
    print(f"  Weighted Recall:    {weighted_recall:.4f}")
    print(f"  Weighted F1:        {weighted_f1:.4f}")
    print(f"  Macro Precision:    {macro_precision:.4f}")
    print(f"  Macro Recall:       {macro_recall:.4f}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'accuracy': accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def main():
    # Configuration
    train_path = "/mnt/data/drugbank/data_splits/train_extracted.csv"
    valid_path = "/mnt/data/drugbank/data_splits/validation_extracted.csv"
    features_file = "/mnt/data/drugbank/list_of_all_features_ascending_order.txt"
    
    batch_size = 32
    num_epochs = 30
    n_folds = 3
    num_features_to_drop = 6 + 3500
    
    # Load features to drop
    print("Loading features to drop...")
    with open(features_file, "r", encoding="utf-8") as f:
        all_features = [line.strip() for line in f if line.strip()]
    
    columns_to_drop = all_features[:num_features_to_drop]
    print(f"Dropping {len(columns_to_drop)} features: {columns_to_drop}")
    
    # Load data
    print("\nLoading datasets...")
    train_df = load_and_clean_data(train_path, columns_to_drop)
    valid_df = load_and_clean_data(valid_path, columns_to_drop)
    
    if train_df is None or valid_df is None:
        print("Error: Failed to load datasets")
        return
    
    print(f"Train size: {len(train_df):,} samples")
    print(f"Valid size: {len(valid_df):,} samples")
    
    # Preprocess
    print("\nPreprocessing data...")
    X_train, y_train, X_valid, y_valid, label_encoder = preprocess_data(train_df, valid_df)
    
    # Combine train and validation for CV
    X_combined = pd.concat([X_train, X_valid], ignore_index=True)
    y_combined = np.concatenate([y_train, y_valid])
    
    print(f"\nCombined data shape: {X_combined.shape}")
    print(f"Number of features: {X_combined.shape[1]}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Get model parameters
    n_cat = 0  # No categorical features
    n_num = X_combined.shape[1]  # All features are numerical
    n_out = len(label_encoder.classes_)  # Number of classes
    
    print(f"\nModel parameters:")
    print(f"  n_cat: {n_cat}")
    print(f"  n_num: {n_num}")
    print(f"  n_out: {n_out}")
    
    # 3-fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=100)
    cv_splits = list(skf.split(X_combined, y_combined))
    
    all_fold_metrics = []
    
    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-fold Cross-Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        print(f"\n\nFold {fold}/{n_folds}")
        print(f"Train size: {len(train_idx):,}, Val size: {len(val_idx):,}")
        
        # Split data
        X_fold_train = X_combined.iloc[train_idx]
        y_fold_train = y_combined[train_idx]
        X_fold_val = X_combined.iloc[val_idx]
        y_fold_val = y_combined[val_idx]
        
        # Create data loaders
        train_loader = create_dataloader(X_fold_train, y_fold_train, batch_size, shuffle=True)
        val_loader = create_dataloader(X_fold_val, y_fold_val, batch_size, shuffle=False)
        
        # Create model
        model = BiSHop(
            n_cat=n_cat,
            n_num=n_num,
            n_out=n_out,
            emb_dim=32,
            out_dim=24,
            patch_dim=8,
            factor=10,
            flip=True,
            n_agg=4,
            actv='entmax',
            hopfield=True,
            d_model=256,
            d_ff=512,
            n_heads=4,
            e_layer=3,
            d_layer=4,
            dropout=0.2,
            share=True,
            share_div=8,
            share_add=False,
            full_dropout=False,
            emb_dropout=0.1,
            mlp_actv=nn.ReLU(),
            mlp_bn=True,
            mlp_bn_final=False,
            mlp_dropout=0.2,
            mlp_hidden=(4, 2, 1),
            mlp_skip=True,
            mlp_softmax=False,
            device=device
        ).to(device)
        
        # Get bins for numerical features (required by BiSHop)
        # Move tensor to device before calling get_bins to ensure quantiles are on correct device
        X_num_tensor = torch.FloatTensor(X_fold_train.values).to(device)
        model.get_bins(X_num_tensor)
        
        # Train model
        print(f"\nTraining model for {num_epochs} epochs...")
        model = train_bishop_model(model, train_loader, val_loader, num_epochs, device)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred, y_true = evaluate_model(model, val_loader, device)
        
        # Print metrics
        fold_metrics = print_metrics(y_true, y_pred, fold)
        all_fold_metrics.append(fold_metrics)
        
        # Cleanup
        del model
        MemoryOptimizer.cleanup_memory()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary")
    print(f"{'='*60}")
    
    metrics_names = ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1',
                     'macro_precision', 'macro_recall', 'macro_f1']
    
    for metric_name in metrics_names:
        values = [fold_metrics[metric_name] for fold_metrics in all_fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name:20s}: {mean_val:.4f} (+/- {std_val:.4f})")
        print(f"  Fold values: {[f'{v:.4f}' for v in values]}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

