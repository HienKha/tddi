"""
Training utilities for TabTransformer.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from .models import FocalLoss
from .utils import MemoryOptimizer


def train_single_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val, 
                      fold_num, enhanced_model, best_params, categories, 
                      num_continuous, device, num_epochs=300, patience=120):
    """
    Train a single model for one fold.
    
    Args:
        X_fold_train: Training features
        y_fold_train: Training labels
        X_fold_val: Validation features
        y_fold_val: Validation labels
        fold_num: Fold number for logging
        enhanced_model: EnhancedTabTransformerWithImprovements instance
        best_params: Dictionary of hyperparameters
        categories: List of category sizes
        num_continuous: Number of continuous features
        device: torch device
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
    
    Returns:
        Tuple of (trained_model, best_val_acc, train_losses, val_accuracies)
    """
    print(f"\n=== Training Fold {fold_num} ===")
    
    model = enhanced_model.create_model()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    criterion = FocalLoss(gamma=best_params['gamma'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_fold_train.values),
        torch.LongTensor(y_fold_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_fold_val.values),
        torch.LongTensor(y_fold_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            cat_features = batch_x[:, :len(categories)].long()
            cont_features = batch_x[:, len(categories):] if num_continuous > 0 else None
            
            optimizer.zero_grad()
            outputs = model(cat_features, cont_features)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                cat_features = batch_x[:, :len(categories)].long()
                cont_features = batch_x[:, len(categories):] if num_continuous > 0 else None
                
                outputs = model(cat_features, cont_features)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = correct / total
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Fold {fold_num}, Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}')
            print(f'  Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        scheduler.step()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Fold {fold_num} completed - Best Val Acc: {best_val_acc:.4f}")
    
    return model, best_val_acc, train_losses, val_accuracies

