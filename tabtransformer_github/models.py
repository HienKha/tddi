"""
Model classes for TabTransformer including loss functions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer
from .utils import UncertaintyEstimator


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedTabTransformerWithImprovements:   
    """Enhanced TabTransformer with uncertainty estimation and ensemble support."""
    
    def __init__(self, categories: List[int], num_continuous: int, 
                 num_classes: int, device: torch.device, 
                 best_params: Dict):
        self.categories = categories
        self.num_continuous = num_continuous
        self.num_classes = num_classes
        self.device = device
        self.best_params = best_params
        self.models = []
        self.uncertainty_estimator = UncertaintyEstimator()
        self.interpreter = None
        
    def create_model(self) -> TabTransformer:
        """Create a new TabTransformer model instance."""
        model = TabTransformer(
            categories=tuple(self.categories),
            num_continuous=self.num_continuous,
            dim=self.best_params['dim'],
            dim_out=self.num_classes,
            depth=self.best_params['depth'],
            heads=self.best_params['heads'],
            attn_dropout=self.best_params['attn_dropout'],
            ff_dropout=self.best_params['ff_dropout'],
            mlp_hidden_mults=(
                self.best_params['mlp_hidden_mult_1'], 
                self.best_params['mlp_hidden_mult_2']
            ),
            mlp_act=nn.ReLU()
        ).to(self.device)
        return model
    
    def predict_with_uncertainty(self, test_df: pd.DataFrame) -> Dict:
        """
        Make predictions with uncertainty estimation using ensemble models.
        
        Args:
            test_df: DataFrame with test data (may include 'class' column)
        
        Returns:
            Dictionary with predictions, probabilities, and uncertainties
        """
        test_df_enhanced = test_df.copy()
        
        X_test = test_df_enhanced.drop(columns=['class'], errors='ignore')
        test_tensor = torch.FloatTensor(X_test.values)
        test_dataset = TensorDataset(test_tensor, torch.zeros(len(test_tensor)))
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        
        for model in self.models:
            model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for batch_x, _ in test_loader:
                    batch_x = batch_x.to(self.device)
                    cat_features = batch_x[:, :len(self.categories)].long()
                    cont_features = batch_x[:, len(self.categories):] if self.num_continuous > 0 else None
                    
                    outputs = model(cat_features, cont_features)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())
            
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        ensemble_probs = np.mean(all_probabilities, axis=0)
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        
        uncertainties = self.uncertainty_estimator.estimate_uncertainty(
            all_predictions, all_probabilities
        )
        
        return {
            'predictions': ensemble_predictions,
            'probabilities': ensemble_probs,
            'uncertainties': uncertainties,
            'individual_predictions': all_predictions,
            'individual_probabilities': all_probabilities
        }
    
    def get_feature_importance(self, test_loader: DataLoader) -> Dict[str, float]:
        """Get feature importance using the interpreter."""
        if self.interpreter is None:
            raise ValueError("Interpreter not initialized. Set self.interpreter first.")
        return self.interpreter.permutation_importance(test_loader)

