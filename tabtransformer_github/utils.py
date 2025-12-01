"""
Utility classes and functions for TabTransformer training and inference.
"""

import gc
import torch
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader


class MemoryOptimizer:        
    @staticmethod
    def cleanup_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage():
        if torch.cuda.is_available():
            return {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        return {'gpu_allocated': 0, 'gpu_reserved': 0, 'gpu_max_allocated': 0}
    
    @staticmethod
    def optimize_dataloader_memory(dataset, batch_size, num_workers=0):
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )


class UncertaintyEstimator:
    """Estimates prediction uncertainty using ensemble methods."""
    
    @staticmethod
    def entropy_uncertainty(probabilities: np.ndarray) -> np.ndarray:
        epsilon = 1e-8
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        entropy_vals = -np.sum(probabilities * np.log(probabilities), axis=1)
        return entropy_vals
    
    @staticmethod
    def variance_uncertainty(ensemble_probs: np.ndarray) -> np.ndarray:
        return np.mean(np.var(ensemble_probs, axis=0), axis=1)
    
    @staticmethod
    def mutual_information(ensemble_probs: np.ndarray) -> np.ndarray:
        mean_probs = np.mean(ensemble_probs, axis=0)
        mean_entropy = UncertaintyEstimator.entropy_uncertainty(mean_probs)
        
        individual_entropies = np.array([
            UncertaintyEstimator.entropy_uncertainty(probs) 
            for probs in ensemble_probs
        ])
        expected_entropy = np.mean(individual_entropies, axis=0)
        
        return mean_entropy - expected_entropy
    
    def estimate_uncertainty(self, ensemble_predictions: np.ndarray, 
                           ensemble_probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty metrics from ensemble predictions.
        
        Args:
            ensemble_predictions: Array of shape (n_models, n_samples)
            ensemble_probabilities: Array of shape (n_models, n_samples, n_classes)
        
        Returns:
            Dictionary with uncertainty metrics
        """
        mean_probs = np.mean(ensemble_probabilities, axis=0)
        
        uncertainties = {
            'entropy': self.entropy_uncertainty(mean_probs),
            'variance': self.variance_uncertainty(ensemble_probabilities),
            'mutual_info': self.mutual_information(ensemble_probabilities),
        }
        
        uncertainties['total'] = uncertainties['entropy'] + uncertainties['variance']
        
        max_entropy = np.log(mean_probs.shape[1])  # log(num_classes)
        uncertainties['confidence'] = 1 - (uncertainties['entropy'] / max_entropy)
        
        return uncertainties

