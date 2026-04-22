import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
import os
import json
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import MemoryOptimizer, UncertaintyEstimator
from models import TDDI_Model, FocalLoss
from preprocessing import load_and_clean_data, preprocess_ultra_fast
from training import train_single_model

warnings.filterwarnings('ignore')

PAPER_DEFAULTS = {
    "learning_rate": 9.4526e-5,
    "weight_decay": 1.5446e-4,
    "batch_size": 256,
    "focal_gamma": 1.0,
    "hidden_dim": 64,
    "depth": 3,
    "heads": 16,
    "attn_dropout": 0.4,
    "ff_dropout": 0.2,
    "seed": 42,
}


def set_random_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_random_seed(PAPER_DEFAULTS["seed"])

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


def load_thresholds(
    threshold_json: Optional[str],
    t_high: float,
    t_low: float
) -> Tuple[float, float, str]:
    """Load frozen confidence thresholds, if a JSON artifact is provided."""
    if not threshold_json:
        return t_high, t_low, "command-line/default arguments"

    with open(threshold_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    loaded_t_high = payload.get("t_high", payload.get("selected_threshold", t_high))
    loaded_t_low = payload.get("t_low", t_low)
    return float(loaded_t_high), float(loaded_t_low), threshold_json


def load_config_defaults(config_path: Optional[str]) -> Dict[str, object]:
    """Load YAML defaults for the training CLI."""
    if not config_path:
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required when using --config. Install dependencies from requirements.txt."
        ) from exc

    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")

    return payload


def build_best_params(
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    focal_gamma: float,
    hidden_dim: int,
    depth: int,
    heads: int,
    attn_dropout: float,
    ff_dropout: float,
) -> Dict[str, float]:
    """Translate user-facing CLI args into the model/training parameter dictionary."""
    return {
        "dim": hidden_dim,
        "depth": depth,
        "heads": heads,
        "attn_dropout": attn_dropout,
        "ff_dropout": ff_dropout,
        "mlp_hidden_mult_1": 2,
        "mlp_hidden_mult_2": 2,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "gamma": focal_gamma,
    }


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return the metric set used in the manuscript tables."""
    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "micro_precision": np.nan,
            "micro_recall": np.nan,
            "micro_f1": np.nan,
            "weighted_precision": np.nan,
            "weighted_recall": np.nan,
            "weighted_f1": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
        }

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "micro_precision": accuracy,
        "micro_recall": accuracy,
        "micro_f1": accuracy,
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
    }


def confidence_stratum_metrics(name: str, mask, y_true, y_pred) -> Dict[str, float]:
    """Compute metrics for one confidence stratum."""
    n_total = len(y_true)
    n_subset = int(np.sum(mask))
    result = {
        "group": name,
        "n": n_subset,
        "coverage": n_subset / n_total if n_total else 0.0,
        "true_class_count": int(len(np.unique(y_true[mask]))) if n_subset else 0,
    }
    if n_subset == 0:
        result.update({
            "accuracy": np.nan,
            "micro_precision": np.nan,
            "micro_recall": np.nan,
            "micro_f1": np.nan,
            "weighted_precision": np.nan,
            "weighted_recall": np.nan,
            "weighted_f1": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
        })
        return result

    result.update(classification_metrics(y_true[mask], y_pred[mask]))
    return result


def json_safe(obj):
    """Convert numpy/pandas scalar values to JSON-serializable Python values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def dataframe_to_float32_array(X_df) -> np.ndarray:
    """Convert a feature dataframe/array to float32 without changing row order."""
    if isinstance(X_df, pd.DataFrame):
        return X_df.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(X_df, dtype=np.float32)


def predict_single_model_with_confidence(
    model,
    X_df,
    y_true,
    categories: List[int],
    num_continuous: int,
    device: torch.device,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Predict one fold's validation split for OOF threshold tuning."""
    X_np = dataframe_to_float32_array(X_df)
    y_np = np.asarray(y_true)
    dataset = TensorDataset(torch.from_numpy(X_np), torch.LongTensor(y_np))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    pred_list = []
    pred_prob_max_list = []
    entropy_list = []
    conf_list = []
    true_list = []
    cat_len = len(categories)

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            cat_features = batch_x[:, :cat_len].long()
            cont_features = batch_x[:, cat_len:] if num_continuous > 0 else None

            outputs = model(cat_features, cont_features)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            probs_clip = np.clip(probs, 1e-8, 1 - 1e-8)
            entropy = -np.sum(probs_clip * np.log(probs_clip), axis=1)
            confidence = 1.0 - entropy / np.log(probs.shape[1])

            pred_list.append(np.argmax(probs, axis=1))
            pred_prob_max_list.append(probs.max(axis=1))
            entropy_list.append(entropy)
            conf_list.append(confidence)
            true_list.append(batch_y.numpy())

    return {
        "pred_class": np.concatenate(pred_list),
        "pred_prob_max": np.concatenate(pred_prob_max_list),
        "entropy": np.concatenate(entropy_list),
        "confidence": np.concatenate(conf_list),
        "true_class": np.concatenate(true_list),
    }


def build_oof_predictions(
    X_combined,
    y_combined,
    cv_splits,
    fold_models,
    categories: List[int],
    num_continuous: int,
    device: torch.device,
    batch_size: int
) -> pd.DataFrame:
    """Generate leak-free OOF predictions, one held-out fold at a time."""
    y_combined_np = np.asarray(y_combined)
    oof_frames = []

    for fold_idx, ((_, val_idx), model) in enumerate(zip(cv_splits, fold_models), start=1):
        print(f"Running OOF inference for fold {fold_idx}/{len(cv_splits)} ({len(val_idx):,} rows)")
        X_fold_val = X_combined.iloc[val_idx] if isinstance(X_combined, pd.DataFrame) else X_combined[val_idx]
        y_fold_val = y_combined_np[val_idx]

        fold_res = predict_single_model_with_confidence(
            model=model,
            X_df=X_fold_val,
            y_true=y_fold_val,
            categories=categories,
            num_continuous=num_continuous,
            device=device,
            batch_size=batch_size,
        )

        fold_df = pd.DataFrame({
            "global_idx": val_idx,
            "fold": fold_idx,
            "true_class": fold_res["true_class"].astype(int),
            "pred_class": fold_res["pred_class"].astype(int),
            "pred_prob_max": fold_res["pred_prob_max"].astype(float),
            "entropy": fold_res["entropy"].astype(float),
            "confidence": fold_res["confidence"].astype(float),
        })
        fold_df["correct"] = (fold_df["true_class"] == fold_df["pred_class"]).astype(int)
        oof_frames.append(fold_df)
        MemoryOptimizer.cleanup_memory()

    oof_df = (
        pd.concat(oof_frames, axis=0, ignore_index=True)
        .sort_values("global_idx")
        .reset_index(drop=True)
    )

    if len(oof_df) != len(y_combined_np):
        raise RuntimeError("OOF row count does not match development-set size")
    if oof_df["global_idx"].nunique() != len(y_combined_np):
        raise RuntimeError("Each development-set sample must appear exactly once in OOF predictions")

    return oof_df


def select_threshold_from_oof(
    oof_df: pd.DataFrame,
    t_low: float,
    target_oof_accuracy: float
) -> Tuple[pd.DataFrame, Dict]:
    """Sweep OOF confidence thresholds and select the smallest qualifying value."""
    y_oof = oof_df["true_class"].to_numpy()
    p_oof = oof_df["pred_class"].to_numpy()
    c_oof = oof_df["confidence"].to_numpy()

    baseline_metrics = classification_metrics(y_oof, p_oof)
    baseline_metrics["n"] = int(len(y_oof))

    rows = []
    threshold_grid = np.round(np.arange(t_low, 0.991, 0.01), 2)
    for threshold in threshold_grid:
        high_mask = c_oof >= threshold
        metrics = classification_metrics(y_oof[high_mask], p_oof[high_mask])
        rows.append({
            "threshold": float(threshold),
            "n_high": int(high_mask.sum()),
            "coverage": float(high_mask.mean()),
            **metrics,
        })

    sweep_df = pd.DataFrame(rows)
    eligible = sweep_df[sweep_df["accuracy"] >= target_oof_accuracy].sort_values(
        ["threshold", "coverage"],
        ascending=[True, False],
    )
    if eligible.empty:
        raise RuntimeError(
            f"No OOF threshold reached accuracy >= {target_oof_accuracy:.2f}"
        )

    selected_row = eligible.iloc[0].to_dict()
    selected_payload = {
        "target_oof_accuracy": target_oof_accuracy,
        "selection_rule": f"smallest threshold with OOF accuracy >= {target_oof_accuracy:.2f}",
        "t_high": float(selected_row["threshold"]),
        "t_low": float(t_low),
        "baseline_oof_metrics": baseline_metrics,
        "selected_row": selected_row,
        "tier_rule": "high: confidence >= t_high; medium: t_low <= confidence < t_high; low: confidence < t_low",
    }
    return sweep_df, selected_payload


def main(
    train_path: str,
    test_path: str,
    valid_path: str,
    feature_list_path: Optional[str] = None,
    number_of_features_to_drop: int = 0,
    best_params: Optional[Dict] = None,
    threshold_json: Optional[str] = None,
    t_high: float = 0.88,
    t_low: float = 0.50,
    target_oof_accuracy: float = 0.95,
    skip_oof_threshold_selection: bool = False,
    n_folds: int = 3,
    num_epochs: int = 200,
    patience: int = 50,
    output_dir: str = "results",
    learning_rate: float = PAPER_DEFAULTS["learning_rate"],
    weight_decay: float = PAPER_DEFAULTS["weight_decay"],
    batch_size: int = PAPER_DEFAULTS["batch_size"],
    focal_gamma: float = PAPER_DEFAULTS["focal_gamma"],
    hidden_dim: int = PAPER_DEFAULTS["hidden_dim"],
    depth: int = PAPER_DEFAULTS["depth"],
    heads: int = PAPER_DEFAULTS["heads"],
    attn_dropout: float = PAPER_DEFAULTS["attn_dropout"],
    ff_dropout: float = PAPER_DEFAULTS["ff_dropout"],
    seed: int = PAPER_DEFAULTS["seed"],
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
        threshold_json: JSON artifact containing frozen t_high and t_low values
        t_high: High-confidence threshold selected from OOF development predictions
        t_low: Lower confidence boundary fixed a priori
        target_oof_accuracy: OOF accuracy target for threshold selection
        skip_oof_threshold_selection: Skip OOF prediction/sweep generation
        n_folds: Number of cross-validation folds
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        output_dir: Directory to save results
        learning_rate: AdamW learning rate
        weight_decay: AdamW weight decay
        batch_size: Training and inference batch size
        focal_gamma: Gamma value for focal loss
        hidden_dim: TabTransformer embedding dimension
        depth: Transformer depth
        heads: Number of attention heads
        attn_dropout: Attention dropout rate
        ff_dropout: Feed-forward dropout rate
        seed: Random seed for NumPy/PyTorch/CV split reproducibility
    """
    set_random_seed(seed)
    print(f"Random seed: {seed}")

    t_high, t_low, threshold_source = load_thresholds(threshold_json, t_high, t_low)
    print(f"Confidence thresholds: t_high={t_high:.2f}, t_low={t_low:.2f}")
    print(f"Threshold source: {threshold_source}")

    effective_best_params = build_best_params(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        focal_gamma=focal_gamma,
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
    )
    if best_params is not None:
        effective_best_params.update(best_params)
    best_params = effective_best_params
    
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
    enhanced_model = TDDI_Model(
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
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
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

    os.makedirs(output_dir, exist_ok=True)

    if not skip_oof_threshold_selection:
        print("\nGenerating OOF predictions for confidence-threshold selection...")
        oof_df = build_oof_predictions(
            X_combined=X_combined,
            y_combined=y_combined,
            cv_splits=cv_splits,
            fold_models=fold_models,
            categories=categories,
            num_continuous=num_continuous,
            device=device,
            batch_size=best_params["batch_size"],
        )
        oof_predictions_path = os.path.join(output_dir, "oof_predictions.csv")
        oof_df.to_csv(oof_predictions_path, index=False)

        oof_sweep_df, oof_threshold_payload = select_threshold_from_oof(
            oof_df=oof_df,
            t_low=t_low,
            target_oof_accuracy=target_oof_accuracy,
        )
        oof_sweep_path = os.path.join(output_dir, "oof_threshold_sweep.csv")
        selected_thresholds_path = os.path.join(output_dir, "selected_thresholds_from_oof.json")
        oof_sweep_df.to_csv(oof_sweep_path, index=False)
        with open(selected_thresholds_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(oof_threshold_payload), f, indent=2)

        print(f"OOF predictions saved to: {oof_predictions_path}")
        print(f"OOF threshold sweep saved to: {oof_sweep_path}")
        print(f"Selected OOF thresholds saved to: {selected_thresholds_path}")
        print(f"OOF-selected t_high={oof_threshold_payload['t_high']:.2f}")

        if threshold_json is None:
            t_high = float(oof_threshold_payload["t_high"])
            t_low = float(oof_threshold_payload["t_low"])
            threshold_source = selected_thresholds_path
            print(f"Using OOF-selected thresholds for held-out test: t_high={t_high:.2f}, t_low={t_low:.2f}")
        else:
            print("Keeping thresholds loaded from --threshold_json for held-out test evaluation.")
    
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
        
        full_metrics = classification_metrics(y_test, predictions)
        f1_macro = full_metrics['macro_f1']
        f1_weighted = full_metrics['weighted_f1']
        precision_macro = full_metrics['macro_precision']
        precision_weighted = full_metrics['weighted_precision']
        recall_macro = full_metrics['macro_recall']
        recall_weighted = full_metrics['weighted_recall']
        
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        
        # Uncertainty analysis
        print(f"\nUncertainty Statistics:")
        print(f"Mean Entropy: {np.mean(uncertainties['entropy']):.4f}")
        print(f"Mean Confidence: {np.mean(uncertainties['confidence']):.4f}")
        print(f"Mean Variance: {np.mean(uncertainties['variance']):.4f}")
        
        confidence_scores = uncertainties['confidence']
        high_conf_mask = confidence_scores >= t_high
        medium_conf_mask = (confidence_scores >= t_low) & (confidence_scores < t_high)
        low_conf_mask = confidence_scores < t_low
        
        print(f"\nConfidence Distribution:")
        print(f"High confidence (>={t_high:.2f}): {np.sum(high_conf_mask):,} ({np.mean(high_conf_mask)*100:.1f}%)")
        print(f"Medium confidence [{t_low:.2f}, {t_high:.2f}): {np.sum(medium_conf_mask):,} ({np.mean(medium_conf_mask)*100:.1f}%)")
        print(f"Low confidence (<{t_low:.2f}): {np.sum(low_conf_mask):,} ({np.mean(low_conf_mask)*100:.1f}%)")
        
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

        strata_rows = [
            confidence_stratum_metrics("high", high_conf_mask, y_test_np, preds_np),
            confidence_stratum_metrics("medium", medium_conf_mask, y_test_np, preds_np),
            confidence_stratum_metrics("low", low_conf_mask, y_test_np, preds_np),
        ]
        
        # Save results
        result_dict = {
            'num_features_dropped': number_of_features_to_drop,
            'num_classes': num_classes,
            'test_accuracy': test_accuracy,
            'micro_precision': full_metrics['micro_precision'],
            'micro_recall': full_metrics['micro_recall'],
            'micro_f1': full_metrics['micro_f1'],
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
            't_high': t_high,
            't_low': t_low,
            'threshold_source': threshold_source,
        }
        
        results_df = pd.DataFrame([result_dict])
        results_path = os.path.join(output_dir, f"results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        strata_path = os.path.join(output_dir, "confidence_strata_metrics.csv")
        pd.DataFrame(strata_rows).to_csv(strata_path, index=False)
        print(f"Confidence strata metrics saved to: {strata_path}")

        thresholds_path = os.path.join(output_dir, "thresholds_used.json")
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(json_safe({
                "t_high": t_high,
                "t_low": t_low,
                "threshold_source": threshold_source,
                "tier_rule": "high: confidence >= t_high; medium: t_low <= confidence < t_high; low: confidence < t_low",
            }), f, indent=2)
        print(f"Threshold metadata saved to: {thresholds_path}")
    
    print("\nTraining and evaluation completed!")
    return enhanced_model, test_results

if __name__ == "__main__":
    import argparse

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file that provides CLI defaults",
    )
    config_args, _ = config_parser.parse_known_args()
    config_defaults = load_config_defaults(config_args.config)

    parser = argparse.ArgumentParser(description='Train TDDI Model with uncertainty estimation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a YAML config file that provides CLI defaults')
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
    parser.add_argument('--threshold_json', type=str, default=None,
                        help='Path to JSON file with frozen t_high and t_low thresholds')
    parser.add_argument('--t_high', type=float, default=0.88,
                        help='High-confidence threshold selected from OOF development predictions')
    parser.add_argument('--t_low', type=float, default=0.50,
                        help='Lower confidence boundary')
    parser.add_argument('--target_oof_accuracy', type=float, default=0.95,
                        help='OOF accuracy target used to select t_high')
    parser.add_argument('--skip_oof_threshold_selection', action='store_true',
                        help='Skip OOF prediction generation and threshold sweep')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of CV folds')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--learning_rate', type=float, default=PAPER_DEFAULTS['learning_rate'],
                        help='AdamW learning rate')
    parser.add_argument('--weight_decay', type=float, default=PAPER_DEFAULTS['weight_decay'],
                        help='AdamW weight decay')
    parser.add_argument('--batch_size', type=int, default=PAPER_DEFAULTS['batch_size'],
                        help='Training and inference batch size')
    parser.add_argument('--focal_gamma', type=float, default=PAPER_DEFAULTS['focal_gamma'],
                        help='Gamma value for focal loss')
    parser.add_argument('--hidden_dim', type=int, default=PAPER_DEFAULTS['hidden_dim'],
                        help='TabTransformer embedding dimension')
    parser.add_argument('--depth', type=int, default=PAPER_DEFAULTS['depth'],
                        help='Transformer depth')
    parser.add_argument('--heads', type=int, default=PAPER_DEFAULTS['heads'],
                        help='Number of attention heads')
    parser.add_argument('--attn_dropout', type=float, default=PAPER_DEFAULTS['attn_dropout'],
                        help='Attention dropout rate')
    parser.add_argument('--ff_dropout', type=float, default=PAPER_DEFAULTS['ff_dropout'],
                        help='Feed-forward dropout rate')
    parser.add_argument('--seed', type=int, default=PAPER_DEFAULTS['seed'],
                        help='Random seed for NumPy/PyTorch/CV split reproducibility')

    allowed_config_keys = {action.dest for action in parser._actions}
    unknown_config_keys = sorted(set(config_defaults) - allowed_config_keys)
    if unknown_config_keys:
        raise ValueError(
            f"Unsupported config keys in {config_args.config}: {', '.join(unknown_config_keys)}"
        )
    parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    main(
        train_path=args.train_path,
        test_path=args.test_path,
        valid_path=args.valid_path,
        feature_list_path=args.feature_list_path,
        number_of_features_to_drop=args.num_features_to_drop,
        threshold_json=args.threshold_json,
        t_high=args.t_high,
        t_low=args.t_low,
        target_oof_accuracy=args.target_oof_accuracy,
        skip_oof_threshold_selection=args.skip_oof_threshold_selection,
        n_folds=args.n_folds,
        num_epochs=args.num_epochs,
        patience=args.patience,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        focal_gamma=args.focal_gamma,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout,
        seed=args.seed,
    )
