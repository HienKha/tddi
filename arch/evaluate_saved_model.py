#!/usr/bin/env python3
"""Evaluate a pretrained T-DDI ensemble checkpoint without retraining."""

import argparse
import __main__
import io
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import accuracy_score, classification_report

try:
    from .models import TDDI_Model
    from .utils import UncertaintyEstimator
except ImportError:
    from models import TDDI_Model
    from utils import UncertaintyEstimator


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def load_feature_list(path: Optional[str], n_drop: int) -> list:
    if not path or n_drop <= 0:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()][:n_drop]


def load_thresholds(
    threshold_json: Optional[str], t_high: float, t_low: float
) -> Tuple[float, float, str]:
    if not threshold_json:
        return t_high, t_low, "command-line/default arguments"

    with open(threshold_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    loaded_t_high = payload.get("t_high", payload.get("selected_threshold", t_high))
    loaded_t_low = payload.get("t_low", t_low)
    return float(loaded_t_high), float(loaded_t_low), threshold_json


def patch_notebook_pickle_classes() -> None:
    """Register notebook class names used by the archived pickle."""
    __main__.UncertaintyEstimator = UncertaintyEstimator
    __main__.EnhancedTabTransformerWithImprovements = TDDI_Model


def load_pickled_model(model_path: str, device: torch.device) -> TDDI_Model:
    """Load a notebook-created pickle and move fold models to the target device."""
    patch_notebook_pickle_classes()

    original_load_from_bytes = None
    if device.type == "cpu" and hasattr(torch.storage, "_load_from_bytes"):
        original_load_from_bytes = torch.storage._load_from_bytes

        def cpu_load_from_bytes(binary_storage):
            try:
                return torch.load(
                    io.BytesIO(binary_storage),
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                return torch.load(io.BytesIO(binary_storage), map_location="cpu")

        torch.storage._load_from_bytes = cpu_load_from_bytes

    try:
        with open(model_path, "rb") as handle:
            model = pickle.load(handle)
    finally:
        if original_load_from_bytes is not None:
            torch.storage._load_from_bytes = original_load_from_bytes

    model.device = device
    for fold_model in model.models:
        fold_model.to(device)
        fold_model.eval()
    return model


def load_evaluation_frame(
    csv_path: str, feature_list_path: Optional[str], n_drop: int
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    columns_to_drop = load_feature_list(feature_list_path, n_drop)
    frame = pl.read_csv(csv_path)
    existing_drop_cols = [col for col in columns_to_drop if col in frame.columns]
    if existing_drop_cols:
        frame = frame.drop(existing_drop_cols)

    df = frame.to_pandas()
    y_true = None
    if "class" in df.columns:
        y_true = df["class"].to_numpy()
        x_df = df.drop(columns=["class"])
    else:
        x_df = df

    non_numeric = x_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Non-numeric columns remain after preprocessing. "
            f"Check --feature_list_path/--num_features_to_drop. Columns: {non_numeric[:10]}"
        )

    return x_df, y_true


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if y_true is None or len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "weighted_precision": None,
            "weighted_recall": None,
            "weighted_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
        }

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy),
        "micro_precision": float(accuracy),
        "micro_recall": float(accuracy),
        "micro_f1": float(accuracy),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }


def stratum_metrics(
    group: str,
    mask: np.ndarray,
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    n_total = len(y_pred)
    n_group = int(mask.sum())
    row = {
        "group": group,
        "n": n_group,
        "coverage": float(n_group / n_total) if n_total else 0.0,
        "true_class_count": int(len(np.unique(y_true[mask]))) if y_true is not None and n_group else 0,
    }
    if y_true is None or n_group == 0:
        row.update(classification_metrics(np.array([]), np.array([])))
        row["n"] = n_group
        return row
    row.update(classification_metrics(y_true[mask], y_pred[mask]))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained T-DDI checkpoint on descriptor CSV input."
    )
    parser.add_argument("--model_path", required=True, help="Path to full3780_new_submit.pkl")
    parser.add_argument("--test_path", required=True, help="CSV with descriptor columns and optional class")
    parser.add_argument("--feature_list_path", default=None, help="Feature/drop-list file")
    parser.add_argument("--num_features_to_drop", type=int, default=0)
    parser.add_argument("--threshold_json", default=None, help="JSON with frozen t_high/t_low")
    parser.add_argument("--t_high", type=float, default=0.88)
    parser.add_argument("--t_low", type=float, default=0.50)
    parser.add_argument("--output_dir", default="eval_outputs")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_high, t_low, threshold_source = load_thresholds(
        args.threshold_json, args.t_high, args.t_low
    )
    print(f"Loading model: {args.model_path}")
    model = load_pickled_model(args.model_path, device)

    print(f"Loading evaluation data: {args.test_path}")
    x_df, y_true = load_evaluation_frame(
        args.test_path, args.feature_list_path, args.num_features_to_drop
    )
    print(f"Evaluation feature shape: {x_df.shape}")

    prediction_input = x_df.copy()
    if y_true is not None:
        prediction_input["class"] = y_true
    results = model.predict_with_uncertainty(prediction_input)

    y_pred = np.asarray(results["predictions"])
    probabilities = np.asarray(results["probabilities"])
    confidence = np.asarray(results["uncertainties"]["confidence"])
    max_probability = probabilities.max(axis=1)

    high_mask = confidence >= t_high
    medium_mask = (confidence >= t_low) & (confidence < t_high)
    low_mask = confidence < t_low

    metrics_payload = {
        "model_path": args.model_path,
        "test_path": args.test_path,
        "threshold_source": threshold_source,
        "t_high": t_high,
        "t_low": t_low,
        "full_test_metrics": classification_metrics(y_true, y_pred),
        "confidence_strata": [
            stratum_metrics("high", high_mask, y_true, y_pred),
            stratum_metrics("medium", medium_mask, y_true, y_pred),
            stratum_metrics("low", low_mask, y_true, y_pred),
        ],
        "class_counts_note": "Class counts are not additive; one class can appear in multiple strata.",
    }

    predictions_df = pd.DataFrame(
        {
            "row_index": np.arange(len(y_pred)),
            "predicted_class": y_pred,
            "confidence": confidence,
            "max_probability": max_probability,
            "confidence_group": np.where(
                high_mask, "high", np.where(medium_mask, "medium", "low")
            ),
        }
    )
    if y_true is not None:
        predictions_df.insert(1, "true_class", y_true)

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    strata_path = output_dir / "confidence_strata_metrics.csv"

    predictions_df.to_csv(predictions_path, index=False)
    pd.DataFrame(metrics_payload["confidence_strata"]).to_csv(strata_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metrics_payload), handle, indent=2)

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confidence strata: {strata_path}")


if __name__ == "__main__":
    main()
