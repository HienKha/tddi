#!/usr/bin/env python3
"""Select confidence thresholds from OOF predictions.

This script reproduces the confidence-threshold rule used for the DDI2025
analysis in the manuscript. It never uses the held-out test set to choose the
threshold. Test predictions can be supplied only after the threshold is frozen
to compute confidence-stratified held-out metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def empty_metrics():
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


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return empty_metrics()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_precision": float(report["accuracy"]),
        "micro_recall": float(report["accuracy"]),
        "micro_f1": float(report["accuracy"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }


def stratum_metrics(group, mask, y_true, y_pred):
    n_total = len(y_true)
    n_group = int(mask.sum())
    row = {
        "group": group,
        "n": n_group,
        "coverage": float(n_group / n_total) if n_total else 0.0,
        "true_class_count": int(len(np.unique(y_true[mask]))) if n_group else 0,
    }
    if n_group == 0:
        row.update(empty_metrics())
        return row

    row.update(compute_metrics(y_true[mask], y_pred[mask]))
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Select OOF confidence threshold and optionally evaluate held-out test strata."
    )
    parser.add_argument("--oof_predictions", required=True, help="CSV with true/pred/confidence columns")
    parser.add_argument("--test_predictions", default=None, help="Optional held-out test prediction CSV")
    parser.add_argument("--out_dir", default="reproducibility/ddi2025", help="Output directory")
    parser.add_argument("--target_oof_accuracy", type=float, default=0.95)
    parser.add_argument("--t_low", type=float, default=0.50)
    parser.add_argument("--true_col", default="true_class")
    parser.add_argument("--pred_col", default="pred_class")
    parser.add_argument("--confidence_col", default="confidence")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.read_csv(args.oof_predictions)
    required = {args.true_col, args.pred_col, args.confidence_col}
    missing = required.difference(oof_df.columns)
    if missing:
        raise ValueError(f"OOF predictions missing required columns: {sorted(missing)}")

    y_oof = oof_df[args.true_col].to_numpy()
    p_oof = oof_df[args.pred_col].to_numpy()
    c_oof = oof_df[args.confidence_col].to_numpy()

    baseline_metrics = compute_metrics(y_oof, p_oof)
    threshold_grid = np.round(np.arange(args.t_low, 0.991, 0.01), 2)

    rows = []
    for threshold in threshold_grid:
        high_mask = c_oof >= threshold
        metrics = compute_metrics(y_oof[high_mask], p_oof[high_mask])
        metrics.pop("n", None)
        rows.append({
            "threshold": float(threshold),
            "n_high": int(high_mask.sum()),
            "coverage": float(high_mask.mean()),
            **metrics,
        })

    sweep_df = pd.DataFrame(rows)
    eligible = sweep_df[sweep_df["accuracy"] >= args.target_oof_accuracy].sort_values(
        ["threshold", "coverage"],
        ascending=[True, False],
    )
    if eligible.empty:
        raise RuntimeError(
            f"No threshold reached target OOF accuracy >= {args.target_oof_accuracy:.2f}"
        )

    selected_row = eligible.iloc[0].to_dict()
    t_high = float(selected_row["threshold"])
    t_low = float(args.t_low)

    selected_payload = {
        "target_oof_accuracy": args.target_oof_accuracy,
        "selection_rule": f"smallest threshold with OOF accuracy >= {args.target_oof_accuracy:.2f}",
        "t_high": t_high,
        "t_low": t_low,
        "baseline_oof_metrics": baseline_metrics,
        "selected_row": selected_row,
        "tier_rule": "high: confidence >= t_high; medium: t_low <= confidence < t_high; low: confidence < t_low",
    }

    sweep_path = out_dir / "oof_threshold_sweep.csv"
    selected_path = out_dir / "selected_thresholds.json"
    sweep_df.to_csv(sweep_path, index=False)
    selected_path.write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    print(f"Selected t_high={t_high:.2f}, t_low={t_low:.2f}")
    print(f"Saved: {sweep_path}")
    print(f"Saved: {selected_path}")

    if args.test_predictions:
        test_df = pd.read_csv(args.test_predictions)
        missing = required.difference(test_df.columns)
        if missing:
            raise ValueError(f"Test predictions missing required columns: {sorted(missing)}")

        y_test = test_df[args.true_col].to_numpy()
        p_test = test_df[args.pred_col].to_numpy()
        c_test = test_df[args.confidence_col].to_numpy()

        high = c_test >= t_high
        medium = (c_test >= t_low) & (c_test < t_high)
        low = c_test < t_low
        test_payload = {
            "threshold_source": str(selected_path),
            "t_high": t_high,
            "t_low": t_low,
            "full_test_metrics": compute_metrics(y_test, p_test),
            "confidence_strata": [
                stratum_metrics("high", high, y_test, p_test),
                stratum_metrics("medium", medium, y_test, p_test),
                stratum_metrics("low", low, y_test, p_test),
            ],
            "class_counts_note": "Class counts are not additive; one class can appear in multiple strata.",
        }

        test_metrics_path = out_dir / "test_metrics_by_confidence.csv"
        final_json_path = out_dir / "final_test_threshold_eval.json"
        pd.DataFrame(test_payload["confidence_strata"]).to_csv(test_metrics_path, index=False)
        final_json_path.write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
        print(f"Saved: {test_metrics_path}")
        print(f"Saved: {final_json_path}")


if __name__ == "__main__":
    main()
