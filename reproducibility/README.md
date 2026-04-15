# Reproducibility Artifacts

This directory contains the lightweight artifacts needed to audit the confidence
thresholds reported in the T-DDI manuscript.

The larger prediction CSVs and pretrained DDI2025 model checkpoints are
archived on Zenodo:

```text
https://doi.org/10.5281/zenodo.19588891
```

## DDI2025 Confidence Threshold

The final DDI2025 high-confidence threshold is `t_high = 0.88`; the lower
confidence boundary is `t_low = 0.50`.

Threshold selection used only three-fold out-of-fold (OOF) predictions from the
development data (`train + validation`). The held-out test set was evaluated
only after the threshold was frozen.

Selection rule:

```text
Sweep candidate t_high values from 0.50 to 0.99 in 0.01 increments.
Select the smallest t_high where OOF accuracy >= 0.95.
```

Confidence tier rule:

```text
High:   confidence >= t_high
Medium: t_low <= confidence < t_high
Low:    confidence < t_low
```

## Files

`ddi2025/selected_thresholds_full3780_new_submit.json`
: Frozen DDI2025 threshold metadata and the selected OOF sweep row.

`ddi2025/oof_threshold_sweep_full3780_new_submit.csv`
: Full OOF threshold sweep used to select `t_high = 0.88`.

`ddi2025/final_test_threshold_eval_full3780_new_submit.json`
: Held-out test metrics after applying the frozen OOF-selected thresholds.

`ddi2025/test_metrics_full3780_new_submit_threshold.csv`
: Held-out test metrics by confidence stratum.

`select_confidence_threshold.py`
: Standalone script for recomputing the threshold sweep from OOF predictions.

## Recompute From Predictions

If the Zenodo model artifact archive has been downloaded and extracted locally:

```bash
python reproducibility/select_confidence_threshold.py \
    --oof_predictions path/to/tddi_ddi2025_model_artifacts_2026-04-15/threshold_artifacts/oof_predictions_full3780_new_submit.csv \
    --test_predictions path/to/tddi_ddi2025_model_artifacts_2026-04-15/threshold_artifacts/test_predictions_full3780_new_submit_threshold.csv \
    --out_dir reproducibility/ddi2025
```

The prediction CSVs are not committed here because they are generated outputs
from the full training and evaluation pipeline. The expected selected threshold
is `t_high = 0.88`.
