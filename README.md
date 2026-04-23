# Robust Prediction of Drug Interactions using Chemical Descriptors

## Environment

- Python 3.12.0 (model training)
- Python 2.7.18 (feature extraction with PyBioMed + RDKit 2017.09.3)
- NVIDIA RTX 4090 (24 GB VRAM)

## Dataset

Pre-processed DDI2025 dataset: https://doi.org/10.5281/zenodo.17923583

## Reproducing paper results

```bash
python arch/general.py \
    --train_path material/train_extracted.csv \
    --valid_path material/validation_extracted.csv \
    --test_path material/test_extracted.csv \
    --feature_list_path material/list_of_all_features_ascending_order.txt \
    --num_features_to_drop 6 \
    --n_folds 3 \
    --num_epochs 200 \
    --patience 50 \
    --learning_rate 9.4526e-5 \
    --weight_decay 1.5446e-4 \
    --batch_size 256 \
    --focal_gamma 1.0 \
    --seed 42 \
    --output_dir results
```

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12.0](https://img.shields.io/badge/python-3.12.0-blue.svg)](https://www.python.org/)
[![Web App](https://img.shields.io/badge/Web_App-Live-green)](https://projectxddi-tddi-docker.hf.space/)

A descriptor-based deep learning framework for multi-class drug-drug interaction (DDI) prediction with uncertainty-aware estimation. T-DDI uses explicit physicochemical descriptors from molecular SMILES and an ensemble-based uncertainty estimator to handle severe class imbalance across 178 DDI types.

---

## Key Results (DDI2025 test set)

| Setting | Accuracy | Macro F1 |
|---|---|---|
| T-DDI (full test set) | 0.9434 | 0.8452 |
| T-DDI with UE (high-confidence subset, threshold = 0.88, 87.91% of samples) | 0.9796 | 0.8992 |

The DDI2025 high-confidence threshold is `0.88`, selected on three-fold out-of-fold (OOF) development predictions as the smallest confidence cutoff reaching at least 95% OOF accuracy. The lower confidence boundary is fixed at `0.50`. The frozen threshold artifact and OOF sweep are provided under `reproducibility/ddi2025/`. In the DDI2018 benchmark analysis, the same OOF selection rule selected a dataset-specific threshold of `0.90`.

---

## Installation

```bash
git clone https://github.com/HienKha/tddi.git
cd tddi
pip install -r requirements.txt
```

**Requirements:** Python 3.12.0 for model training, with a CUDA-capable GPU recommended.
See `requirements.txt` for the full dependency list. The live web application is deployed separately and does not need to be installed to reproduce the manuscript tables.

> **Note on descriptor computation:** The 3,780 QSAR descriptors were computed using PyBioMed (PyInteraction module) with RDKit 2017.09.3 under Python 2.7.18. This step is separate from model training under Python 3.12.0. Pre-computed descriptors are available via the dataset link below. The vendored `PyBioMed/` directory is retained for provenance; model training starts from the pre-computed descriptor CSV files.

---

## Dataset

The DDI2025 dataset (868,069 drug pairs, 178 interaction types, 3,780 QSAR descriptors) is publicly available on Zenodo:

**https://doi.org/10.5281/zenodo.17923583**

Download and extract the dataset:

```bash
unzip data_splits.zip
```

The archive contains the following files under `data_splits/`:

| File | Size | Pairs |
|---|---|---|
| `train_extracted.csv` | 10.1 GB | 520,841 (60%) |
| `validation_extracted.csv` | 3.4 GB | 173,614 (20%) |
| `test_extracted.csv` | 3.4 GB | 173,614 (20%) |

Move the files into `material/` before training:

```bash
mv data_splits/train_extracted.csv material/train_extracted.csv
mv data_splits/validation_extracted.csv material/validation_extracted.csv
mv data_splits/test_extracted.csv material/test_extracted.csv
```

### CSV format

**Drug identity columns (6 columns):**

| Column | Description |
|---|---|
| `drugid-drug_a` | DrugBank ID of the first drug |
| `drugid-drug_b` | DrugBank ID of the second drug |
| `drugname-drug_a` | Generic name of the first drug |
| `drugname-drug_b` | Generic name of the second drug |
| `drugsmiles-drug_a` | Canonical SMILES string of the first drug |
| `drugsmiles-drug_b` | Canonical SMILES string of the second drug |

**QSAR descriptor columns (3,780 columns):**

Computed per drug pair using PyBioMed (PyInteraction module) and RDKit 2017.09.3. The descriptors span seven base families plus their pairwise cross-family interaction terms:

| Family | Full name | Properties captured |
|---|---|---|
| `MR_VSA` | Molar Refractivity van der Waals Surface Area | Steric bulk, van der Waals volume |
| `EState_VSA` | E-state van der Waals Surface Area | Electronic topology |
| `SlogP_VSA` | Octanol/water partition coefficient-based VSA | Lipophilicity |
| `LabuteASA` | Labute Atomic Surface Area | Solvent-accessible surface area |
| `MTPSA` | Molecular Topological Polar Surface Area | Polarity, membrane permeability |
| `PEOE_VSA` | Partial Equalization of Orbital Electronegativity VSA | Ionization, partial atomic charges |
| `VSA_EState` | van der Waals Surface Area-based E-state | Electronic-spatial combined |

Cross-family interaction terms (pairwise products of descriptors from distinct families, e.g., `SlogP_VSA × PEOE_VSA`) are also included and account for the majority of the 3,780 total features.

**Target column (1 column):**

| Column | Description |
|---|---|
| `class` | Integer 0–177 encoding the DDI type |

The files should be placed under `material/`:

```
material/
├── train_extracted.csv
├── validation_extracted.csv
└── test_extracted.csv
```

### Descriptor-generation provenance

The descriptor-generation code is vendored under `PyBioMed/` to preserve the legacy PyBioMed/PyInteraction code path. This is not required for model training if the Zenodo descriptor CSV files are used.

For descriptor recomputation from SMILES, use a separate legacy environment:

```bash
cd PyBioMed
conda env create -f conda-env-27.yml
conda activate py27
python setup.py install
```

The environment pins Python 2.7.18 and RDKit 2017.09.3 to match the descriptor generation setup reported in the manuscript.

---

## Pretrained Model Artifacts

The DDI2025 pretrained model and confidence-threshold artifacts are available on Zenodo:

**https://doi.org/10.5281/zenodo.19588891**

The archive contains the final 3-fold ensemble checkpoint, individual fold checkpoints, the OOF prediction CSV used for threshold selection, held-out test predictions, feature schema, label mapping, and checksum manifest.

After downloading and extracting the model archive, the prediction artifacts used
for threshold reproduction are under `threshold_artifacts/`.

The held-out DDI2025 test set can be evaluated without retraining:

```bash
python arch/evaluate_saved_model.py \
    --model_path path/to/tddi_ddi2025_model_artifacts_2026-04-15/models/full3780_new_submit.pkl \
    --test_path material/test_extracted.csv \
    --feature_list_path material/list_of_all_features_ascending_order.txt \
    --num_features_to_drop 6 \
    --threshold_json reproducibility/ddi2025/selected_thresholds_full3780_new_submit.json \
    --output_dir results/eval_ddi2025
```

---

## Training

To replicate the numerical-only T-DDI configuration, drop the six identity columns (described above) before training. The number of epochs is set to 200 with a patience of 50.

```bash
python arch/general.py \
    --train_path material/train_extracted.csv \
    --valid_path material/validation_extracted.csv \
    --test_path material/test_extracted.csv \
    --feature_list_path material/list_of_all_features_ascending_order.txt \
    --num_features_to_drop 6 \
    --n_folds 3 \
    --num_epochs 200 \
    --patience 50 \
    --output_dir results
```

The training script writes `oof_predictions.csv`, `oof_threshold_sweep.csv`, `selected_thresholds_from_oof.json`, `results.csv`, `confidence_strata_metrics.csv`, and `thresholds_used.json` under the requested output directory. By default, the held-out test set is evaluated using the OOF-selected threshold from the current run. To reuse the frozen DDI2025 manuscript threshold artifact instead, pass `--threshold_json reproducibility/ddi2025/selected_thresholds_full3780_new_submit.json`.

The active manuscript hyperparameters for the numerical-only configuration are exposed directly via CLI and mirrored in [`configs/paper_default.yaml`](configs/paper_default.yaml). That preset also retains compatibility fields required by the original TabTransformer constructor, but those transformer-specific fields do not affect the numerical-only forward path because no categorical columns are used. You can load the preset and still override any individual value on the command line:

```bash
python arch/general.py \
    --config configs/paper_default.yaml \
    --train_path material/train_extracted.csv \
    --valid_path material/validation_extracted.csv \
    --test_path material/test_extracted.csv \
    --feature_list_path material/list_of_all_features_ascending_order.txt \
    --num_features_to_drop 6 \
    --output_dir results
```

Supported active manuscript hyperparameters: `--learning_rate`, `--weight_decay`, `--batch_size`, `--focal_gamma`, and `--seed`.

### Confidence threshold reproduction

The DDI2025 OOF threshold selection can be recomputed from OOF prediction CSVs with:

```bash
python reproducibility/select_confidence_threshold.py \
    --oof_predictions path/to/tddi_ddi2025_model_artifacts_2026-04-15/threshold_artifacts/oof_predictions_full3780_new_submit.csv \
    --test_predictions path/to/tddi_ddi2025_model_artifacts_2026-04-15/threshold_artifacts/test_predictions_full3780_new_submit_threshold.csv \
    --out_dir reproducibility/ddi2025
```

The script selects the smallest threshold in `[0.50, 0.99]` that reaches at least 95% OOF accuracy, then applies the frozen threshold to the held-out test predictions. The expected selected threshold is `t_high = 0.88`.

---

## Project Structure

```
tddi/
├── arch/
│   ├── general.py          # Main training and evaluation pipeline
│   ├── evaluate_saved_model.py  # Evaluation-only script for pretrained checkpoints
│   ├── models.py           # Model architecture and FocalLoss
│   ├── preprocessing.py    # Data loading and label encoding
│   ├── training.py         # Fold-level training loop
│   └── utils.py            # Uncertainty quantification utilities
├── material/               # Place dataset CSV files here
├── PyBioMed/               # Vendored legacy descriptor code provenance
├── reproducibility/        # Threshold artifacts and OOF threshold script
├── lime_explanations_300dpi/  # Example LIME output figures
├── .env.example            # Environment variable template
├── requirements.txt
└── README.md
```

---

## Using the Web Application

The web application is a deployed demo built around the same model artifacts. No installation needed:

**https://projectxddi-tddi-docker.hf.space/**

### Single Pair Prediction
1. Go to the **Single Pair** tab
2. Enter the names of two drugs (e.g., *Metformin* and *Cimetidine*)
3. Optionally enable **LIME Explanation** to see which molecular features drive the prediction
4. Optionally enable **AI Interpretation (Gemini)** for a natural language summary
5. Click **Analyze Interaction**

The output includes the predicted DDI type, confidence score, confidence tier, LIME feature importance plot (if enabled), and natural language explanation (if enabled).

Average processing time: ~0.015ms–1s per pair (model only), ~7–8s with LIME, ~24s with AI interpretation.

### Multiple Pair Prediction
1. Go to the **Multiple Pair** tab
2. Enter multiple drug names, one per row
3. Click **Analyze All Pairs** to screen all combinations simultaneously

### Confidence Tiers

| Tier | Confidence | Interpretation |
|---|---|---|
| High | ≥ 0.88 | Suitable for automated flagging |
| Medium | 0.50–0.88 | Should undergo expert review |
| Low | < 0.50 | Warrants additional investigation |

### Accessibility
The interface includes an adaptive color palette (orange-yellow-blue) optimized for users with color vision deficiencies.

---

## Model Architecture

T-DDI is a TabTransformer-inspired architecture operating exclusively on continuous numerical QSAR descriptors (no categorical inputs):

- **Input:** 3,780 physicochemical descriptors per drug pair (7 descriptor families + cross-family interaction terms)
- **Architecture:** Layer normalization over 3,780 continuous QSAR descriptors followed by the TabTransformer MLP prediction head. The categorical embedding and self-attention branch is inactive in the final numerical-only configuration.
- **Parameters:** 87,448,646 (all trainable)
- **Training:** 3-fold stratified ensemble with unweighted focal loss (`gamma=1.0`)
- **Uncertainty:** Entropy, variance, and mutual information aggregated across ensemble members, normalized to a [0,1] confidence score
- **Hardware:** Trained on NVIDIA RTX 4090 (24 GB VRAM); inference ~0.0151 ms per drug pair on CUDA

---

## Citation

If you use T-DDI in your research, please cite:

```
Kha, Q.-H., Nguyen, D.-Q.-A., Pham, V.-H.-P., Huynh, K.-M.-U., Pham, D.-K., Phung, M.-T., Huynh, T.-P. et al. T-DDI: Robust Prediction of Drug
Interactions using Chemical Descriptors. npj Digital Medicine (under review).
```

---

## Acknowledgements

Part of the architecture was adapted from TabTransformer:  
https://github.com/lucidrains/tab-transformer-pytorch
