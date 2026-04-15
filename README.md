# T-DDI: Robust Prediction of Drug Interactions using Chemical Descriptors

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Web App](https://img.shields.io/badge/Web_App-Live-green)](https://projectxddi-tddi-docker.hf.space/)

A descriptor-based deep learning framework for multi-class drug-drug interaction (DDI) prediction with uncertainty-aware estimation. T-DDI uses explicit physicochemical descriptors from molecular SMILES and an ensemble-based uncertainty estimator to handle severe class imbalance across 178 DDI types.

**Live demo:** https://projectxddi-tddi-docker.hf.space/

---

## Key Results (DDI2025 test set)

| Setting | Accuracy | Macro F1 |
|---|---|---|
| T-DDI (full test set) | 0.9434 | 0.8452 |
| T-DDI with UE (high-confidence subset, threshold = 0.88, 87.91% of samples) | 0.9796 | 0.8992 |

The DDI2025 high-confidence threshold is `0.88`. It was selected on three-fold out-of-fold development predictions as the smallest confidence cutoff reaching at least 95% OOF accuracy. The lower confidence boundary is fixed at `0.50`. The frozen threshold artifact and OOF sweep are provided under `reproducibility/ddi2025/`. In the DDI2018 benchmark analysis, the same OOF selection rule selected a dataset-specific high-confidence threshold of `0.90`.

---

## Installation

```bash
git clone https://github.com/HienKha/tddi.git
cd tddi
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, CUDA-capable GPU recommended.
See `requirements.txt` for the training and reproducibility dependency list.
The live web application is deployed separately and does not need to be
installed to reproduce the manuscript tables.

> **Note on descriptor computation:** The 3,780 QSAR descriptors used in this study were computed using PyBioMed (PyInteraction module) with RDKit 2017.09.3 under Python 2.7.18. This step is separate from model training (which uses Python 3.8+). Pre-computed descriptors are available via the dataset link below. The vendored `PyBioMed/` directory is retained for provenance of the legacy descriptor-generation environment; model training starts from the pre-computed descriptor CSV files.

---

## Dataset

The DDI2025 dataset (868,069 drug pairs, 178 interaction types, 3,780 QSAR descriptors) is publicly available on Zenodo:

**https://doi.org/10.5281/zenodo.17923583**

The DDI2025 pretrained model and confidence-threshold artifacts are available on Zenodo:

**https://doi.org/10.5281/zenodo.19588891**

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

Each file contains the following columns:

**Drug identity columns (6 columns):**

| Column | Description |
|---|---|
| `drugid_drug_a` | DrugBank ID of the first drug |
| `drugid_drug_b` | DrugBank ID of the second drug |
| `drugname_drug_a` | Generic name of the first drug |
| `drugname_drug_b` | Generic name of the second drug |
| `drugsmiles_drug_a` | Canonical SMILES string of the first drug |
| `drugsmiles_drug_b` | Canonical SMILES string of the second drug |

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

Cross-family interaction terms (pairwise products of descriptors from distinct families, e.g., `SlogP_VSA × PEOE_VSA`) are also included and account for the majority of the 3,780 total features. These cross-terms capture synergistic relationships between polarity and hydrophobicity at the drug-pair level.

### Descriptor-generation provenance

The descriptor-generation code is vendored under `PyBioMed/` to preserve the
legacy PyBioMed/PyInteraction code path used for descriptor provenance. This is
not required for model training if the Zenodo descriptor CSV files are used.

For descriptor recomputation from SMILES, use a separate legacy environment:

```bash
cd PyBioMed
conda env create -f conda-env-27.yml
conda activate py27
python setup.py install
```

The environment pins Python 2.7.18 and RDKit 2017.09.3 to match the descriptor
generation setup reported in the manuscript.

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

---

## Pretrained Model Artifacts

The Zenodo model archive contains the final DDI2025 3-fold ensemble checkpoint,
individual fold checkpoints, the OOF prediction CSV used for threshold
selection, held-out test predictions, feature schema, label mapping, and
checksum manifest:

**https://doi.org/10.5281/zenodo.19588891**

The expected DDI2025 threshold artifact selects `t_high = 0.88` and reproduces
the manuscript high-confidence subset metrics:

| Setting | Coverage | Accuracy | Macro F1 |
|---|---:|---:|---:|
| High-confidence subset | 87.91% | 0.9796 | 0.8992 |

---

## Training
To replicate the numerical-only T-DDI configuration described in the paper and ensure its feasibility for production, we established the number of epochs at 200 and patience at 50, drop the six metadata/identity columns (`drugid-drug_a`, `drugid-drug_b`, `drugname-drug_a`, `drugname-drug_b`, `drugsmiles-drug_a`, `drugsmiles-drug_b`) before training.
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

### Confidence threshold reproduction

The DDI2025 OOF threshold selection can be recomputed from OOF prediction CSVs with:

```bash
python reproducibility/select_confidence_threshold.py \
    --oof_predictions best_model_and_results/full3780_new_submit_oof_threshold/oof_predictions_full3780_new_submit.csv \
    --test_predictions best_model_and_results/full3780_new_submit_oof_threshold/test_predictions_full3780_new_submit_threshold.csv \
    --out_dir reproducibility/ddi2025
```

The script selects the smallest threshold in `[0.50, 0.99]` that reaches at least 95% OOF accuracy, then applies the frozen threshold to the held-out test predictions.


## Project Structure

```
tddi/
├── arch/
│   ├── general.py          # Main training and evaluation pipeline
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

This public repository contains the training, evaluation, and reproducibility
code for the manuscript. The web application is a deployed demo built around
the same model artifacts. This web application allows users to predict drug-drug interactions (DDI) between two medications for analytical purposes.

No installation needed. The web app is available at:  
**https://projectxddi-tddi-docker.hf.space/**

### Single Pair Prediction
1. Go to the **Single Pair** tab
2. Enter the names of two drugs (e.g., *Metformin* and *Cimetidine*)
3. Optionally enable **LIME Explanation** to see which molecular features drive the prediction
4. Optionally enable **AI Interpretation (Gemini)** for a natural language summary of the predicted interaction
5. Click **Analyze Interaction**

The output includes:
- Predicted DDI type and confidence score
- Confidence tier: High (≥0.88), Medium (0.50–0.88), or Low (<0.50)
- LIME feature importance plot (if enabled)
- Natural language explanation (if enabled)

Average processing time is approximately 0.015ms to 1 second per pair (depending on using cuda/cpu or network stability) for model prediction without LIME/AI interpretation. Average processing time is approximately 7–8 seconds per pair for model prediction with LIME explanation, and approximately 24 seconds per pair when the AI interpretation module is enabled.

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

- **Input:** 3,780 physicochemical descriptors per drug pair spanning 7 descriptor families (MR_VSA, EState_VSA, SlogP_VSA, LabuteASA, MTPSA, PEOE_VSA, VSA_EState) and their cross-family interaction terms
- **Architecture:** TabTransformer-inspired numerical branch: layer normalization over 3,780 continuous QSAR descriptors followed by the TabTransformer MLP prediction head. The categorical embedding and self-attention branch is inactive in the final numerical-only T-DDI configuration. We exclusively employ the numerical branch for inference due to the large parameters in the ensemble model.
- **Parameters:** 87,448,646 (all trainable)
- **Training:** 3-fold stratified ensemble with unweighted focal loss (`gamma=1.0`) to emphasize hard examples under class imbalance
- **Uncertainty:** Entropy, variance, and mutual information aggregated across ensemble members, normalized to a [0,1] confidence score
- **Hardware:** Trained on NVIDIA RTX 4090 (24 GB VRAM); inference ~0.0151 ms per drug pair (on cuda)

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
