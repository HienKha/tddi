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
| T-DDI with UE (high-confidence subset, 95.62% of samples) | 0.9661 | 0.8843 |

---

## Installation

```bash
git clone https://github.com/HienKha/tddi.git
cd tddi
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, CUDA-capable GPU recommended.
See `requirements.txt` for the full dependency list.

> **Note on descriptor computation:** The 3,780 QSAR descriptors used in this study were computed using PyBioMed (PyInteraction module) with RDKit 2017.09.3 under Python 2.7.18. This step is separate from model training (which uses Python 3.8+). Pre-computed descriptors are available via the dataset link below.

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

**Target column (1 column):**

| Column | Description |
|---|---|
| `label` | Integer 0–177 encoding the DDI type |

The files should be placed under `material/`:

```
material/
├── train_extracted.csv
├── validation_extracted.csv
└── test_extracted.csv
```

---

## Training

```bash
python arch/general.py \
    --train_path material/train_extracted.csv \
    --valid_path material/validation_extracted.csv \
    --test_path material/test_extracted.csv \
    --n_folds 3 \
    --num_epochs 300 \
    --patience 120 \
    --output_dir results
```

To train with feature selection (dropping low-ranked features):

```bash
python arch/general.py \
    --train_path material/train_extracted.csv \
    --valid_path material/validation_extracted.csv \
    --test_path material/test_extracted.csv \
    --feature_list_path material/list_of_all_features_ascending_order.txt \
    --num_features_to_drop 6 \
    --output_dir results
```

### CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--train_path` | ✅ | — | Path to training CSV |
| `--valid_path` | ✅ | — | Path to validation CSV |
| `--test_path` | ✅ | — | Path to test CSV |
| `--feature_list_path` | ❌ | None | Text file of feature names (one per line) |
| `--num_features_to_drop` | ❌ | 0 | Number of features to drop from the list |
| `--n_folds` | ❌ | 3 | Number of stratified CV folds |
| `--num_epochs` | ❌ | 300 | Max training epochs per fold |
| `--patience` | ❌ | 120 | Early stopping patience |
| `--output_dir` | ❌ | results | Output directory for results |

---

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
├── PyBioMed/               # Descriptor computation (Python 2.7)
├── lime_explanations_300dpi/  # Example LIME output figures
├── .env.example            # Environment variable template
├── requirements.txt
└── README.md
```

---

## Using the Web Application

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
- Confidence tier: High (≥0.86), Medium (0.50–0.86), or Low (<0.50)
- LIME feature importance plot (if enabled)
- Natural language explanation (if enabled)

Average processing time is approximately 7–8 seconds per pair for model prediction with LIME explanation, and approximately 24 seconds per pair when the AI interpretation module is enabled.

### Multiple Pair Prediction
1. Go to the **Multiple Pair** tab
2. Enter multiple drug names, one per row
3. Click **Analyze All Pairs** to screen all combinations simultaneously

### Confidence Tiers

| Tier | Confidence | Interpretation |
|---|---|---|
| High | ≥ 0.86 | Suitable for automated flagging |
| Medium | 0.50–0.86 | Should undergo expert review |
| Low | < 0.50 | Warrants additional investigation |

### Accessibility
The interface includes an adaptive color palette (orange-yellow-blue) optimized for users with color vision deficiencies.

---

## Model Architecture

T-DDI is a TabTransformer-inspired architecture operating exclusively on continuous numerical QSAR descriptors (no categorical inputs):

- **Input:** 3,780 physicochemical descriptors per drug pair spanning 7 descriptor families (MR_VSA, EState_VSA, SlogP_VSA, LabuteASA, MTPSA, PEOE_VSA, VSA_EState) and their cross-family interaction terms
- **Architecture:** 3-layer Transformer-style encoder, 16 attention heads, 64-dimensional feature embedding
- **Parameters:** 87,448,646 (all trainable)
- **Training:** 3-fold stratified ensemble with focal loss to handle class imbalance
- **Uncertainty:** Entropy, variance, and mutual information aggregated across ensemble members, normalized to a [0,1] confidence score
- **Hardware:** Trained on NVIDIA RTX 4090 (24 GB VRAM); inference ~0.0151 ms per drug pair

---

## Citation

If you use T-DDI in your research, please cite:

```
Kha, Q.-H., Nguyen, D.-Q.-A., et al. T-DDI: Robust Prediction of Drug
Interactions using Chemical Descriptors. npj Digital Medicine (under review).
```

---

## Acknowledgements

Part of the architecture was adapted from TabTransformer:  
https://github.com/lucidrains/tab-transformer-pytorch
