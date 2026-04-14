# T-DDI: Robust Prediction of Drug Interactions using Chemical Descriptors

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Web App](https://img.shields.io/badge/Web_App-Live-green)](https://projectxddi-tddi-docker.hf.space/)

A descriptor-based deep learning framework for multi-class drug-drug interaction (DDI) prediction with uncertainty-aware estimation. T-DDI uses explicit physicochemical descriptors from molecular SMILES and an ensemble-based uncertainty estimator to handle severe class imbalance across 179 DDI types.

**Live demo:** https://projectxddi-tddi-docker.hf.space/

---

## Key Results

### DDI2025 test set (178 classes, 868,069 drug pairs)

| Setting | Accuracy | Macro F1 |
|---|---|---|
| T-DDI (full test set) | 0.9434 | 0.8452 |
| T-DDI with UE (high-confidence subset, 95.62% of samples, threshold = 0.86) | 0.9661 | 0.8843 |

### DDI2018 benchmark (77 classes, updated DrugBank preprocessing)

T-DDI outperforms all compared DDI-specific baselines on the updated DDI2018 split:

| Model | Accuracy | Macro F1 |
|---|---|---|
| DDI-GCN | 0.7118 | 0.3600 |
| SSI-DDI | 0.8610 | 0.8609 |
| DeepDDI | 0.9228 | 0.7048 |
| **T-DDI without UE** | **0.9296** | **0.8990** |
| **T-DDI with UE** (threshold = 0.90) | **0.9774** | **0.9398** |

> The DDI2018 split uses the same DrugBank preprocessing pipeline as DDI2025, updated to v5.1.12. The original 86 interaction types reduce to 77 after removing multi-adverse-event entries and rare classes.

---

## Prospective Validation on FDA-Approved Drugs (2025)

To assess generalization to unseen compounds, T-DDI was evaluated on five novel small-molecule drugs approved by the FDA between May and November 2025 — **none of which were present in the training set**:

| Drug | Brand name | Approval | Indication |
|---|---|---|---|
| Imlunestrant | Inluriyo | Sep 2025 | ER⁺/HER2⁻ breast cancer (SERD) |
| Remibrutinib | Rhapsido | Sep 2025 | Chronic spontaneous urticaria (BTK inhibitor) |
| Elinzanetant | Lynkuet | Oct 2025 | Menopausal vasomotor symptoms (NK1,3 antagonist) |
| Ziftomenib | Komzifti | Nov 2025 | NPM1-mutated AML (menin-KMT2A inhibitor) |
| Sevabertinib | Hyrnuo | Nov 2025 | Non-small cell lung cancer (HER2 TKI) |

**Representative results:**
- **Imlunestrant + Itraconazole** (CYP3A4 substrate + inhibitor): predicted with *P* > 0.99, consistent with clinical dose-modification guidelines.
- **Elinzanetant + Alprazolam** (NK antagonist + benzodiazepine): predicted with *P* ≈ 1.0, reflecting pharmacodynamic CNS depression synergy.
- **Ziftomenib + Famotidine** (weak base + gastric acid suppressor): predicted with *P* = 0.39, mechanistically consistent with pH-dependent absorption reduction.
- **Remibrutinib + Rifampin**: correctly flagged as interaction but directionality of exposure change was inverted, highlighting a known limitation with strong induction scenarios.

Full predicted DDI tables for all five drugs are provided in Supplementary Document S4.

---

## Error Analysis and Global Explainability (LIME)

### Error analysis

Mispredictions (*N* = 9,830 errors on the DDI2025 test set) are systematic rather than random:
- **87.29%** of errors involve predicting a label from a *different* semantic group (cross-domain leakage).
- **12.71%** are near-misses within the correct semantic group.
- The dominant error pattern is **bidirectional confusion between "Metabolism" and "Serum concentration"** classes, reflecting the model's difficulty in separating an enzymatic mechanism from its downstream outcome.
- A secondary pattern involves PD events (e.g., cardiovascular or CNS depression) being misclassified as PK events.

### Global LIME explanation

To quantify global feature importance, LIME was applied to a **stratified random sample of 9,463 test instances** spanning all 178 interaction classes (up to 100 per class):

- **Cross-family interaction terms** (pairwise products of descriptors from distinct families, e.g., `SlogP_VSA × PEOE_VSA`) account for **77.6%** of top-30 LIME contributions and achieve the highest mean absolute weight across all semantic groups (mean |ŵ| = 0.060).
- The single highest-ranked feature globally is `slogPVSA10 × PEOEVSA3` (mean |ŵ| = 0.276), capturing the interplay between lipophilicity-weighted surface area and partial atomic charges.
- **Pharmacokinetic metabolism groups** rely more heavily on `SlogP_VSA` and `MR_VSA` descriptors, consistent with the role of lipophilicity and van der Waals volume in CYP450 affinity.
- **Seizure-related groups** show the highest `MTPSA` contribution, reflecting the low polar surface area requirement for CNS penetration.
- **Diuretic/renal groups** exhibit elevated `PEOE_VSA` weights, consistent with ionization state importance in renal tubular secretion.

Kruskal–Wallis tests confirmed statistically significant differentiation for Cross-term (*H* = 89.0, *p* < 0.001), MR_VSA (*H* = 27.0, *p* < 0.001), and SlogP_VSA (*H* = 10.0, *p* = 0.019) across broad DDI mechanism categories (PK, PD, clinical-outcome).

Global feature importance heatmaps across 30 semantic DDI groups are provided in Supplementary Document S8.

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
| `class` | Integer 0–177 encoding the DDI type |

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
    --num_epochs 200 \
    --patience 50 \
    --output_dir results
```

To train with feature selection (dropping low-ranked features or at least 6 categorical features [i.e., --num_features_to_drop 6]):

```bash
python arch/general.py \
    --train_path material/train_extracted.csv \
    --valid_path material/validation_extracted.csv \
    --test_path material/test_extracted.csv \
    --feature_list_path material/list_of_all_features_ascending_order.txt \
    --num_features_to_drop 6 \
    --output_dir results
```


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

Average processing time is approximately 0.015ms to 1 second per pair (depending on using cuda/cpu or network stability) for model prediction without LIME/AI interpretation. Average processing time is approximately 7–8 seconds per pair for model prediction with LIME explanation, and approximately 24 seconds per pair when the AI interpretation module is enabled.

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
- **Hardware:** Trained on NVIDIA RTX 4090 (24 GB VRAM); inference ~0.0151 ms per drug pair (on cuda)

---

## Further Reading

If you are interested in the full technical details, complete benchmark tables, ablation studies, and supplementary analyses, please refer to the accompanying manuscript:

- **Main paper** — covers the full methodology, all benchmark comparisons (DDI2025 and DDI2018), prospective FDA-drug validation, error analysis, and global LIME interpretability results.
- **Supplementary document** — includes additional material referenced throughout the paper: DDI type definitions (S1), dataset imbalance statistics (S2), full error breakdown (S3), complete prospective validation tables for all five FDA-approved drugs (S4), Spearman correlation analysis (S5), OOF threshold sensitivity sweep (S6), computational efficiency comparison (S7), global LIME feature importance heatmaps across 30 semantic DDI groups (S8), feature selection results (S9), error clustering analysis (S10), and web interface screenshots (S11).

> The manuscript is currently under review at *npj Digital Medicine*. A preprint will be made available upon acceptance.

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
