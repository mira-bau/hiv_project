# Medical Case: HIV Treatment Meta-Recommender

This directory contains the complete medical case implementation for the HIV antiretroviral therapy (ART) meta-recommender system.

## Overview

The system compares:
- **5 Individual Expert Policies**: rule-based, per-action supervised, DQN, safety-aware, CF-kNN
- **3 Meta-Selectors**: XGBoost Simple (baseline), Transformer-GB (hybrid), Static Multi-Context Transformer (main contribution)

All models are trained and evaluated on the HealthGym ART4HIV dataset using offline evaluation metrics (IPS, SNIPS, DR).

## Quick Start

### 0. Extract the artifacts

The model artifacts are stored in a split zip file (due to GitHub's 100 MB file size limit). To extract them:

**Option 1: Recombine and extract (recommended)**
```bash
cd artifacts
cat extract_me.zip.part* > extract_me.zip
unzip extract_me.zip
```

**Option 2: Extract directly without creating the full zip file**
```bash
cd artifacts
cat extract_me.zip.part* | unzip -d . -
```

Make sure all three parts (`extract_me.zip.partaa`, `extract_me.zip.partab`, `extract_me.zip.partac`) are present in the `artifacts/` directory.

### 1. Install Dependencies

```bash
cd medical_case
pip install -r requirements.txt
```

### 2. Run Complete Comparison

**Default mode (cached results - fast):**
```bash
cd medical_case
python run_comparison.py
```

This will:
- Load and display cached results from `outputs/results.json` (instant)
- Show IPS/SNIPS/DR metrics for all models
- Show policy selection distributions

**Recompute mode (full training):**
```bash
cd medical_case
python run_comparison.py --rerun
```

This will:
- Train all 5 individual policies
- Train all 3 meta-selectors
- Output IPS/SNIPS/DR metrics for comparison
- Save trained models to `artifacts/`
- Overwrite cached results in `outputs/results.json`

### 3. Run Individual Meta-Selectors

All run scripts support cached results by default:

**XGBoost Simple (state-only baseline):**
```bash
cd medical_case
python run_xgb_simple_full.py          # Load cached results
python run_xgb_simple_full.py --rerun  # Recompute
```

**Transformer-GB (hybrid baseline):**
```bash
cd medical_case
python run_transformer_gb.py          # Load cached results
python run_transformer_gb.py --rerun  # Recompute
```

**Static Multi-Context Transformer (main contribution):**
```bash
cd medical_case
python run_static_multi_context_full.py          # Load cached results
python run_static_multi_context_full.py --rerun  # Recompute
```

### 4. Launch Clinical UI

Run the Streamlit UI for patient-level recommendations:

```bash
cd medical_case
streamlit run app/clinical_app_real.py
```

Then open in browser: `http://localhost:8501`

The UI loads the saved final fusion model from `artifacts/` (prioritizes `static_multi_context` if available). The UI does not retrain models.

## Dataset

The dataset is located at:
- `dataset/HealthGymV2_CbdrhDatathon_ART4HIV.csv`

This contains patient trajectories with:
- Patient state features (CD4, VL, demographics, treatment history)
- Actions (drug combinations)
- Outcomes (VL, CD4 at next timestep)

## Model Artifacts

Trained models are saved to:
- `artifacts/meta_selector_static_multi_context.pkl` (latest fusion model)
- `artifacts/meta_selector_transformer_gb.pkl`
- `artifacts/meta_selector_xgboost_simple.pkl`

## Cached Results

Final results (metrics, policy selections) are stored in:
- `outputs/results.json` - Complete comparison results

**Default behavior:** All run scripts load and display cached results instantly (no training).

**To recompute:** Use the `--rerun` flag on any run script.

## Expected Performance

Based on full dataset runs:

| Model | IPS | SNIPS | DR |
|-------|-----|-------|-----|
| **Individual Policies** |
| Rule | 0.0869 | 0.0869 | -0.0030 |
| Per-Action | 0.1926 | 0.1926 | 0.0080 |
| DQN | 0.1537 | 0.1537 | 0.0069 |
| Safety | 0.1815 | 0.1815 | 0.0074 |
| CF-kNN | 0.0907 | 0.0907 | 0.0069 |
| **Meta-Selectors** |
| XGBoost Simple | 0.3515 | 0.3515 | 0.0060 |
| Transformer-GB | 1.3539 | 1.3539 | 0.0666 |
| Static Multi-Context | 2.3654 | 2.3654 | 0.0843 |

## Directory Structure

```
medical_case/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── dataset/                     # HealthGym ART4HIV dataset
├── src/                         # Core code
│   ├── data/loaders.py         # Dataset loading
│   ├── features/                # Feature engineering
│   ├── models/baselines.py      # 5 expert policies
│   ├── meta/                    # 3 meta-selectors
│   ├── eval/ips.py             # Offline evaluation metrics
│   ├── ui/twin3d.py            # UI rendering helper
│   └── train.py                # Main training pipeline
├── app/
│   └── clinical_app_real.py    # Final 2D patient view UI
├── run_comparison.py            # Single entrypoint for full comparison
├── run_xgb_simple_full.py       # XGBoost simple full run
├── run_transformer_gb.py       # Transformer-GB full run
├── run_static_multi_context_full.py  # Static fusion full run
├── outputs/                     # Cached results (metrics, distributions)
│   └── results.json
└── artifacts/                   # Trained model artifacts (.pkl files)
```

## Notes

- **Default mode (cached):** All run scripts load cached results from `outputs/results.json` instantly
- **Recompute mode:** Use `--rerun` flag to train models and overwrite cached results
- All training runs use the full dataset by default (`max_rows: null`)
- Models are saved automatically after training to `artifacts/`
- MLflow logs are saved to `mlruns/` (can be viewed with `mlflow ui`)
- The UI automatically loads the best available model from `artifacts/` (prioritizes `static_multi_context`)
- The UI does not retrain models - it only loads pre-trained artifacts

