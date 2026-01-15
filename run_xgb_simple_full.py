"""
Full dataset runner for XGBoostSimpleMetaSelector.

This script runs training on the full dataset:
- Full dataset (max_rows = null)
- All 5 policies enabled
- XGBoost simple meta-selector (state features only)
- Class weights enabled

By default, loads and displays cached results from outputs/results.json.
Use --rerun to recompute and overwrite cached results.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omegaconf import OmegaConf
from src.train import main
from src.utils.results import load_cached_results, print_results

# Configuration for full experiment
cfg = {
    "seed": 0,
    "data": {"sample_path": "dataset/HealthGymV2_CbdrhDatathon_ART4HIV.csv", "max_rows": None},
    "columns": {"action": "Drug (M)", "vl_out": "VL (M)", "cd4_out": "CD4 (M)"},
    "split": {"val_frac": 0.3},
    "model": {"n_estimators": 50, "safety_threshold": 0.3},
    "policies": {"rule": True, "per_action": True, "dqn": True, "safety": True, "cf_knn": True},
    "meta": {
        "use_patient_features": True,
        "model_type": "rf",
        "selector_type": "xgboost_simple",
        # XGBoost hyperparameters for full run
        "xgb_simple_max_depth": 4,
        "xgb_simple_n_estimators": 200,
        "xgb_simple_learning_rate": 0.05,
        "xgb_simple_subsample": 0.8,
        "xgb_simple_colsample_bytree": 0.8,
        "xgb_simple_reg_lambda": 1.0,
        "xgb_simple_reg_alpha": 0.0,
        "xgb_simple_use_class_weights": True,
        "device": "cpu",
    },
    "logging": {
        "mlflow_dir": "mlruns",
        "experiment": "meta-recommender",
        "run_name": "xgb-simple-full"
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Simple Meta-Selector")
    parser.add_argument("--rerun", action="store_true", 
                       help="Recompute results instead of loading cached outputs")
    args = parser.parse_args()
    
    # Check for cached results first (unless --rerun is specified)
    if not args.rerun:
        results = load_cached_results()
        if results and "meta_selectors" in results and "xgboost_simple" in results["meta_selectors"]:
            print("=" * 80)
            print("XGBOOST SIMPLE META-SELECTOR - CACHED RESULTS")
            print("=" * 80)
            print_results(results, selector_type="xgboost_simple")
            sys.exit(0)
        else:
            print("No cached results found. Running training...")
            print("(Use --rerun to force recomputation)")
            print()
    
    # Run training
    print("=" * 80)
    print("XGBOOST SIMPLE META-SELECTOR FULL DATASET EXPERIMENT")
    print("=" * 80)
    print("Running with:")
    print("  - Full dataset (max_rows = null)")
    print("  - All 5 policies enabled (rule, per_action, dqn, safety, cf_knn)")
    print("  - Model: XGBoost with max_depth=4, n_estimators=200")
    print("  - Features: State only (no policy or CF features)")
    print("  - Class weights: enabled")
    print("=" * 80)
    print()
    
    main(OmegaConf.create(cfg))


