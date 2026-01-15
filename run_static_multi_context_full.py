"""
Full dataset runner for StaticMultiContextMetaSelector.

This script runs training on the full dataset with all 5 policies enabled:
- Full dataset (no subsampling)
- All policies: rule, per_action, dqn, safety, cf_knn
- Static multi-context architecture
- All fixes enabled (class weights, reward normalization)

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

# Configuration for full dataset experiment
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
        "selector_type": "static_multi_context",
        # Static multi-context hyperparameters for full run
        "static_d_model": 128,
        "static_nheads": 4,
        "static_nlayers": 2,
        "static_dim_feedforward": 256,
        "static_dropout": 0.15,
        "static_lr": 0.001,
        "static_batch_size": 64,
        "static_epochs": 25,
        "static_patience": 5,
        "static_min_delta": 1e-4,
        # Keep fixes enabled
        "static_use_class_weights": True,
        "static_reward_normalization": "zscore",
        "device": "cpu",
    },
    "logging": {
        "mlflow_dir": "mlruns",
        "experiment": "meta-recommender",
        "run_name": "static-multi-context-full"
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static Multi-Context Meta-Selector")
    parser.add_argument("--rerun", action="store_true", 
                       help="Recompute results instead of loading cached outputs")
    args = parser.parse_args()
    
    # Check for cached results first (unless --rerun is specified)
    if not args.rerun:
        results = load_cached_results()
        if results and "meta_selectors" in results and "static_multi_context" in results["meta_selectors"]:
            print("=" * 80)
            print("STATIC MULTI-CONTEXT META-SELECTOR - CACHED RESULTS")
            print("=" * 80)
            print_results(results, selector_type="static_multi_context")
            sys.exit(0)
        else:
            print("No cached results found. Running training...")
            print("(Use --rerun to force recomputation)")
            print()
    
    # Run training
    print("=" * 80)
    print("STATIC MULTI-CONTEXT FULL DATASET EXPERIMENT")
    print("=" * 80)
    print("Running with:")
    print("  - Full dataset (no subsampling)")
    print("  - All 5 policies enabled (rule, per_action, dqn, safety, cf_knn)")
    print("  - Model: d_model=128, n_heads=4, n_layers=2")
    print("  - Sequence length: N/A (static, no sequences)")
    print("  - Class weights: enabled")
    print("  - Reward normalization: zscore")
    print("  - Epochs: 25, patience: 5")
    print("=" * 80)
    print()
    
    main(OmegaConf.create(cfg))



