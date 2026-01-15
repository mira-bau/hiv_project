"""
Run complete comparison: All 5 individual policies + 3 meta-selectors.

This script trains and evaluates:
- 5 Individual Expert Policies: rule, per_action, dqn, safety, cf_knn
- 3 Meta-Selectors: xgboost_simple, transformer_gb, static_multi_context

Outputs IPS/SNIPS/DR metrics for all models in a comparison table.

By default, loads and displays cached results from outputs/results.json.
Use --rerun to recompute and overwrite cached results.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omegaconf import OmegaConf
from src.train import main
from src.utils.results import load_cached_results, print_results
import pandas as pd

def run_experiment(selector_type, run_name, max_rows=None):
    """Run training with specified selector type."""
    print("\n" + "="*80)
    print(f"Running: {selector_type.upper()} Meta-Selector")
    print("="*80)
    
    cfg = {
        "seed": 0,
        "data": {"sample_path": "dataset/HealthGymV2_CbdrhDatathon_ART4HIV.csv", "max_rows": max_rows},
        "columns": {"action": "Drug (M)", "vl_out": "VL (M)", "cd4_out": "CD4 (M)"},
        "split": {"val_frac": 0.3},
        "model": {"n_estimators": 50, "safety_threshold": 0.3},
        "policies": {"rule": True, "per_action": True, "dqn": True, "safety": True, "cf_knn": True},
        "meta": {
            "use_patient_features": True,
            "model_type": "rf",
            "selector_type": selector_type,
            "device": "cpu",
        },
        "logging": {
            "mlflow_dir": "mlruns",
            "experiment": "meta-recommender",
            "run_name": run_name
        },
    }
    
    # Add selector-specific hyperparameters
    if selector_type == "xgboost_simple":
        cfg["meta"].update({
            "xgb_simple_max_depth": 4,
            "xgb_simple_n_estimators": 200,
            "xgb_simple_learning_rate": 0.05,
            "xgb_simple_subsample": 0.8,
            "xgb_simple_colsample_bytree": 0.8,
            "xgb_simple_reg_lambda": 1.0,
            "xgb_simple_reg_alpha": 0.0,
            "xgb_simple_use_class_weights": True,
        })
    elif selector_type == "transformer_gb":
        cfg["meta"].update({
            "transformer_hidden_size": 128,
            "transformer_num_heads": 4,
            "transformer_num_layers": 2,
            "transformer_dropout": 0.25,
            "transformer_lr": 0.001,
            "transformer_batch_size": 64,
            "transformer_epochs": 50,
            "transformer_patience": 8,
            "transformer_min_delta": 1e-4,
            "transformer_grad_clip": 1.0,
            "gb_backend": "xgboost",
        })
    elif selector_type == "static_multi_context":
        cfg["meta"].update({
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
            "static_use_class_weights": True,
            "static_reward_normalization": "zscore",
        })
    
    try:
        main(OmegaConf.create(cfg))
        print(f"\n✓ {selector_type.upper()} completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ {selector_type.upper()} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_comparison_summary():
    """Print summary of results (would need to parse MLflow logs for full automation)."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("\nTo view detailed results:")
    print("  1. Run 'mlflow ui' in the medical_case/ directory")
    print("  2. Compare runs:")
    print("     - Individual policies: Check val_ips_rule, val_ips_per_action, etc.")
    print("     - Meta-selectors: Check val_ips_meta for each selector type")
    print("\nExpected performance (from previous runs):")
    print("  Individual Policies:")
    print("    - Rule: IPS ~0.09")
    print("    - Per-Action: IPS ~0.19")
    print("    - DQN: IPS ~0.15")
    print("    - Safety: IPS ~0.18")
    print("    - CF-kNN: IPS ~0.09")
    print("\n  Meta-Selectors:")
    print("    - XGBoost Simple: IPS ~0.35")
    print("    - Transformer-GB: IPS ~1.35")
    print("    - Static Multi-Context: IPS ~2.37 (best)")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete Model Comparison")
    parser.add_argument("--rerun", action="store_true", 
                       help="Recompute results instead of loading cached outputs")
    args = parser.parse_args()
    
    # Check for cached results first (unless --rerun is specified)
    if not args.rerun:
        results = load_cached_results()
        if results:
            print("="*80)
            print("COMPLETE MODEL COMPARISON - CACHED RESULTS")
            print("="*80)
            print_results(results)
            sys.exit(0)
        else:
            print("No cached results found. Running training...")
            print("(Use --rerun to force recomputation)")
            print()
    
    # Run training
    print("="*80)
    print("COMPLETE MODEL COMPARISON SUITE")
    print("="*80)
    print("\nThis script will train and evaluate:")
    print("  1. All 5 individual expert policies")
    print("  2. All 3 meta-selectors (xgboost_simple, transformer_gb, static_multi_context)")
    print("\nAll results will be logged to MLflow for comparison.")
    print("="*80)
    
    # Run all meta-selectors (individual policies are trained as part of each run)
    results = {}
    
    selectors = [
        ("xgboost_simple", "comparison-xgb-simple"),
        ("transformer_gb", "comparison-transformer-gb"),
        ("static_multi_context", "comparison-static-multi-context"),
    ]
    
    for selector_type, run_name in selectors:
        success = run_experiment(selector_type, run_name, max_rows=None)
        results[selector_type] = success
    
    # Print summary
    print_comparison_summary()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review logs above for IPS/SNIPS/DR metrics")
    print("  2. Run 'mlflow ui' to view detailed comparison")
    print("  3. Check artifacts/ for saved model files")
    print("="*80)

