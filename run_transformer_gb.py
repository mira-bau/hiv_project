"""
Run training with Hybrid Transformer + Gradient Boosting meta-selector.

This combines:
- Transformer encoder for feature extraction
- XGBoost/LightGBM for final classification

Expected advantages:
- Rich representations from Transformer
- Efficient non-linear classification from GB
- Best of both deep learning and tree-based methods

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

# Configuration with Transformer-GB hybrid meta-selector
transformer_gb_cfg = {
    "seed": 0,
    "data": {"sample_path": "dataset/HealthGymV2_CbdrhDatathon_ART4HIV.csv", "max_rows": None},
    "columns": {"action": "Drug (M)", "vl_out": "VL (M)", "cd4_out": "CD4 (M)"},
    "split": {"val_frac": 0.3},
    "model": {"n_estimators": 50, "safety_threshold": 0.3},
    "policies": {"rule": True, "per_action": True, "dqn": True, "safety": True, "cf_knn": True},
    "meta": {
        "use_patient_features": True,
        "model_type": "rf",
        "selector_type": "transformer_gb",  # Hybrid approach
        # Transformer encoder parameters
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
        "gb_backend": "xgboost",  # or "lightgbm"
        "device": "cpu",
    },
    "logging": {
        "mlflow_dir": "mlruns",
        "experiment": "meta-recommender",
        "run_name": "transformer-gb-run"
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-GB Meta-Selector")
    parser.add_argument("--rerun", action="store_true", 
                       help="Recompute results instead of loading cached outputs")
    args = parser.parse_args()
    
    # Check for cached results first (unless --rerun is specified)
    if not args.rerun:
        results = load_cached_results()
        if results and "meta_selectors" in results and "transformer_gb" in results["meta_selectors"]:
            print("="*80)
            print("TRANSFORMER-GB META-SELECTOR - CACHED RESULTS")
            print("="*80)
            print_results(results, selector_type="transformer_gb")
            sys.exit(0)
        else:
            print("No cached results found. Running training...")
            print("(Use --rerun to force recomputation)")
            print()
    
    # Run training
    print("="*80)
    print("Running Meta-Recommender with HYBRID TRANSFORMER-GB Meta-Selector")
    print("="*80)
    print()
    print("Architecture:")
    print("  Phase 1: Train Transformer encoder for feature extraction")
    print("  Phase 2: Train XGBoost classifier on embeddings")
    print()
    print("Advantages:")
    print("  ✓ Deep feature learning from Transformer")
    print("  ✓ Efficient classification from Gradient Boosting")
    print("  ✓ Combines strengths of neural nets and trees")
    print()
    print("All base policies remain the same:")
    print("  - Rule-based")
    print("  - Per-action supervised")
    print("  - DQN (RL)")
    print("  - Safety-aware")
    print()
    print("="*80)
    print()
    
    main(OmegaConf.create(transformer_gb_cfg))
    
    print()
    print("="*80)
    print("Training complete!")
    print("Compare results in MLflow UI:")
    print("  - Run 'mlflow ui' in terminal")
    print("  - Compare all selector types:")
    print("    * sample-run (Random Forest)")
    print("    * transformer-v2-run (Transformer-V2)")
    print("    * transformer-gb-run (Transformer-GB Hybrid)")
    print("="*80)

