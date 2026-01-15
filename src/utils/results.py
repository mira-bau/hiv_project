"""
Utility functions for loading and displaying cached results.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


def load_cached_results(results_file: Optional[str] = None) -> Optional[Dict]:
    """Load cached results from outputs directory."""
    if results_file is None:
        results_file = Path(__file__).parent.parent.parent / "outputs" / "results.json"
    else:
        results_file = Path(results_file)
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading cached results: {e}")
        return None


def print_results(results: Dict, selector_type: Optional[str] = None):
    """Print formatted results."""
    print("\n" + "="*80)
    print("CACHED RESULTS")
    print("="*80)
    
    # Print individual policies
    if "individual_policies" in results:
        print("\nIndividual Expert Policies:")
        print("-" * 80)
        print(f"{'Policy':<20} {'IPS':>12} {'SNIPS':>12} {'DR':>12}")
        print("-" * 80)
        for policy_name, metrics in results["individual_policies"].items():
            ips = metrics.get("ips", 0.0)
            snips = metrics.get("snips", 0.0)
            dr = metrics.get("dr", 0.0)
            print(f"{policy_name:<20} {ips:>12.4f} {snips:>12.4f} {dr:>12.4f}")
    
    # Print meta-selectors
    if "meta_selectors" in results:
        print("\nMeta-Selectors:")
        print("-" * 80)
        print(f"{'Selector':<25} {'IPS':>12} {'SNIPS':>12} {'DR':>12}")
        print("-" * 80)
        for selector_name, metrics in results["meta_selectors"].items():
            if selector_type and selector_name != selector_type:
                continue
            ips = metrics.get("ips", 0.0)
            snips = metrics.get("snips", 0.0)
            dr = metrics.get("dr", 0.0)
            print(f"{selector_name:<25} {ips:>12.4f} {snips:>12.4f} {dr:>12.4f}")
            
            # Print policy selections if available
            if "policy_selections" in metrics:
                print(f"\n  Policy Selection Distribution:")
                total = sum(metrics["policy_selections"].values())
                for policy, count in metrics["policy_selections"].items():
                    pct = (count / total * 100) if total > 0 else 0.0
                    print(f"    {policy:<15} {count:>8,} ({pct:>5.1f}%)")
    
    # Print metadata
    if "metadata" in results:
        print("\n" + "-" * 80)
        print("Metadata:")
        for key, value in results["metadata"].items():
            print(f"  {key}: {value}")
    
    print("="*80)
    print("\nTo recompute results, run with --rerun flag")


def save_results(results: Dict, results_file: Optional[str] = None):
    """Save results to outputs directory."""
    if results_file is None:
        results_file = Path(__file__).parent.parent.parent / "outputs" / "results.json"
    else:
        results_file = Path(results_file)
    
    # Ensure directory exists
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

