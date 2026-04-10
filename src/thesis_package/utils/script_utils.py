"""
Shared utility functions for model evaluation scripts.
"""

from pathlib import Path

def ensure_output_dirs(*paths: Path) -> None:
    """Create output directories if needed."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def print_performance(results: dict, title: str) -> None:
    """Pretty-print summary metrics for one experiment."""
    summary = results["summary"]
    print(f"=== {title} ===")
    print(f"MAE:        {summary['MAE']:.4f} ± {summary['MAE_STD']:.4f}")
    print(f"Spearman ρ: {summary['Spearman_R']:.4f}")
    print(f"R²:         {summary['R2']:.4f}\n")

def build_results_row(model_name: str, results: dict) -> dict:
    """Convert evaluation output into a flat row for CSV export."""
    summary = results["summary"]
    return {
        "Model": model_name,
        "MAE": f"{summary['MAE']:.4f}",
        "MAE_STD": f"{summary['MAE_STD']:.4f}",
        "Spearman_R": f"{summary['Spearman_R']:.4f}",
        "R2": f"{summary['R2']:.4f}",
    }