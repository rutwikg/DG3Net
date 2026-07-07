"""
ablation_v1.py — V1: V0 + geometry-aware normalization
    - 5x GATv2 blocks (no FEMAT)
    - GEOMETRY-AWARE normalization
    - supervised MSE loss ONLY

Isolates the effect of the preprocessing normalization vs. dataset-wide minmax.

Usage:
    python ablation_v1.py --gpu 1 --root /path/graphs --output_mode disp
"""
from runner import run_ablation

if __name__ == "__main__":
    run_ablation(
        variant                = "v1",
        use_femat_first        = False,
        use_geom_norm          = True,
        default_lambda_E       = 0.0,
        default_lambda_M       = 0.0,
        default_use_aug_edges  = False,
    )
