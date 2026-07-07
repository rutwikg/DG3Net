"""
ablation_v4a.py — V4a: FEMAT + augmented edges + geometry-aware norm, MSE only
    - 1x FEMAT (mesh + augmented edges) + 4x GATv2 blocks
    - geometry-aware normalization
    - supervised MSE ONLY  (no L_E, no L_M)

Purpose: isolates the effect of the FEMAT + augmented-edge architecture from
the auxiliary losses. Compare directly against V1 (which has GATv2 in the
first block instead of FEMAT and no augmented edges). Any val-MSE gap between
V4a and V1 is attributable to the FEMAT-augmented-edge block alone.

Usage:
    python ablation_v4a.py --gpu 0 --root /path/graphs --output_mode disp
"""
import sys
from runner import run_ablation

if "--use_aug_edges" not in sys.argv:
    sys.argv.extend(["--use_aug_edges", "on"])

if __name__ == "__main__":
    run_ablation(
        variant                = "v4a",
        use_femat_first        = True,
        use_geom_norm          = True,
        default_lambda_E       = 0.0,
        default_lambda_M       = 0.0,
        default_use_aug_edges  = True,
    )
