"""
ablation_v3.py — V3: V2 + constitutive consistency regularizer
    - 5x GATv2 blocks
    - geometry-aware normalization
    - MSE + adaptive L_E + constitutive residual L_M

L_M requires output_mode = 'sspe' or 'all' (needs S, LE, PE channels).
If --output_mode=disp is passed, L_M is silently disabled and V3 == V2.

Usage (recommended, all channels):
    python ablation_v3.py --gpu 3 --root /path/graphs --output_mode all
Usage (stresses only, no displacement):
    python ablation_v3.py --gpu 3 --root /path/graphs --output_mode sspe
"""
import sys
from runner import run_ablation

if "--adaptive_lambda_E" not in sys.argv:
    sys.argv.extend(["--adaptive_lambda_E", "on"])

if __name__ == "__main__":
    run_ablation(
        variant                = "v3",
        use_femat_first        = False,
        use_geom_norm          = True,
        default_lambda_E       = 0.0,
        default_lambda_M       = 1e-4,
        default_use_aug_edges  = False,
    )
