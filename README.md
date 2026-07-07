# DG3Net -  Dynamics-Informed, Geometry-Aware, Gradient-Based Graph Neural Network in Computational Science

This directory contains source files of **DG3Net**, a Dynamics-Informed, Geometry-Aware, Gradient-Based Graph Neural Network in Computational Science. The code accompanies the manuscript:

> R. Gulakala, K. S. Pappu, M. Stoffel. *DG3Net — A Dynamics-Informed, Geometry-Aware, Gradient-Based Graph Neural Network in Computational Science.*

## Status

The repository is released in **view-only** mode during the peer-review period. Once the paper is accepted, the complete pipeline (training driver scripts, table builders, plotting utilities, `.DG3` reference converter, and preprocessed dataset) will be released, and the license will be converted to open-source terms. See `LICENSE`.

The files currently released cover the model architecture, evaluation infrastructure, and training runner.

---

## What's here

| File | Purpose |
|---|---|
| `femat.py` | The **FEMAT** attention operator — conditions attention weights on relative geometric direction, material contrast, and DOF compatibility. |
| `common.py` | Shared model backbone (`GATv2Backbone`), data loading, geometry-aware normalization, augmented-edge construction, and statistics computation. |
| `baseline_models.py` | Reference implementations of the four state-of-the-art graph baselines: **MeshGraphNet**, **X-MeshGraphNet**, **EA-GNN**, **Graph U-Net**. |
| `metrics.py` | Six evaluation metrics: test MSE, relative L², peak error, max nodal error, top-5% error, per-sample R². |
| `checkpoint.py` | Atomic checkpoint save/load. |
| `runner.py` | Main training loop for the DG3Net configuration variants. |
| `train_baselines.py` | Trains the four graph baselines under the DG3Net training protocol. |
| `compute_baseline_metrics.py` | Evaluates trained baseline checkpoints on the test set. |
| `kfold_cv.py` | k-fold cross-validation. |

Also released:

- `DG3_SPEC.md` — Format specification for the `.DG3` interchange schema.
- Trained checkpoints (`.pt` files) — released for reviewer inspection.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.4
- NumPy, SciPy, matplotlib
- tqdm

DG3Net was trained on a single NVIDIA H100 80 GB GPU; a single V4 seed for 300 epochs completes in approximately 1 hour.

---

## Data format

The pipeline consumes preprocessed graph files in the `.DG3` interchange schema — a PyTorch Geometric `Data` object serialized as a `.pt` file, extended with FE-specific semantic attributes (boundary conditions, degree-of-freedom flags, load definitions, contact pairs, and material metadata) as first-class graph attributes.

The full format specification is in [`DG3_SPEC.md`](DG3_SPEC.md).

The 3D bumper crash dataset (225 designs) and the reference `.DG3` converter from Abaqus/Explicit `.odb` files will be released upon publication of the article.

---

## License

**Academic Research License — View Only**

Copyright (c) 2025 Rutwik Gulakala

Permission is granted to view and inspect this repository and its contents solely for the purpose of academic review, replication assessment, and scholarly discussion.

No permission is granted to:
- Copy or redistribute the code
- Modify or create derivative works
- Use the code in other research, software, or commercial products

Any use beyond viewing requires explicit written permission from the author.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.**

The license may be converted to an open-source license upon publication of the article. See `LICENSE` for the full terms.

---

## Contact

For questions about the code or the paper, please open an issue on the repository or contact the corresponding author. - gulakala@iam.rwth-aachen.de 
