# `.DG3` — Interchange Schema for FE-Derived Graph Data

**Version 1.0** — This specification describes the `.DG3` interchange schema used by DG3Net for finite-element surrogate modelling.

---

## 1. Overview

`.dg3` is an interchange schema for FE-derived graph data intended for machine-learning surrogate modelling. Files with the `.dg3` extension are Torch-serialized binaries carrying a `torch_geometric.data.Data` instance with a **specific, fixed attribute layout** that preserves the FE-specific semantics required for supervised surrogate training: boundary conditions, degree-of-freedom flags, load definitions, contact pairs, and material metadata.

A file that does not conform to the attribute layout of Sections 3–6 is not a valid `.dg3` file, even if it is a Torch-serialized `Data` object. The `.dg3` extension identifies the file as ML-consumable input for graph-based FE surrogates without further schema inspection.

### 1.1 Design goal and scope

`.DG3` is designed specifically as an *interchange format between an FE solver and a graph-based surrogate model*. It is **not** a general-purpose mesh format. Where CGNS, ExodusII, and VTU/PVD are optimized for solver-native I/O or for post-processing pipelines, `.DG3` is optimized for preserving the FE-specific semantics that a graph-based surrogate needs to learn from.

| Format | Primary purpose | FE-semantic preservation |
|---|---|---|
| `.vtu` / `.vtp` | Post-processing visualization | Geometry + per-node scalars only |
| `.cgns` | Solver-native CFD interchange | Structured; solver-oriented |
| `.exodusII` | Solver-native FE interchange | Structured; solver-oriented |
| `.dg3` | **Graph-based ML surrogate interchange** | Node/edge/graph attributes with fixed layout preserving BCs, DOF flags, loads, contact pairs, and materials as first-class ML-training attributes |

### 1.2 Scope of the solver-agnostic property

The `.DG3` schema is neutral with respect to which FE solver produced the reference data. A reference converter reads solver-specific outputs (`.odb` for Abaqus/Explicit; extensible to other solvers) and writes the `.DG3` schema. The trained model performs inference without reference to the source solver. Data generation for training, however, necessarily depends on the specific FE solver used.

---

## 2. File-level format

A `.dg3` file is a Torch-serialized binary carrying a single `torch_geometric.data.Data` instance whose attribute set conforms to the layout described in Sections 3–6 below. The `.dg3` extension identifies the file as conforming to this specification; a generic Torch-serialized binary is *not* a `.dg3` file unless it exposes exactly the attribute schema defined here.

Loading from disk uses standard Torch machinery:

```python
import torch
data = torch.load('bumper_042.dg3', weights_only=False)
# data is a torch_geometric.data.Data instance conforming to this specification
```

The reference converter (Section 8) produces `.dg3` files with all required attributes populated. Consumers of `.dg3` files must validate the presence and shape of the required attributes (Sections 3–6). Additional user-defined fields are permitted only under the reserved `data.metadata` dict and are ignored by DG3Net.

---

## 3. Node-level attributes

Node attributes describe per-node quantities and are stored as tensors with a leading dimension of `N` (the number of nodes in the graph).

| Attribute | dtype | Shape | Description |
|---|---|---|---|
| `data.x` | float32 | `[N, F_x]` | Concatenated node feature tensor (see 3.1 below for composition) |
| `data.pos` | float32 | `[N, 3]` | Undeformed nodal coordinates $(x, y, z)$ in millimetres |
| `data.bc_mask` | int64 | `[N, D]` | Per-DOF binary Dirichlet flag: `1` = constrained, `0` = free. `D = 3` for solid, `D = 6` for shell (three translations, three rotations) |
| `data.bc_value` | float32 | `[N, D]` | Prescribed Dirichlet value for constrained DOFs; zero for free DOFs |
| `data.dof_active` | int64 | `[N, D]` | Per-DOF active/inactive mask. Distinguishes degrees of freedom actually resolved by the FE solver from those tied to zero by construction (e.g., in-plane vs out-of-plane at a shell node) |
| `data.material_id` | int64 | `[N]` | Integer index into the graph-level `material_registry` (see Section 5) |
| `data.thickness` | float32 | `[N]` | Section thickness in millimetres (for shell/beam models); `nan` for solid elements |

### 3.1 Composition of `data.x`

The concatenated node feature `data.x` at position $i$ is the horizontal concatenation of:

$$
x_i = [\, \mathbf{p}_i \parallel \mathbf{m}_i \parallel \mathbf{bc}_i \parallel \mathbf{bc}^{val}_i \parallel \mathbf{d}_i \parallel t_i \,]
$$

where:
- $\mathbf{p}_i \in \mathbb{R}^3$: node coordinates (from `data.pos`)
- $\mathbf{m}_i \in \mathbb{R}^{F_m}$: material features (looked up from `data.material_id` via the registry — see Section 5)
- $\mathbf{bc}_i \in \{0,1\}^D$: BC flag (from `data.bc_mask`)
- $\mathbf{bc}^{val}_i \in \mathbb{R}^D$: prescribed BC values
- $\mathbf{d}_i \in \{0,1\}^D$: active-DOF mask
- $t_i \in \mathbb{R}$: thickness (or 0 if solid)

Total feature dimension $F_x = 3 + F_m + D + D + D + 1$. The concatenation is precomputed by the reference converter and stored explicitly in `data.x` so that models can consume the feature tensor without accessing the individual attributes at runtime.

---

## 4. Edge-level attributes

The graph edges are stored in the standard PyG format.

| Attribute | dtype | Shape | Description |
|---|---|---|---|
| `data.edge_index` | int64 | `[2, E]` | Edge index tensor. Source nodes in row 0, target nodes in row 1. |
| `data.edge_attr` | float32 | `[E, F_e]` | Per-edge feature tensor (see 4.1) |
| `data.edge_type` | int64 | `[E]` | Edge classification: `0` = FE mesh edge, `1` = augmented edge (see 4.2) |

### 4.1 Composition of `data.edge_attr`

For each edge $(i, j)$ the feature vector is:

$$
e_{ij} = [\, \Delta\mathbf{x}_{ij} \parallel \|\Delta\mathbf{x}_{ij}\| \parallel \tau_{ij} \,]
$$

where $\Delta\mathbf{x}_{ij} = \mathbf{p}_j - \mathbf{p}_i$ is the relative position, $\|\Delta\mathbf{x}_{ij}\|$ is its magnitude, and $\tau_{ij}$ is the edge type (mesh vs augmented). Total edge feature dimension $F_e = 3 + 1 + 1 = 5$.

### 4.2 Mesh edges vs augmented edges

**Mesh edges** are constructed directly from the FE element connectivity: each element contributes one edge per adjacent node pair. Edge orientation is undirected (both `(i,j)` and `(j,i)` entries appear in `edge_index`).

**Augmented edges** are constructed via k-nearest-neighbour search in the undeformed coordinate space, with $k = 5$ typical. They broaden the receptive field of the FEMAT operator beyond nearest-neighbour mesh topology alone. Augmented edges do not correspond to any element in the source FE model and are for training purposes only.

Augmented-edge construction is performed once at data-preprocessing time. Nearest-neighbour search is restricted to nodes belonging to the same physical component (identified via `material_id` or an explicit `component_id` graph attribute) to avoid spurious cross-component edges.

---

## 5. Graph-level attributes

Graph-level attributes are stored as fields on the `Data` object with a leading dimension of `1` (or as scalar tensors) so that they broadcast correctly during batching by `torch_geometric.data.Batch`.

| Attribute | dtype | Shape | Description |
|---|---|---|---|
| `data.load_vector` | float32 | `[1, 6]` | Global loading definition. Components: `(F_x, F_y, F_z, v_x, v_y, v_z)`. For crash simulations, this stores the initial impact velocity (m/s); for quasi-static, the concentrated force (N). Zero for components not applicable to the load case. |
| `data.load_type` | int64 | `[1]` | Load case identifier: `0` = displacement-driven, `1` = force-driven, `2` = velocity-driven (crash), `3` = mixed |
| `data.material_registry` | float32 | `[M, F_m]` | Lookup table: row `k` holds the material properties for `material_id = k`. Standard entries: `[E, ν, ρ, σ_y, ε_p_max, T_ref, ...]` |
| `data.contact_pairs` | int64 | `[P, 2]` | Node index pairs participating in prescribed contact interactions. Empty tensor `[0, 2]` if no contact defined |
| `data.contact_type` | int64 | `[P]` | Contact classification per pair: `0` = tied, `1` = self-contact, `2` = general contact |
| `data.metadata` | dict | — | Free-form Python dict with: `design_id` (str), `cross_section_class` (str, one of A–E for the 3D benchmark), `source_solver` (str, e.g. `"abaqus_explicit"`), and any additional trace fields |

---

## 6. Target attributes

Target fields are present in training/validation/test datasets and are the quantities the surrogate is trained to predict.

| Attribute | dtype | Shape | Description |
|---|---|---|---|
| `data.y_disp` | float32 | `[N, 3]` | Nodal displacement field at the reported time step (end-state for crash; final loaded state for static) |
| `data.y_stress` | float32 | `[N, 6]` | Cauchy stress in Voigt form `[σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz]`. Optional; present only for benchmarks with stress targets |
| `data.y_strain` | float32 | `[N, 6]` | Total strain (Green–Lagrange for finite-strain benchmarks; small-strain otherwise) in Voigt form. Optional |
| `data.y_plastic_strain` | float32 | `[N, 6]` | Plastic strain in Voigt form. Optional |
| `data.y` | float32 | `[N, F_y]` | Concatenated target tensor for models with a shared output head. Order: `[disp ‖ stress ‖ strain ‖ plastic_strain]`, with missing quantities zero-padded and masked at loss time |

Not all datasets carry all four target fields. The 3D bumper crash benchmark uses `y_disp` only. The 2D plate benchmark uses `y_stress` and `y_plastic_strain`.

---

## 7. Loading a `.DG3` file — example

```python
import torch
from torch_geometric.data import Data

data = torch.load('bumper_042.dg3', weights_only=False)

# --- inspect structure ---
print(f"Nodes:       {data.x.shape[0]}")
print(f"Edges:       {data.edge_index.shape[1]}")
print(f"Feature dim: {data.x.shape[1]}")
print(f"Aug edges:   {(data.edge_type == 1).sum().item()}")

# --- access FE semantics ---
n_constrained = (data.bc_mask.sum(dim=1) > 0).sum().item()
print(f"Constrained nodes: {n_constrained}")

velocity = data.load_vector[0, 3:6]
print(f"Impact velocity (m/s): {velocity}")

# --- access target ---
disp_max = data.y_disp.norm(dim=1).max().item()
print(f"Max nodal displacement: {disp_max:.2f} mm")
```

---

## 8. Reference converter

The reference `abaqus_to_dg3.py` converter (to be released) reads an Abaqus/Explicit `.odb` file and writes a `.DG3` file conforming to this specification. The converter is intended to serve as a template for extension to other solvers (Ansys LS-Dyna, PamCrash, in-house codes). Extensions are welcome once the license is converted to open source upon publication.

---

## 9. Versioning and compatibility

Files conforming to this specification carry `data.metadata["schema_version"] = "1.0"`.

Future schema revisions will preserve backward compatibility for the attributes defined in Sections 3–6 above. New optional fields may be added; existing fields will not be removed or repurposed.

---

## 10. License

`.DG3` files produced by the reference converter and included with DG3Net are released under the same terms as the code (see `LICENSE`). The `.DG3` schema itself, as documented in this specification, may be freely implemented by third parties.
