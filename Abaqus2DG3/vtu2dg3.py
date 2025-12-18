# Author - Rutwik Gulakala - gulakala@iam.rwth-aachen.de
# Convert VTU to DG3 format
# vtu2dg3.py
from __future__ import annotations
import os, re, glob, hashlib, xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

import meshio
import torch
from torch_geometric.data import Data

# ---------- Natural sort (prefer natsort if available; fallback otherwise) ----------
try:
    from natsort import natsorted as _natsorted  # pip install natsort (if present in env)
except Exception:
    import re as _re
    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in _re.split(r'(\d+)', str(s))]
    def _natsorted(it):
        return sorted(it, key=_natural_key)

# ------------------------- helpers -------------------------

def _find_file(root: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        hits = glob.glob(os.path.join(root, pat))
        if hits:
            hits = _natsorted(hits)
            return hits[0]
    return None

def _read_pvd(pvd_path: str) -> List[Tuple[float, str]]:
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    ts_files = []
    for ds in root.iter("DataSet"):
        t = float(ds.attrib.get("timestep", "0.0"))
        f = ds.attrib["file"]
        ts_files.append((t, f))
    ts_files.sort(key=lambda x: x[0])
    return ts_files

def _vtk_cells_used_nodes(cells: Dict[str, np.ndarray]) -> np.ndarray:
    used: Set[int] = set()
    for _, conn in cells.items():
        if conn.size == 0:
            continue
        a = np.asarray(conn, dtype=np.int64)
        if a.ndim != 2:
            continue
        used.update(int(i) for i in a.ravel())
    if not used:
        return np.array([], dtype=np.int64)
    return np.array(sorted(used), dtype=np.int64)

def _vtk_cells_to_edges(cells: Dict[str, np.ndarray]) -> np.ndarray:
    edges = set()
    for _, conn in cells.items():
        if conn.size == 0:
            continue
        conn = np.asarray(conn, dtype=np.int64)
        if conn.ndim != 2:
            continue
        k = conn.shape[1]
        for row in conn:
            for i in range(k):
                for j in range(i + 1, k):
                    a = int(row[i]); b = int(row[j])
                    if a == b:
                        continue
                    edges.add((min(a, b), max(a, b)))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(list(edges), dtype=np.int64).T

def _get_point_array(m: meshio.Mesh, names: List[str]) -> Optional[np.ndarray]:
    """
    Read a point_data array robustly:
    - Case-insensitive matching
    - Supports aliases (names list)
    - Supports suffixes like _vec, _tensor, _components, _vector
    - Normalizes shape to [n_points, C]
    """
    if not m.point_data:
        return None

    n_pts = m.points.shape[0]
    pd_map = {k.lower(): k for k in m.point_data.keys()}

    suffixes = ["", "_vec", "_vector", "_components", "_component", "_tensor"]

    def try_key(base: str) -> Optional[np.ndarray]:
        base_l = base.lower()
        for suf in suffixes:
            cand = (base_l + suf).lower()
            if cand in pd_map:
                real_key = pd_map[cand]
                arr = np.asarray(m.point_data[real_key])

                # Normalize to [n_pts, C]
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim >= 3:
                    arr = arr.reshape(arr.shape[0], -1)

                if arr.shape[0] != n_pts:
                    return None

                return arr.astype(np.float64, copy=False)
        return None

    for nm in names:
        out = try_key(nm)
        if out is not None:
            return out
    return None


def _get_cell_array_as_nodes(
    m: meshio.Mesh,
    names: List[str],
    n_nodes: int,
    allow_cell_avg: bool
) -> Optional[np.ndarray]:
    """
    Convert a cell_data field into a node-based array by averaging values of
    all incident cells for each node.

    - Case-insensitive matching
    - Supports aliases (names list)
    - Supports suffixes: "", _vec, _vector, _components, _component, _tensor
    - Normalizes shape to [n_cells, C] before averaging
    - Returns [n_nodes, C] or None
    """
    if not allow_cell_avg or not m.cell_data:
        return None

    cd_map = {k.lower(): k for k in m.cell_data.keys()}
    suffixes = ["", "_vec", "_vector", "_components", "_component", "_tensor"]

    def resolve_key(base: str) -> Optional[str]:
        base_l = base.lower()
        for suf in suffixes:
            cand = (base_l + suf).lower()
            if cand in cd_map:
                return cd_map[cand]
        return None

    picked_key = None
    for nm in names:
        k = resolve_key(nm)
        if k is not None:
            picked_key = k
            break
    if picked_key is None:
        return None

    values_per_block = m.cell_data[picked_key]

    node_sums = None
    node_counts = np.zeros((n_nodes, 1), dtype=np.float64)

    for blk_idx, cell_block in enumerate(m.cells):
        conn = cell_block.data  # [n_cells_block, nodes_per_cell]
        if conn.size == 0:
            continue

        vals = np.asarray(values_per_block[blk_idx])

        # Normalize to [n_cells_block, C]
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        elif vals.ndim >= 3:
            vals = vals.reshape(vals.shape[0], -1)

        if vals.shape[0] != conn.shape[0]:
            # mismatched block, skip
            continue

        C = vals.shape[1]
        if node_sums is None:
            node_sums = np.zeros((n_nodes, C), dtype=np.float64)

        for ci, nodes in enumerate(conn):
            v = vals[ci]
            unique_nodes = np.unique(nodes.astype(np.int64))
            node_sums[unique_nodes] += v
            node_counts[unique_nodes, 0] += 1.0

    if node_sums is None:
        return None

    out = np.zeros_like(node_sums)
    mask = node_counts[:, 0] > 0
    out[mask] = node_sums[mask] / node_counts[mask]
    return out


def _to_vec3(arr: np.ndarray) -> np.ndarray:
    if arr is None: return None
    if arr.shape[1] >= 3: return arr[:, :3]
    if arr.shape[1] == 2:
        return np.hstack([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)])
    if arr.shape[1] == 1:
        return np.hstack([arr, np.zeros((arr.shape[0], 2), dtype=arr.dtype)])
    return np.zeros((arr.shape[0], 3), dtype=np.float64)

def _tensor_like_to_voigt6(arr: np.ndarray) -> np.ndarray:
    if arr is None: return None
    if arr.shape[1] == 6: return arr
    if arr.shape[1] == 9:
        out = np.empty((arr.shape[0], 6), dtype=arr.dtype)
        out[:, 0] = arr[:, 0]  # 11
        out[:, 1] = arr[:, 4]  # 22
        out[:, 2] = arr[:, 8]  # 33
        out[:, 3] = arr[:, 1]  # 12
        out[:, 4] = arr[:, 2]  # 13
        out[:, 5] = arr[:, 5]  # 23
        return out
    out = np.zeros((arr.shape[0], 6), dtype=arr.dtype)
    out[:, 0] = arr[:, 0] if arr.shape[1] > 0 else 0.0
    return out

def _get_possible_point_ids(m: meshio.Mesh) -> Optional[np.ndarray]:
    candidates = [
        "GlobalNodeId", "GlobalIds", "GlobalPointIds",
        "vtkOriginalPointIds", "nodeId", "NodeId", "NODEID", "NID"
    ]
    if not m.point_data:
        return None
    keys = {k.lower(): k for k in m.point_data.keys()}
    for c in candidates:
        k = keys.get(c.lower())
        if k is not None:
            arr = np.asarray(m.point_data[k]).reshape(-1)
            if arr.shape[0] == m.points.shape[0]:
                return arr.astype(np.int64, copy=False)
    return None

# ------------------------- INP parser (lightweight) -------------------------

INP_SECTION_SHELL = re.compile(r'^\*SHELL SECTION\b', re.IGNORECASE)

def _parse_uniform_shell_thickness(inp_path: str) -> Tuple[Optional[float], Dict[str, float]]:
    if not os.path.isfile(inp_path):
        return None, {}
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    sections = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if INP_SECTION_SHELL.match(line):
            params = line.split(',')
            elset = None
            for p in params[1:]:
                if p.strip().lower().startswith('elset='):
                    elset = p.split('=')[1].strip()
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith(('*','**'))):
                j += 1
            if j < len(lines):
                try:
                    thickness = float(lines[j].strip().split(',')[0])
                    sections.append((elset or f"elset_{len(sections)+1}", thickness))
                    i = j
                except: pass
        i += 1
    if not sections:
        return None, {}
    uniq = {round(t, 12) for _, t in sections}
    if len(uniq) == 1:
        return float(next(iter(uniq))), {name: t for name, t in sections}
    return None, {name: t for name, t in sections}

# ------------------------- main builder -------------------------

def build_graph_from_run(
    vtu_run_dir: str,
    run_src_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    fields_y: Optional[List[str]] = None,
    *,
    prune_unused_points: bool = True,
    allow_cell_avg: bool = True,
    strict_point_count: bool = True,
    debug: bool = True,
    print_counts: bool = True,
) -> Data:
    """
    Build a PyG Data object from a VTU time series (referenced by a PVD).
    """
    if fields_y is None:
        fields_y = ["U", "S", "LE", "PE"]

    # --- identifiers you asked for ---
    vtu_folder_abs = os.path.abspath(vtu_run_dir)
    vtu_folder_name = os.path.basename(os.path.normpath(vtu_folder_abs))
    # common patterns: either the folder is exactly "<run>" or "<run>_VTU"
    run_label_guess = vtu_folder_name[:-4] if vtu_folder_name.lower().endswith("_vtu") else vtu_folder_name

    pvd_path = _find_file(vtu_run_dir, ["export.pvd", "*.pvd"])
    if not pvd_path:
        raise FileNotFoundError("PVD file not found in %s" % vtu_run_dir)

    ts_files = _read_pvd(pvd_path)
    if not ts_files:
        raise RuntimeError("No datasets in %s" % pvd_path)

    # small, stable fingerprint for later sanity checks
    try:
        with open(pvd_path, "rb") as _f:
            pvd_sha1 = hashlib.sha1(_f.read()).hexdigest()
    except Exception:
        pvd_sha1 = ""

    # --- pass 0: inspect frames for point-count consistency & union of used nodes ---
    first_mesh = meshio.read(os.path.join(vtu_run_dir, ts_files[0][1]))
    N0 = int(first_mesh.points.shape[0])

    used_union: Set[int] = set(_vtk_cells_used_nodes(first_mesh.cells_dict).tolist())
    N_inconsistent = False
    id_field_present = False
    for _, rel in ts_files[1:]:
        m = meshio.read(os.path.join(vtu_run_dir, rel))
        if m.points.shape[0] != N0:
            N_inconsistent = True
        used_union.update(_vtk_cells_used_nodes(m.cells_dict).tolist())
        if _get_possible_point_ids(m) is not None:
            id_field_present = True

    if N_inconsistent and strict_point_count and not id_field_present:
        raise RuntimeError(
            "Point count differs across VTU frames and no common point-ID array was found.\n"
            "Either export with consistent points per frame or set strict_point_count=False."
        )

    # Keep mask (either all points or only those referenced by any cell across frames)
    if prune_unused_points and len(used_union) > 0:
        keep_idx = np.array(sorted(used_union), dtype=np.int64)
    else:
        keep_idx = np.arange(N0, dtype=np.int64)

    # Build edge_index from FIRST frame then reindex to kept nodes
    edges0 = _vtk_cells_to_edges(first_mesh.cells_dict)
    if edges0.size > 0:
        old_to_new = -np.ones(N0, dtype=np.int64)
        old_to_new[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)
        e0 = old_to_new[edges0]
        edge_index = e0
        valid = np.all(edge_index >= 0, axis=0)
        edge_index = edge_index[:, valid]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    # --- X construction from frame 0 ---
    pts0 = np.asarray(first_mesh.points, dtype=np.float64)
    pos_full = pts0[:, :3] if pts0.shape[1] >= 3 else np.pad(pts0, ((0,0),(0,3-pts0.shape[1])))
    pos = pos_full[keep_idx]

    V0 = _get_point_array(first_mesh, ["V", "Velocity", "Spatial_velocity"])
    V0 = _to_vec3(V0) if V0 is not None else np.zeros((N0, 3), dtype=np.float64)
    V0 = V0[keep_idx]

    thickness = None
    thickness_sections = {}
    if run_src_dir:
        inp = _find_file(run_src_dir, ["*.inp", "*.INP"])
        if inp:
            t_uniform, t_map = _parse_uniform_shell_thickness(inp)
            thickness_sections = t_map
            if t_uniform is not None:
                thickness = np.full((keep_idx.shape[0], 1), float(t_uniform), dtype=np.float64)

    x_parts = [pos, V0]
    if thickness is not None:
        x_parts.append(thickness)
    x = np.hstack(x_parts).astype(np.float64)
    x_columns = ["X","Y","Z","Vx","Vy","Vz"] + (["thickness"] if thickness is not None else [])

    # --- Y time series ---
    series_per_frame = []
    times = []
    y_columns = None
    first_nonzero = {"U": None, "S": None, "LE": None, "PE": None}

    for idx, (t, relpath) in enumerate(ts_files):
        m = meshio.read(os.path.join(vtu_run_dir, relpath))
        if strict_point_count and m.points.shape[0] != N0:
            raise RuntimeError(f"Frame {idx} has {m.points.shape[0]} points, expected {N0}.")

        frame_feats = []

        # --- Displacement U ---
        U = _to_vec3(_get_point_array(m, ["U", "Displacement", "Spatial_displacement"]))
        if U is None:
            U = _to_vec3(_get_cell_array_as_nodes(
                m, ["U", "Displacement", "Spatial_displacement"], N0, allow_cell_avg
            ))

        if U is None:
            if debug:
                print(f"[WARN] Frame {idx}: no displacement field found. Filling zeros.")
            U = np.zeros((N0, 3), dtype=np.float64)
        else:
            if debug and idx == (len(ts_files) - 1):
                print(f"[DEBUG] U frame0 raw: shape={U.shape}, min={U.min()}, max={U.max()}")

        U = U[keep_idx]
        frame_feats.append(U)



        # --- Stress S ---
        if "S" in fields_y:
            S = _get_point_array(m, ["S", "Stress"])
            if S is None:
                S = _get_cell_array_as_nodes(m, ["S", "Stress"], N0, allow_cell_avg)

            if S is None:
                if debug:
                    print(f"[WARN] Frame {idx}: no stress field found. Filling zeros.")
                S = np.zeros((N0, 6), dtype=np.float64)
            else:
                if debug and idx == (len(ts_files) - 1):
                    print(f"[DEBUG] S frame0 raw (pre-voigt): shape={S.shape}, "
                          f"min={S.min()}, max={S.max()}")
                S = _tensor_like_to_voigt6(S)
                if debug and idx == (len(ts_files) - 1):
                    print(f"[DEBUG] S frame0 voigt6: shape={S.shape}, "
                          f"min={S.min()}, max={S.max()}")

            S = S[keep_idx]
            frame_feats.append(S)


        # --- Log strain LE ---
        if "LE" in fields_y:
            LE = _get_point_array(m, ["LE", "Logarithmic_strain"])
            if LE is None:
                LE = _get_cell_array_as_nodes(m, ["LE", "Logarithmic_strain"], N0, allow_cell_avg)

            if LE is None:
                if debug:
                    print(f"[WARN] Frame {idx}: no LE field found. Filling zeros.")
                LE = np.zeros((N0, 6), dtype=np.float64)
            else:
                if debug and idx == (len(ts_files) - 1):
                    print(f"[DEBUG] LE frame0 raw (pre-voigt): shape={LE.shape}, "
                          f"min={LE.min()}, max={LE.max()}")
                LE = _tensor_like_to_voigt6(LE)
                if debug and idx == (len(ts_files) - 1):
                    print(f"[DEBUG] LE frame0 voigt6: shape={LE.shape}, "
                          f"min={LE.min()}, max={LE.max()}")

            LE = LE[keep_idx]
            frame_feats.append(LE)


        # --- Plastic strain PE ---
        if "PE" in fields_y:
            PE = _get_point_array(m, ["PE", "Plastic_strain"])
            if PE is None:
                PE = _get_cell_array_as_nodes(m, ["PE", "Plastic_strain"], N0, allow_cell_avg)

            if PE is None:
                if debug:
                    print(f"[WARN] Frame {idx}: no PE field found. Filling zeros.")
                PE = np.zeros((N0, 6), dtype=np.float64)
            else:
                if debug and idx == (len(ts_files) - 1):
                    print(f"[DEBUG] PE frame0 raw (pre-voigt): shape={PE.shape}, "
                          f"min={PE.min()}, max={PE.max()}")
                PE = _tensor_like_to_voigt6(PE)
                if debug and idx == (len(ts_files) - 1):
                    print(f"[DEBUG] PE frame0 voigt6: shape={PE.shape}, "
                          f"min={PE.min()}, max={PE.max()}")

            PE = PE[keep_idx]
            frame_feats.append(PE)


        Yf = np.hstack(frame_feats)
        series_per_frame.append(Yf)
        times.append(float(t))

        if y_columns is None:
            y_columns = []
            y_columns += ["U_x","U_y","U_z"]
            if "S" in fields_y:  y_columns += ["S11","S22","S33","S12","S13","S23"]
            if "LE" in fields_y: y_columns += ["LE11","LE22","LE33","LE12","LE13","LE23"]
            if "PE" in fields_y: y_columns += ["PE11","PE22","PE33","PE12","PE13","PE23"]

        if first_nonzero["U"] is None and np.any(np.abs(U) > 0):
            first_nonzero["U"] = idx
        if "S" in fields_y and first_nonzero["S"] is None:
            if 'S' in locals() and np.any(np.abs(S) > 0):
                first_nonzero["S"] = idx
        if "LE" in fields_y and first_nonzero["LE"] is None:
            if 'LE' in locals() and np.any(np.abs(LE) > 0):
                first_nonzero["LE"] = idx
        if "PE" in fields_y and first_nonzero["PE"] is None:
            if 'PE' in locals() and np.any(np.abs(PE) > 0):
                first_nonzero["PE"] = idx

        if debug and idx == (len(ts_files) - 1):
            print(f"[DEBUG] Frame0 point_data keys:", sorted(m.point_data.keys()))
            print(f"[DEBUG] Frame0 cell_data  keys:", sorted(m.cell_data.keys()))

    Y = np.stack(series_per_frame, axis=1)  # [n_keep, T, C]

    # --- Node-count consistency checks ---
    n_keep = keep_idx.shape[0]
    assert pos.shape[0] == n_keep, "pos/keep mismatch"
    assert x.shape[0]   == n_keep, "x/keep mismatch"
    assert Y.shape[0]   == n_keep, "y/keep mismatch"

    # --- Boundary mask (degree==0 after pruning) ---
    deg = np.zeros(n_keep, dtype=np.int32)
    if edge_index.size > 0:
        for i in edge_index[0].tolist():
            deg[i] += 1
    boundary_mask = torch.from_numpy((deg == 0))

    # Build PyG Data
    data = Data()
    data.num_nodes = int(n_keep)
    data.pos = torch.from_numpy(pos).float()
    data.x = torch.from_numpy(x).float()
    data.edge_index = torch.from_numpy(edge_index).long()
    data.y = torch.from_numpy(Y).float()
    data.frame_times = torch.tensor(times, dtype=torch.float32)
    data.boundary_mask = boundary_mask

    # Meta/info — includes VTU folder identifiers for your matching
    old_to_new = -np.ones(N0, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(n_keep, dtype=np.int64)

    frame_files_rel = [rel for _, rel in ts_files]
    frame_files_base = [os.path.basename(rel) for rel in frame_files_rel]
    frame_index_by_basename = {bn: i for i, bn in enumerate(frame_files_base)}
    dir_vtus = glob.glob(os.path.join(vtu_run_dir, "*.vtu"))
    frame_files_base_natsorted = [os.path.basename(p) for p in _natsorted(dir_vtus)]
    last_by_time = frame_files_base[-1]
    last_by_name = frame_files_base_natsorted[-1] if frame_files_base_natsorted else last_by_time
    last_index_by_time = len(frame_files_base) - 1
    last_index_by_name = frame_index_by_basename.get(last_by_name, last_index_by_time)

    data.meta = {
        # --- identifiers you can use downstream ---
        "vtu_folder_name": vtu_folder_name,
        "vtu_folder_abs": vtu_folder_abs,
        "run_label_guess": run_label_guess,
        "pvd_path": os.path.abspath(pvd_path),
        "pvd_sha1": pvd_sha1,

        # bookkeeping
        "run_dir": os.path.abspath(run_src_dir) if run_src_dir else None,
        "n_points_frame0": int(N0),
        "n_nodes_kept": int(n_keep),
        "n_edges": int(edge_index.shape[1]),
        "n_frames": int(len(times)),
        "x_columns": x_columns,
        "y_columns": y_columns,
        "first_nonzero_frame": first_nonzero,
        "frame_files_rel": frame_files_rel,
        "frame_files_base": frame_files_base,
        "frame_index_by_basename": frame_index_by_basename,
        "frame_files_base_natsorted": frame_files_base_natsorted,
        "last_by_time": last_by_time,
        "last_by_name": last_by_name,
        "last_index_by_time": last_index_by_time,
        "last_index_by_name": last_index_by_name,
        "keep_idx": keep_idx.tolist(),
        "old_to_new_index": old_to_new.tolist(),
        "prune_unused_points": prune_unused_points,
        "strict_point_count": strict_point_count,
        "allow_cell_avg": allow_cell_avg,
    }

    pid = _get_possible_point_ids(first_mesh)
    if pid is not None:
        data.meta["point_ids_frame0"] = pid[keep_idx].tolist()
    if thickness is not None:
        data.meta["thickness_uniform"] = float(thickness[0, 0])

    # quick-access string (handy during prediction scripts)
    data.sample_id = vtu_folder_name  # <- the folder name you asked to carry forward

    if save_path:
        out_dir = os.path.dirname(os.path.abspath(save_path))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        torch.save(data, save_path)

    if print_counts:
        pruned = N0 - n_keep
        print(f"[NODES] {vtu_folder_name}: VTU(frame0)={N0} | PyG={n_keep} | pruned={pruned}")

    return data

# ------------------------- batch runner -------------------------

def batch_build_all_runs(
    vtu_root: str,
    runs_root: Optional[str] = None,
    out_root: Optional[str] = None,
    suffix_vtu: str = "_VTU",
    fields_y: Optional[List[str]] = None,
    *,
    prune_unused_points: bool = True,
    allow_cell_avg: bool = True,
    strict_point_count: bool = True,
    debug: bool = True,
) -> Dict[str, str]:
    if fields_y is None:
        fields_y = ["U", "S", "LE", "PE"]

    results = {}
    if runs_root:
        run_names = [d for d in _natsorted(os.listdir(runs_root))
                     if os.path.isdir(os.path.join(runs_root, d))]
    else:
        run_names = [d[:-len(suffix_vtu)] for d in _natsorted(os.listdir(vtu_root))
                     if d.endswith(suffix_vtu) and os.path.isdir(os.path.join(vtu_root, d))]

    for run in run_names:
        vtu_dir = os.path.join(vtu_root, run + suffix_vtu)
        if not os.path.isdir(vtu_dir):
            alt = os.path.join(vtu_root, run)
            if os.path.isdir(alt):
                vtu_dir = alt
            else:
                print(f"[SKIP] {run}: VTU dir not found")
                continue

        run_dir = os.path.join(runs_root, run) if runs_root else None
        save_path = None
        if out_root:
            os.makedirs(out_root, exist_ok=True)
            save_path = os.path.join(out_root, f"{run}.dg3")

        try:
            _ = build_graph_from_run(
                vtu_dir, run_dir, save_path=save_path, fields_y=fields_y,
                prune_unused_points=prune_unused_points,
                allow_cell_avg=allow_cell_avg,
                strict_point_count=strict_point_count,
                debug=debug,
            )
            results[run] = save_path or "<memory>"
            print(f"[OK]   {run} -> {results[run]}")
        except Exception as e:
            print(f"[FAIL] {run}: {e}")

    return results


