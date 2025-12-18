# Author - Rutwik Gulakala - gulakala@iam.rwth-aachen.de
# odb2vtkParallel.py
# Parallel batch export ODB -> VTU with robust Windows worker spawning
from __future__ import print_function
import sys, os, argparse, subprocess, time
from collections import defaultdict
from odbAccess import openOdb  # Abaqus

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Parallel batch export ODB -> VTU (per-run folders).")
    # batch / launcher args
    p.add_argument("--batch-root", help="Folder containing run folders (D001, D002, ...).")
    p.add_argument("--dest-root",  help="Destination root for *_VTU outputs.")
    p.add_argument("--jobs", type=int, default=1, help="Max parallel workers (default 1).")
    p.add_argument("--odb-name", default=None, help="If set, select ODB with this exact file name in each run.")
    p.add_argument("--pick", choices=["newest","largest","first"], default="newest",
                   help="If multiple ODBs exist in a run, which to use. Default: newest.")
    p.add_argument("--abaqus-cmd", default="abaqus", help='Abaqus launcher command (default "abaqus").')
    # common export args (forwarded to workers)
    p.add_argument("--fields", default="ALL", help='CSV list or "ALL".')
    p.add_argument("--instance", default="ALL", help='Instance name or "ALL".')
    p.add_argument("--all-steps", action="store_true", help="Export all steps.")
    p.add_argument("--all-frames", action="store_true", help="Export all frames in the selected step.")
    p.add_argument("--step", default="LAST", help='Step name or index (0-based). Default: LAST')
    p.add_argument("--frame", default="last", help='Frame index or "last". Default: last')
    # worker mode (internal)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--run-dir", help=argparse.SUPPRESS)
    p.add_argument("--dest-folder", help=argparse.SUPPRESS)
    p.add_argument("--single-odb", help=argparse.SUPPRESS)  # absolute path to odb (resolved by launcher)
    return p.parse_args()

# --------- Helpers for Abaqus command detection ---------
def find_abaqus_cmd(user_value):
    if user_value and user_value.lower() != "abaqus":
        return user_value
    candidates = [
        r"C:\SIMULIA\Commands\abaqus.bat",
        r"C:\SIMULIA\EstProducts\2022\Commands\abaqus.bat",
        r"C:\SIMULIA\EstProducts\2023\Commands\abaqus.bat",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "abaqus"

def _winlong(p):
    # Avoid MAX_PATH problems on Windows when paths get long
    p = os.path.abspath(p)
    if os.name == "nt" and not p.startswith("\\\\?\\") and len(p) >= 240:
        if p.startswith("\\\\"):             # UNC -> \\?\UNC\server\share\...
            return "\\\\?\\UNC\\" + p[2:]
        return "\\\\?\\" + p
    return p

def run_worker_subprocess(abaqus_cmd, script_path, run_dir, dest_folder, odb_path, fwd_args):
    run_dir     = _winlong(run_dir)
    dest_folder = _winlong(dest_folder)
    script_path = _winlong(script_path)
    odb_path    = _winlong(odb_path)
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    def q(s):
        s = str(s)
        return '"' + s.replace('"', '\\"') + '"'

    cmd_str = " ".join(
        [q(abaqus_cmd), "python", q(script_path),
         "--worker",
         "--run-dir", q(run_dir),
         "--dest-folder", q(dest_folder),
         "--single-odb", q(odb_path)]
        + [q(x) for x in fwd_args]
    )

    use_shell = abaqus_cmd.lower().endswith(".bat")

    print("[CMD ]", cmd_str)  # helpful for debugging

    return subprocess.Popen(
        cmd_str if use_shell else cmd_str.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=use_shell,
        cwd=run_dir  # <<< important
    )

# --------- VTK basics ----------
VTK_LINE, VTK_TRI, VTK_QUAD, VTK_TET, VTK_HEX, VTK_WEDGE, VTK_PYRAMID = 3,5,9,10,12,13,14
ELTYPE_TO_VTK = {
    'C3D8':12,'C3D8R':12,'C3D8I':12,'C3D4':10,'C3D10':10,'C3D6':13,'C3D6R':13,'C3D5':14,'C3D15':13,
    'S4':9,'S4R':9,'S3':5,'S3R':5,'M3D3':5,'M3D4':9,'CPS4':9,'CPS4R':9,'CPS3':5,'CPE4':9,'CPE4R':9,'CPE3':5,
    'B31':3,'B32':3,'T2D2':3,'T3D2':3,
}
def maybe_reorder_nodes(eltype, nlabels): return nlabels

# --------- Minimal VTK writer ---------
def write_vtu(filename, points_xyz, cells_conn, cells_types, point_data, cell_data):
    type_to_n = {3:2,5:3,9:4,10:4,12:8,13:6,14:5}
    npts = len(points_xyz)
    offsets, off = [], 0
    for t in cells_types:
        off += type_to_n.get(t,0); offsets.append(off)
    def fmt_f(v): return ("%.16g" % float(v))
    with open(filename,"w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n'%(npts,len(cells_types)))
        f.write('      <Points>\n        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for x,y,z in points_xyz: f.write('          %s %s %s\n'%(fmt_f(x),fmt_f(y),fmt_f(z)))
        f.write('        </DataArray>\n      </Points>\n')
        f.write('      <Cells>\n        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i,v in enumerate(cells_conn):
            if i%16==0: f.write('          ')
            f.write('%d '%int(v))
            if i%16==15: f.write('\n')
        if len(cells_conn)%16!=0: f.write('\n')
        f.write('        </DataArray>\n        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i,v in enumerate(offsets):
            if i%16==0: f.write('          ')
            f.write('%d '%int(v))
            if i%16==15: f.write('\n')
        if len(offsets)%16!=0: f.write('\n')
        f.write('        </DataArray>\n        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for i,v in enumerate(cells_types):
            if i%16==0: f.write('          ')
            f.write('%d '%int(v))
            if i%16==15: f.write('\n')
        if len(cells_types)%16!=0: f.write('\n')
        f.write('        </DataArray>\n      </Cells>\n')
        if point_data:
            f.write('      <PointData>\n')
            for name,(n,comp,arr) in point_data.items():
                f.write('        <DataArray type="Float64" Name="%s" NumberOfComponents="%d" format="ascii">\n'%(name,comp))
                for i in range(n):
                    line = ' '.join(fmt_f(arr[i*comp+j]) for j in range(comp))
                    f.write('          %s\n'%line)
                f.write('        </DataArray>\n')
            f.write('      </PointData>\n')
        if cell_data:
            f.write('      <CellData>\n')
            for name,(n,comp,arr) in cell_data.items():
                f.write('        <DataArray type="Float64" Name="%s" NumberOfComponents="%d" format="ascii">\n'%(name,comp))
                for i in range(n):
                    line = ' '.join(fmt_f(arr[i*comp+j]) for j in range(comp))
                    f.write('          %s\n'%line)
                f.write('        </DataArray>\n')
            f.write('      </CellData>\n')
        f.write('    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')

def write_pvd(pvd_path, records):
    with open(pvd_path,"w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for t, rel in records:
            f.write('    <DataSet timestep="%.16g" group="" part="0" file="%s"/>\n'%(t, rel.replace('\\','/')))
        f.write('  </Collection>\n</VTKFile>\n')

DEFAULT_FIELD_ORDER = [
    'U','V', 'W', 'VR','UR', 'WR', 'S','LE','PE','PEEQ','RF','RM','STATUS',
    'AR','CENER','CPRESS','CSHEAR1','CSHEAR2','DMENER','EASEDEN','ECDDEN','EDMDDEN','EDT',
    'ELASE','ELCD','ELDMD','ELPD','ELSE','ELVD','EPODDEN','ESEDEN','EVDDEN','PENER','SENER','VENER'
]
def tensor_to_9(voigt):
    if len(voigt)==6:
        s11,s22,s33,s12,s13,s23 = voigt
        return [s11,s12,s13, s12,s22,s23, s13,s23,s33]
    if len(voigt)==3:
        s11,s22,s12 = voigt
        return [s11,s12,0.0, s12,s22,0.0, 0.0,0.0,0.0]
    out = list(voigt); return (out+[0.0]*9)[:9]
def vec_to_3(v):
    if len(v)>=3: return [v[0],v[1],v[2]]
    if len(v)==2: return [v[0],v[1],0.0]
    if len(v)==1: return [v[0],0.0,0.0]
    return [0.0,0.0,0.0]
def is_tensor(labels): return any(k in ' '.join(labels) for k in ['11','22','33','12','13','23'])
def is_vector(labels): return len(labels) in (2,3)
def coerce_components(name, comp, data, labels):
    n = len(data)//max(comp,1)
    if comp==1: return name,1,data
    if is_tensor(labels):
        out=[]; 
        for i in range(n): out.extend(tensor_to_9(data[i*comp:(i+1)*comp]))
        return name+"_tensor",9,out
    if is_vector(labels) or comp in (2,3):
        out=[]; 
        for i in range(n): out.extend(vec_to_3(data[i*comp:(i+1)*comp]))
        return name+"_vec",3,out
    out=[data[i*comp] for i in range(n)]; return name+"_scal",1,out

def collect_instances(odb, instance_req):
    if instance_req.upper()=="ALL":
        return list(odb.rootAssembly.instances.values())
    inst = odb.rootAssembly.instances.get(instance_req)
    if inst is None: raise ValueError("Instance %s not found"%instance_req)
    return [inst]

def build_mesh(instances):
    points=[]; node_map={}
    node_index_by_label = defaultdict(list)
    elem_index_by_label = defaultdict(list)
    for inst in instances:
        for n in inst.nodes:
            key=(inst.name,n.label); node_map[key]=len(points)
            xyz=list(n.coordinates)+[0.0]*(3-len(n.coordinates))
            points.append(tuple(xyz[:3]))
            node_index_by_label[n.label].append((inst.name, node_map[key]))
    cells_conn=[]; cells_types=[]; skipped=defaultdict(int); cell_idx=0
    for inst in instances:
        for el in inst.elements:
            vtk_t = ELTYPE_TO_VTK.get(el.type)
            if vtk_t is None: skipped[el.type]+=1; continue
            labels=[ node_map[(inst.name,nl)] for nl in maybe_reorder_nodes(el.type, el.connectivity) ]
            cells_conn.extend(labels); cells_types.append(vtk_t)
            elem_index_by_label[el.label].append((inst.name, cell_idx))
            cell_idx += 1
    if skipped:
        print("  NOTE skipped types:", dict(skipped))
    return points, cells_conn, cells_types, node_map, node_index_by_label, elem_index_by_label

def resolve_node_global_index(v, node_map, node_index_by_label):
    if getattr(v, "instance", None) is not None:
        return node_map.get((v.instance.name, v.nodeLabel), None)
    cands = node_index_by_label.get(v.nodeLabel, [])
    if len(cands)==1: return cands[0][1]
    return None
def resolve_element_cell_index(v, elem_index_by_label):
    if getattr(v, "instance", None) is not None:
        lst = elem_index_by_label.get(v.elementLabel, [])
        for nm, idx in lst:
            if nm == v.instance.name: return idx
        return None
    lst = elem_index_by_label.get(v.elementLabel, [])
    if len(lst)==1: return lst[0][1]
    return None

def nodal_field(field, node_map, node_index_by_label, npts):
    tmp={}; comp=None
    for v in field.values:
        gi = resolve_node_global_index(v, node_map, node_index_by_label)
        if gi is None: continue
        arr = list(v.data) if hasattr(v.data,'__len__') else [float(v.data)]
        comp=len(arr); tmp[gi]=[float(a) for a in arr]
    comp=comp or 1
    out=[[0.0]*comp for _ in range(npts)]
    for gi,arr in tmp.items(): out[gi]=arr
    flat=[c for row in out for c in row]
    return comp, flat

def avg_to_nodes_from_element_nodal(field, node_map, node_index_by_label, npts):
    sums={}; cnt=defaultdict(int); comp=None
    for v in field.values:
        gi = resolve_node_global_index(v, node_map, node_index_by_label)
        if gi is None: continue
        arr=list(v.data) if hasattr(v.data,'__len__') else [float(v.data)]
        comp = comp or len(arr)
        if gi not in sums: sums[gi]=[0.0]*len(arr)
        for i,a in enumerate(arr): sums[gi][i]+=float(a)
        cnt[gi]+=1
    comp=comp or 1
    out=[[0.0]*comp for _ in range(npts)]
    for gi,s in sums.items():
        c=float(cnt[gi]); out[gi]=[val/c for val in s]
    flat=[c for row in out for c in row]
    return comp, flat

def centroid_field(field, elem_index_by_label, nCells, is_centroid=False):
    comp=None; out=[None]*nCells
    sums={}; cnt=defaultdict(int)
    for v in field.values:
        ci = resolve_element_cell_index(v, elem_index_by_label)
        if ci is None: continue
        arr=list(v.data) if hasattr(v.data,'__len__') else [float(v.data)]
        comp = comp or len(arr)
        pos_str = getattr(getattr(v, "position", None), "name", None)
        if is_centroid or pos_str == 'CENTROID':
            out[ci]=[float(a) for a in arr]
        else:
            if ci not in sums: sums[ci]=[0.0]*len(arr)
            for i,a in enumerate(arr): sums[ci][i]+=float(a)
            cnt[ci]+=1
    if not is_centroid:
        for ci,s in sums.items():
            c=float(cnt[ci]) if cnt[ci] else 1.0
            out[ci]=[val/c for val in s]
    comp=comp or 1
    for i in range(len(out)):
        if out[i] is None: out[i]=[0.0]*comp
    flat=[c for row in out for c in row]
    return comp, flat

def coerce_and_pack(name, field, comp, arr, n_pts, n_cells, point=True):
    labels = getattr(field, 'componentLabels', [])
    nm, c2, arr2 = coerce_components(name, comp, arr, labels)
    return (nm, (n_pts if point else n_cells, c2, arr2))

def extract_fields(frame, n_pts, n_cells, maps, requested):
    node_map, node_index_by_label, elem_index_by_label = maps
    fo = frame.fieldOutputs
    point_data={}; cell_data={}
    wanted = DEFAULT_FIELD_ORDER if requested.upper()=="ALL" else [s.strip() for s in requested.split(',') if s.strip()]
    for key in wanted:
        if key not in fo:
            print("    Skip: '%s' not present" % key); 
            continue
        fld = fo[key]
        locs=set()
        for v in getattr(fld, 'values', []):
            pos_str = getattr(getattr(v, "position", None), "name", None)
            if pos_str: locs.add(pos_str)
        handled=False
        if 'NODAL' in locs:
            comp,arr = nodal_field(fld, node_map, node_index_by_label, n_pts)
            nm,(n,c,a) = coerce_and_pack(key, fld, comp, arr, n_pts, n_cells, point=True)
            point_data[nm]=(n,c,a); handled=True
        if (not handled) and ('ELEMENT_NODAL' in locs):
            comp,arr = avg_to_nodes_from_element_nodal(fld, node_map, node_index_by_label, n_pts)
            nm,(n,c,a) = coerce_and_pack(key, fld, comp, arr, n_pts, n_cells, point=True)
            point_data[nm]=(n,c,a); handled=True
        if not handled:
            is_centroid = ('CENTROID' in locs) and ('INTEGRATION_POINT' not in locs)
            comp,arr = centroid_field(fld, elem_index_by_label, n_cells, is_centroid=is_centroid)
            nm,(n,c,a) = coerce_and_pack(key, fld, comp, arr, n_pts, n_cells, point=False)
            cell_data[nm]=(n,c,a)
    return point_data, cell_data

def pick_step(odb, step_req):
    if step_req=="LAST": return list(odb.steps.values())[-1]
    if step_req.isdigit(): return list(odb.steps.values())[int(step_req)]
    return odb.steps[step_req]

def ensure_dir(p):
    if not os.path.isdir(p): os.makedirs(p)

def find_odb_in_folder(run_dir, name_hint=None, pick="newest"):
    candidates=[]
    for root,dirs,files in os.walk(run_dir):
        for fn in files:
            if not fn.lower().endswith(".odb"): continue
            if name_hint and fn != name_hint: continue
            path=os.path.join(root,fn)
            try:
                size=os.path.getsize(path); mtime=os.path.getmtime(path)
            except Exception:
                size=0; mtime=0.0
            candidates.append((path,size,mtime))
    if not candidates: return None
    if pick=="first":   return candidates[0][0]
    if pick=="largest": return sorted(candidates, key=lambda x: (-x[1], -x[2]))[0][0]
    return sorted(candidates, key=lambda x: (-x[2], -x[1]))[0][0]  # newest

# --------- Per-ODB export (worker logic) ----------
def export_one_odb(odb_path, dest_folder, args):
    print(" Opening:", odb_path)
    odb = openOdb(odb_path, readOnly=True)
    instances = collect_instances(odb, args.instance)
    points_xyz, cells_conn, cells_types, node_map, node_index_by_label, elem_index_by_label = build_mesh(instances)
    n_pts, n_cells = len(points_xyz), len(cells_types)
    maps = (node_map, node_index_by_label, elem_index_by_label)
    print("  Mesh: %d points, %d cells"%(n_pts, n_cells))
    steps = list(odb.steps.values()) if args.all_steps else [ pick_step(odb, args.step if args.step else "LAST") ]
    records=[]
    for s_idx, step in enumerate(steps):
        frames = step.frames if (args.all_frames or args.all_steps) else \
                 [ step.frames[-1] if str(args.frame).lower()=="last" else step.frames[int(args.frame)] ]
        for f_idx, frame in enumerate(frames):
            t = float(frame.frameValue)
            vtu_name = "export_step%03d_frame%05d.vtu"%(s_idx, f_idx)
            vtu_path = os.path.join(dest_folder, vtu_name)
            point_data, cell_data = extract_fields(frame, n_pts, n_cells, maps, args.fields)
            write_vtu(vtu_path, points_xyz, cells_conn, cells_types, point_data, cell_data)
            records.append((t, vtu_name))
    records.sort(key=lambda x: x[0])
    write_pvd(os.path.join(dest_folder, "export.pvd"), records)
    odb.close()
    print(" Done:", dest_folder)

# ---------------- Launcher / Parallel orchestration ----------------
def build_forward_args(args):
    out = []
    # export options
    out += ["--fields", args.fields]
    out += ["--instance", args.instance]
    if args.all_steps:  out += ["--all-steps"]
    if args.all_frames: out += ["--all-frames"]
    out += ["--step", args.step, "--frame", str(args.frame)]
    return out

def run_worker_subprocess(abaqus_cmd, script_path, run_dir, dest_folder, odb_path, fwd_args):
    # Ensure destination exists
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    # Build the forwarded args as a single string (works with shell=True)
    def q(s):  # simple Windows-safe quoting
        s = str(s)
        if ' ' in s or '"' in s or "'" in s or s.endswith('.bat'):
            return '"' + s.replace('"', '\\"') + '"'
        return s

    # Compose the command line exactly as you'd type in cmd.exe
    # Using "python" subcommand of Abaqus, then our script in worker mode.
    cmd_str = " ".join(
        [q(abaqus_cmd), "python", q(script_path),
         "--worker",
         "--run-dir", q(run_dir),
         "--dest-folder", q(dest_folder),
         "--single-odb", q(odb_path)]
        + [q(x) for x in fwd_args]
    )

    # If it's a .bat (typical on Windows), use shell=True so cmd.exe runs it.
    use_shell = abaqus_cmd.lower().endswith(".bat")

    # Launch
    return subprocess.Popen(
        cmd_str if use_shell else cmd_str.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=use_shell
    )


def launcher(args):
    batch_root = os.path.abspath(args.batch_root)
    dest_root  = os.path.abspath(args.dest_root)
    if not os.path.isdir(batch_root):
        print("ERROR: --batch-root not found:", batch_root); sys.exit(1)
    ensure_dir(dest_root)

    run_folders = [d for d in sorted(os.listdir(batch_root)) if os.path.isdir(os.path.join(batch_root,d))]
    if not run_folders:
        print("No subfolders in", batch_root); sys.exit(0)

    # Discover ODBs first
    work = []
    for run in run_folders:
        if run.lower().endswith("_vtu"):  # ignore output dirs
            continue
        run_dir = os.path.join(batch_root, run)
        odb_path = find_odb_in_folder(run_dir, name_hint=args.odb_name, pick=args.pick)
        if not odb_path:
            print("[SKIP] %-20s : no .odb found" % run)
            continue
        dest_folder = os.path.join(dest_root, "%s_VTU" % run)
        work.append((run, run_dir, dest_folder, odb_path))

    if not work:
        print("No runs with ODBs found. Done."); return

    print("Runs to export:", len(work))
    # Spawn up to args.jobs workers
    fwd_args = build_forward_args(args)
    script_path = os.path.abspath(__file__)
    procs = {}
    summary = {"done":0, "ok":0, "fail":0}

    i = 0
    while i < len(work) or procs:
        # Launch new workers while capacity allows
        while i < len(work) and len(procs) < max(1, args.jobs):
            run, run_dir, dest_folder, odb_path = work[i]; i += 1
            print("[LAUNCH] %-20s -> %s" % (run, dest_folder))
            p = run_worker_subprocess(args.abaqus_cmd, script_path, run_dir, dest_folder, odb_path, fwd_args)
            procs[p] = run
        # Poll existing
        time.sleep(0.5)
        finished = [p for p in procs if p.poll() is not None]
        for p in finished:
            run = procs.pop(p)
            summary["done"] += 1
            out, err = p.communicate()
            if p.returncode == 0:
                print("[ OK  ]", run)
                summary["ok"] += 1
            else:
                print("[FAIL ]", run, "| returncode:", p.returncode)
                # Show a short tail of stderr for quick diagnosis
                tail = "\n".join(err.decode("utf-8","ignore").splitlines()[-20:])
                print(tail)
                summary["fail"] += 1

    print("\n=== Parallel batch summary ===")
    for k,v in summary.items():
        print("  %s : %d" % (k, v))
        
# ---------------- MAIN ----------------
def main():
    args = parse_args()

    if args.worker:
        if not args.single_odb or not args.dest_folder or not args.run_dir:
            print("Worker missing arguments."); sys.exit(2)
        os.chdir(args.run_dir)  # <<< keep worker cwd consistent
        class W: pass
        w = W()
        w.fields = args.fields; w.instance = args.instance
        w.all_steps = args.all_steps; w.all_frames = args.all_frames
        w.step = args.step; w.frame = args.frame
        try:
            export_one_odb(os.path.abspath(args.single_odb),
                           os.path.abspath(args.dest_folder), w)
        except Exception as e:
            print("Worker error:", str(e)); sys.exit(1)
        sys.exit(0)

    # Launcher mode
    args.abaqus_cmd = find_abaqus_cmd(args.abaqus_cmd)
    launcher(args)

if __name__ == "__main__":
    main()

