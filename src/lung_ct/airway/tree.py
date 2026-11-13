"""Airway centreline, branch length and tortuosity from a binary airway mask.

Our own Python implementation ("lung_ct airway tree v2"). It is NOT AirQuant and makes no
claim of equivalence to AirQuant; AirQuant (Pakzad et al. 2024) is a methodological
reference only. Not validated against an airway reference.

Libraries: nibabel (geometry, affine, RAS), scipy (connected components, EDT),
scikit-image (3D skeletonisation), Skan (skeleton graph: branch paths, path lengths,
endpoint coordinates and distances).

Definitions
-----------
Preprocessing
  * mask reoriented to closest canonical RAS (axis permutation/flip only, no interpolation)
  * largest 26-connected component kept; discarded fragments reported
  * cropped, resampled to isotropic voxels (default: smallest input spacing) with
    nearest-neighbour interpolation (labels are never linearly interpolated)
Skeleton graph (Skan)
  * scikit-image `skeletonize` (Lee 3D thinning) -> `skan.Skeleton(spacing=iso)`
  * Skan path = pixel path between two skeleton nodes (pixel degree != 2); its
    `branch_distance` (sum of pixel-to-pixel steps, mm) is reported as
    `skan_path_length_mm`
Local scale (internal only)
  * r(x) = Euclidean distance transform of the mask at node x, in mm. Used ONLY as a local
    scale for the pruning/contraction thresholds below. It is not reported, and it is not
    an airway diameter.
Pruning (iterative, until stable)
  * remove terminal Skan paths (junction-endpoint, not containing the root) with
    skan_path_length_mm < max(prune_min_mm, prune_radius_factor * r(junction))
Node contraction
  * junction-junction paths shorter than max(contract_radius_factor * r, 1.75 * iso)
    are contracted (one anatomical bifurcation split into several skeleton nodes; the
    1.75 voxel floor merges adjacent junction pixels)
  * cycles whose perimeter < loop_perimeter_factor (2*pi) * r - a loop that fits inside the
    airway's own cross-section, i.e. a hole in the mask - are contracted the same way.
    Larger cycles are kept, counted and flagged
Branch
  * after contraction, chains through nodes with exactly two edges are merged: a BRANCH runs
    between endpoints and/or bifurcations (a closed chain is reported as a loop)
Root, generation, side
  * root = most superior skeleton endpoint (RAS z; tracheal inlet); flagged if more than
    root_tolerance_mm below the top of the mask
  * generation = depth in the shortest-path (by length) tree from the root: the trachea is
    generation 0, branches from its distal bifurcation are generation 1, ...
    Branches outside that tree close a cycle and get no generation
  * side: the two generation-1 branches are labelled left/right from the RAS x coordinate
    of their distal ends (never from array index); descendants inherit the label
Branch count
  * n_tree_branches = number of branches in the shortest-path tree (root, internal and
    terminal; contracted paths, loops and cycle-closing branches excluded)
  * n_terminal_branches = tree branches ending at a skeleton endpoint
Length and tortuosity (per branch)
  * length_mm = length of the branch centreline after Gaussian smoothing of its pixel path
    (sigma = coarsest input spacing by default; ends pinned). The raw Skan pixel-path sum
    overestimates oblique paths through the voxel staircase (synthetic test: tortuosity
    1.23 for a straight tube at 0.7x0.7x1.5 mm), so it is reported but not used for
    tortuosity
  * endpoint_distance_mm = straight-line distance between the branch's two end nodes
  * tortuosity = length_mm / endpoint_distance_mm (>= 1; 1 = straight)
Not computed: diameter, tapering (later, separately validated extension), wall thickness.
Terminal branches end where the segmentation stops, not at an anatomical end; branch
visibility depends on resolution, reconstruction, inspiration and segmentation quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import nibabel as nib
import numpy as np
from scipy import ndimage

STRUCT26 = np.ones((3, 3, 3), bool)
METHOD = "lung_ct airway tree v2 (Python: nibabel, scikit-image, Skan); NOT AirQuant; unvalidated"

DEFAULTS = {"iso_mm": None, "path_smoothing_sigma_mm": None, "prune_min_mm": 2.0,
            "prune_radius_factor": 1.5, "contract_radius_factor": 1.0,
            "loop_perimeter_factor": 6.283, "max_prune_iterations": 20,
            "root_tolerance_mm": 10.0}


@dataclass
class TreeResult:
    branches: list[dict]
    summary: dict
    skeleton: np.ndarray = field(repr=False)     # bool, iso grid (canonical RAS axes)
    affine_iso: np.ndarray = field(repr=False)
    gen_by_voxel: dict = field(repr=False, default_factory=dict)
    # branch_id -> (N, 3) RAS mm centreline points, proximal -> distal (tree branches)
    branch_paths: dict = field(repr=False, default_factory=dict)


def preprocess(mask_img: nib.Nifti1Image, iso_mm: float | None = None):
    """-> (iso_mask bool, affine_iso 4x4, info). See module docstring."""
    ras = nib.as_closest_canonical(mask_img)
    m = np.asanyarray(ras.dataobj) > 0
    sp = np.sqrt((ras.affine[:3, :3] ** 2).sum(axis=0))
    vox_ml = float(np.prod(sp)) / 1000.0
    info = {"input_axcodes": "".join(nib.aff2axcodes(mask_img.affine)),
            "input_spacing_mm": [round(float(s), 4) for s in sp],
            "input_voxels": int(m.sum())}
    if not m.any():
        raise ValueError("airway mask is empty")
    lab, n = ndimage.label(m, structure=STRUCT26)
    sizes = np.bincount(lab.ravel())[1:]
    keep = int(np.argmax(sizes)) + 1
    m = lab == keep
    info.update({"components": int(n), "kept_component_voxels": int(sizes[keep - 1]),
                 "discarded_voxels": int(sizes.sum() - sizes[keep - 1]),
                 "discarded_volume_ml": round(float(sizes.sum() - sizes[keep - 1]) * vox_ml, 3),
                 "kept_volume_ml": round(float(sizes[keep - 1]) * vox_ml, 3)})

    nz = np.argwhere(m)
    lo = np.maximum(nz.min(0) - 2, 0)
    hi = np.minimum(nz.max(0) + 3, m.shape)
    crop = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    iso = float(iso_mm or sp.min())
    f = sp / iso
    out_shape = np.maximum(np.round(np.array(crop.shape) * f).astype(int), 1)
    idx = [np.clip(np.floor((np.arange(n_) + 0.5) / f_).astype(int), 0, s_ - 1)
           for n_, f_, s_ in zip(out_shape, f, crop.shape)]
    iso_mask = crop[np.ix_(*idx)]
    lab2, n2 = ndimage.label(iso_mask, structure=STRUCT26)
    if n2 > 1:
        iso_mask = lab2 == (int(np.argmax(np.bincount(lab2.ravel())[1:])) + 1)
    S = np.eye(4)                       # new index j -> old index (j + 0.5)/f - 0.5 + lo
    S[:3, :3] = np.diag(1.0 / f)
    S[:3, 3] = 0.5 / f - 0.5 + lo
    info.update({"iso_mm": round(iso, 4), "iso_shape": list(iso_mask.shape),
                 "resampling": "nearest_neighbour"})
    return iso_mask, ras.affine @ S, info


def skeletonize3d(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(mask, method="lee").astype(bool)


def skan_graph(skel: np.ndarray, iso: float) -> dict:
    """Skan skeleton graph. Nodes are skeleton pixel ids; one entry per Skan path."""
    from skan import Skeleton, summarize
    sk = Skeleton(skel, spacing=iso)
    df = summarize(sk, separator="_")
    paths = []
    for i, row in enumerate(df.itertuples(index=False)):
        paths.append({"u": int(row.node_id_src), "v": int(row.node_id_dst),
                      "pixels": [int(x) for x in sk.path(i)],
                      "skan_length_mm": float(row.branch_distance),
                      "skan_type": int(row.branch_type)})
    return {"sk": sk, "coords": np.asarray(sk.coordinates), "degrees": np.asarray(sk.degrees),
            "paths": paths}


def smoothed_length_mm(pts_vox: np.ndarray, iso_mm: float, sigma_mm: float) -> float:
    """Length of a pixel path after Gaussian smoothing along the path (ends pinned)."""
    from scipy.ndimage import gaussian_filter1d
    pts = np.asarray(pts_vox, float)
    if len(pts) > 3 and sigma_mm > 0:
        sm = gaussian_filter1d(pts, sigma=sigma_mm / iso_mm, axis=0, mode="nearest")
        sm[0], sm[-1] = pts[0], pts[-1]
        pts = sm
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() * iso_mm)


def _root(g: dict) -> int:
    ends = {n for p in g["paths"] for n in (p["u"], p["v"]) if g["degrees"][n] == 1}
    if not ends:
        raise ValueError("skeleton has no endpoints")
    return max(ends, key=lambda n: g["coords"][n][2])      # RAS axis 2 = superior


def analyze_airway_tree(mask_img: nib.Nifti1Image, params: dict | None = None) -> TreeResult:
    p = {**DEFAULTS, **(params or {})}
    mask, aff, info = preprocess(mask_img, p["iso_mm"])
    iso = info["iso_mm"]
    edt = ndimage.distance_transform_edt(mask) * iso       # internal local scale, mm
    skel = skeletonize3d(mask)

    def r_at(vox) -> float:
        v = np.clip(np.round(vox).astype(int), 0, np.array(edt.shape) - 1)
        return float(edt[tuple(v)])

    # ---- iterative pruning of short terminal paths (Skan graph rebuilt each pass)
    it = 0
    while True:
        g = skan_graph(skel, iso)
        coords, deg = g["coords"], g["degrees"]
        root = _root(g)
        remove = []
        for pth in g["paths"]:
            ends = [n for n in (pth["u"], pth["v"]) if deg[n] == 1]
            juncs = [n for n in (pth["u"], pth["v"]) if deg[n] >= 3]
            if root in ends or len(ends) != 1 or len(juncs) != 1:
                continue
            thr = max(p["prune_min_mm"], p["prune_radius_factor"] * r_at(coords[juncs[0]]))
            if pth["skan_length_mm"] < thr:
                remove.extend(x for x in pth["pixels"] if deg[x] < 3)
        if not remove or it >= p["max_prune_iterations"]:
            break
        skel[tuple(coords[remove].astype(int).T)] = False
        it += 1

    paths = g["paths"]
    sigma_mm = p["path_smoothing_sigma_mm"] or max(info["input_spacing_mm"])
    lengths = [smoothed_length_mm(coords[q["pixels"]], iso, sigma_mm) for q in paths]

    # ---- node contraction (union-find over skeleton pixel ids)
    rep = {}

    def find(n):
        rep.setdefault(n, n)
        while rep[n] != n:
            rep[n] = rep[rep[n]]
            n = rep[n]
        return n

    contracted = set()
    floor = 1.75 * iso
    for i, q in enumerate(paths):
        if deg[q["u"]] >= 3 and deg[q["v"]] >= 3 and q["u"] != q["v"]:
            thr = max(p["contract_radius_factor"] * max(r_at(coords[q["u"]]),
                                                        r_at(coords[q["v"]])), floor)
            if lengths[i] < thr:
                contracted.add(i)
                rep[find(q["u"])] = find(q["v"])

    for _ in range(200):                                    # small-loop contraction
        H = nx.MultiGraph()
        for i, q in enumerate(paths):
            if i not in contracted and find(q["u"]) != find(q["v"]):
                H.add_edge(find(q["u"]), find(q["v"]), key=i, weight=lengths[i])
        cycles = []
        for x, y in {(x, y) for x, y in H.edges() if H.number_of_edges(x, y) > 1}:
            cycles.append(([x, y], sorted(H[x][y], key=lambda k: H[x][y][k]["weight"])[:2]))
        Gs = nx.Graph()
        for x, y, k, d in H.edges(keys=True, data=True):
            if not Gs.has_edge(x, y) or d["weight"] < Gs[x][y]["weight"]:
                Gs.add_edge(x, y, weight=d["weight"], key=k)
        for cyc in nx.cycle_basis(Gs):
            cycles.append((cyc, [Gs[cyc[j]][cyc[(j + 1) % len(cyc)]]["key"]
                                 for j in range(len(cyc))]))
        hit = next(((nodes, ks) for nodes, ks in cycles
                    if sum(lengths[k] for k in ks)
                    < p["loop_perimeter_factor"] * max(r_at(coords[n]) for n in nodes)), None)
        if hit is None:
            break
        nodes, ks = hit
        contracted.update(ks)
        for n in nodes[1:]:
            rep[find(n)] = find(nodes[0])

    root_m = find(root)

    # ---- branches: merge chains through degree-2 (contracted) nodes
    segs, loops = {}, {}
    for i, q in enumerate(paths):
        if i in contracted:
            continue
        if find(q["u"]) == find(q["v"]):
            contracted.add(i)                               # self-loop after contraction
            continue
        segs[i] = {"parts": [(i, q["u"], q["v"])], "u": find(q["u"]), "v": find(q["v"])}

    def rev(sg):
        return {"parts": [(i, y, x) for i, x, y in reversed(sg["parts"])],
                "u": sg["v"], "v": sg["u"]}

    next_id = len(paths)
    changed = True
    while changed:
        changed = False
        adj = {}
        for sid, sg in segs.items():
            adj.setdefault(sg["u"], []).append(sid)
            adj.setdefault(sg["v"], []).append(sid)
        for n, sids in adj.items():
            if n == root_m or len(sids) != 2 or sids[0] == sids[1]:
                continue
            s1, s2 = segs[sids[0]], segs[sids[1]]
            s1 = s1 if s1["v"] == n else rev(s1)
            s2 = s2 if s2["u"] == n else rev(s2)
            merged = {"parts": s1["parts"] + s2["parts"], "u": s1["u"], "v": s2["v"]}
            del segs[sids[0]], segs[sids[1]]
            (loops if merged["u"] == merged["v"] else segs)[next_id] = merged
            next_id += 1
            changed = True
            break

    G = nx.MultiGraph()
    G.add_node(root_m)
    seg_len = {sid: sum(lengths[i] for i, _, _ in sg["parts"]) for sid, sg in segs.items()}
    for sid, sg in segs.items():
        G.add_edge(sg["u"], sg["v"], key=sid, weight=seg_len[sid])
    dist, sp_paths = nx.single_source_dijkstra(G, root_m)
    tree_edges = set()
    for n, pth in sp_paths.items():
        if len(pth) >= 2:
            x, y = pth[-2], pth[-1]
            tree_edges.add(min(G[x][y], key=lambda k: G[x][y][k]["weight"]))
    hop = {n: len(pth) - 1 for n, pth in sp_paths.items()}

    def world(v):
        return (aff @ np.append(np.asarray(v, float), 1.0))[:3]

    def ras(v):
        return [round(float(x), 2) for x in world(v)]

    def seg_row(sid, sg, kind_override=None):
        if dist.get(sg["v"], np.inf) < dist.get(sg["u"], np.inf):
            sg = rev(sg)
        pix = []
        for i, x, _ in sg["parts"]:
            pp = paths[i]["pixels"] if paths[i]["u"] == x else paths[i]["pixels"][::-1]
            pix.extend(pp if not pix else pp[1:])
        a0, b1 = sg["parts"][0][1], sg["parts"][-1][2]
        length = sum(lengths[i] for i, _, _ in sg["parts"])
        ep = float(np.linalg.norm(world(coords[b1]) - world(coords[a0])))
        in_tree = sid in tree_edges
        ends = {a0, b1}
        kind = kind_override or ("root" if root in ends else
                                 "terminal" if any(deg[e] == 1 for e in ends) else "internal")
        return {
            "branch_id": sid, "generation": hop.get(sg["u"]) if in_tree else None,
            "in_tree": in_tree, "kind": kind,
            "proximal_node": int(sg["u"]), "distal_node": int(sg["v"]),
            "proximal_ras_mm": ras(coords[a0]), "distal_ras_mm": ras(coords[b1]),
            "length_mm": round(length, 2),
            "endpoint_distance_mm": round(ep, 2),
            "tortuosity": round(length / ep, 4) if ep > 0 else None,
            "skan_path_length_mm": round(sum(paths[i]["skan_length_mm"]
                                             for i, _, _ in sg["parts"]), 2),
            "n_skeleton_voxels": len(pix),
            "skan_path_ids": [i for i, _, _ in sg["parts"]], "side": None,
        }, pix

    rows, gen_by_voxel, branch_paths = [], {}, {}
    for sid, sg in segs.items():
        row, pix = seg_row(sid, sg)
        rows.append(row)
        if row["in_tree"]:
            branch_paths[sid] = np.array([world(coords[x]) for x in pix])
        if row["generation"] is not None:
            for x in pix:
                gen_by_voxel[tuple(int(c) for c in coords[x])] = row["generation"]
    for sid, sg in loops.items():
        rows.append(seg_row(sid, sg, "loop")[0])
    for i in sorted(contracted):
        q = paths[i]
        rows.append({"branch_id": i, "generation": None, "in_tree": False, "kind": "contracted",
                     "proximal_node": int(find(q["u"])), "distal_node": int(find(q["v"])),
                     "proximal_ras_mm": ras(coords[q["u"]]), "distal_ras_mm": ras(coords[q["v"]]),
                     "length_mm": round(lengths[i], 2), "endpoint_distance_mm": None,
                     "tortuosity": None, "skan_path_length_mm": round(q["skan_length_mm"], 2),
                     "n_skeleton_voxels": len(q["pixels"]), "skan_path_ids": [i],
                     "side": None})

    # ---- side from the two generation-1 branches (RAS +x = patient right)
    flags = []
    g1 = [r for r in rows if r["generation"] == 1]
    if len(g1) == 2:
        T = nx.DiGraph([(r["proximal_node"], r["distal_node"]) for r in rows if r["in_tree"]])
        a, c = sorted(g1, key=lambda r: r["distal_ras_mm"][0])     # smaller x = left
        for r_, side in ((a, "left"), (c, "right")):
            r_["side"] = side
            sub = nx.descendants(T, r_["distal_node"]) | {r_["distal_node"]}
            for rr in rows:
                if rr["in_tree"] and rr["proximal_node"] in sub:
                    rr["side"] = side
    else:
        flags.append(f"{len(g1)} generation-1 branches (expected 2): side not assigned")

    top_z = float(max(world(v)[2] for v in np.argwhere(mask)[::50]))
    root_z = float(world(coords[root])[2])
    if top_z - root_z > p["root_tolerance_mm"]:
        flags.append(f"root endpoint {top_z - root_z:.1f} mm below top of mask")
    n_cycles = G.number_of_edges() - G.number_of_nodes() + \
        nx.number_connected_components(G) + len(loops)
    if n_cycles:
        flags.append(f"{n_cycles} cycle(s) remain after contraction (touching branches or "
                     "segmentation/skeleton artefact); generations use the shortest-path tree")

    tree_rows = [r for r in rows if r["in_tree"]]
    gens = [r["generation"] for r in tree_rows]
    summary = {
        "method": METHOD,
        "params": {**p, "path_smoothing_sigma_mm": sigma_mm}, "preprocess": info,
        "prune_iterations": it,
        "root": {"ras_mm": ras(coords[root]), "mask_top_z_mm": round(top_z, 2)},
        "n_tree_branches": len(tree_rows),
        "n_terminal_branches": sum(r["kind"] == "terminal" for r in tree_rows),
        "n_contracted_paths": len(contracted),
        "n_cycle_branches": sum(not r["in_tree"] and r["kind"] != "contracted" for r in rows),
        "n_cycles": int(n_cycles),
        "max_generation": max(gens) if gens else None,
        "branches_per_generation": {int(k): gens.count(k) for k in sorted(set(gens))},
        "branches_by_side": {s: sum(r["side"] == s for r in tree_rows) for s in ("left", "right")},
        "total_centreline_length_mm": round(sum(r["length_mm"] for r in rows
                                                if r["kind"] != "contracted"), 1),
        "median_tortuosity_internal": _median([r["tortuosity"] for r in tree_rows
                                               if r["kind"] == "internal" and r["tortuosity"]]),
        "flags": flags,
    }
    return TreeResult(rows, summary, skel, aff, gen_by_voxel, branch_paths)


def _median(v):
    return round(float(np.median(v)), 4) if v else None
