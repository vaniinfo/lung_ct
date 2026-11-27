"""Development audit, parts 3 and 4 (core env): where does the dense-lung / ground-glass measurement lose tissue?

For a case the HU-selected region is rebuilt step by step with the SAME functions and settings as the checklist
measurement (lung_ct.agent.conditions), and the last step must reproduce the stored evidence value exactly:

  0 lung outline (TotalSegmentator lobes)
  1 HU band inside the lung outline, nothing excluded            "HU-selected region before exclusions"
  2 after eroding the lung outline (edge voxels dropped)
  3 after removing vessels / airways (dilated)
  4 after morphological opening (specks dropped)
  5 after removing lung next to pleural fluid (consolidation band only; that part is counted as compressed lung)
  6 after removing the posterior (dependent) band                = the measured value

Per stage: mL in total and per lobe. Each exclusion is also switched off on its own (ablation) to see what it alone
removes. Independent references, when present, are compared stage by stage: HoloCT's finding mask and the
TotalSegmentator lung_nodules outline. Aligned overlay figures show what each step removes.

DEVELOPMENT AUDIT. Ablation numbers are not rules and no threshold is proposed here.

Usage: python scripts/dev_audit_density.py --cases valid_832_a_2 valid_301_a_2 valid_541_a_2 valid_988_a_2
           --pages-file runs/test_logs/pages_test20b_v3.txt [more] --out runs/D1_dev_audit_20261122
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import load_checklist_config  # noqa: E402
from lung_ct.agent.conditions import _load, _ras_zooms, _to_ras, blocked, dependent_band, image_noise  # noqa: E402
from lung_ct.agent.evidence import LOBES  # noqa: E402

BANDS = {"consolidation": ("consolidation_hu", "E85", "E88"), "ground_glass": ("ground_glass_hu", "E80", None)}
STAGES = ["1 HU band in lung outline (no exclusion)", "2 after lung-edge erosion", "3 after vessel / airway removal", "4 after opening",
          "5 after removing lung next to pleural fluid", "6 after removing the posterior band (= measured)"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Case:
    def __init__(self, cid: str, run: Path, ck: dict):
        self.cid, self.run, self.ck = cid, run, ck
        self.ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
        self.rows = {r["id"]: r for r in self.ev["rows"]}
        qc = json.loads((run / "qc.json").read_text(encoding="utf-8"))
        self.ct_path = ROOT / qc["ct"]["file"].replace("\\", "/")
        self.ct = nib.load(str(self.ct_path))
        hu = np.asanyarray(self.ct.dataobj).astype(np.float32)
        self.seg = ROOT / "data" / "derived" / cid / "segmentation"
        lobes = {}
        for n in LOBES:
            img = nib.load(str(self.seg / "total" / f"{n}.nii.gz"))
            if img.shape != self.ct.shape or not np.allclose(img.affine, self.ct.affine, atol=1e-3):
                raise ValueError(f"{n} is not on the CT grid")
            lobes[n] = np.asanyarray(img.dataobj) > 0
        lab = np.zeros(hu.shape, np.uint8)
        for n, (_, side_label, _) in LOBES.items():
            lab[lobes[n]] = side_label
        self.hu_thr, _, self.noise = image_noise(hu, self.ct, self.seg, ck)
        aff = self.ct.affine
        self.aff = aff
        self.vml = float(abs(np.linalg.det(aff[:3, :3]))) / 1000.0
        self.zr = _ras_zooms(aff)
        self.hu_r = _to_ras(self.hu_thr, aff)
        self.lab_r = _to_ras(lab, aff)
        self.short = [v[2] for v in LOBES.values()]
        li = np.zeros(lab.shape, np.uint8)
        for i, n in enumerate(LOBES, start=1):
            li[lobes[n]] = i
        self.lobe_idx = _to_ras(li, aff)
        self.lung = self.lab_r > 0
        m = ck["measurements"]
        self.m = m
        self.core = ndimage.binary_erosion(self.lung, iterations=m["density_lung_erosion_voxels"])
        ex = np.zeros(self.lung.shape, bool)
        for r_ in m["density_exclude"]:
            e = _load(self.seg, r_, self.ct)
            if e is not None:
                ex |= _to_ras(e, aff)
        self.ex_raw = ex
        self.ex = ndimage.binary_dilation(ex, iterations=m["density_exclude_dilation_voxels"]) if ex.any() else ex
        self.dep = dependent_band(self.lab_r, m["dependent_band_fraction"])
        pe = _load(self.seg, "pleural_pericard_effusion/pleural_effusion", self.ct)
        self.pe = _to_ras(pe, aff) if pe is not None else None
        self.near_eff = None
        nmm = m.get("effusion_adjacent_mm")
        if nmm and self.pe is not None and self.pe.any():
            idx = np.argwhere(self.pe)
            pad = np.ceil(nmm / self.zr).astype(int) + 1
            lo, hi = np.maximum(idx.min(0) - pad, 0), np.minimum(idx.max(0) + pad + 1, self.pe.shape)
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            dist = ndimage.distance_transform_edt(~self.pe[sl], sampling=self.zr)
            self.near_eff = np.zeros(self.lung.shape, bool)
            self.near_eff[sl] = dist <= nmm

    def stages(self, key: str, erosion=True, vessels=True, opening=True, effusion=True, posterior=True) -> list[np.ndarray]:
        band = self.m[BANDS[key][0]]
        inband = (self.hu_r >= band[0]) & (self.hu_r < band[1])
        s1 = self.lung & inband
        s2 = (self.core if erosion else self.lung) & inband
        s3 = s2 & ~self.ex if vessels else s2
        s4 = ndimage.binary_opening(s3, iterations=self.m["density_opening_voxels"]) if (opening and self.m["density_opening_voxels"]) else s3
        s5 = s4 & ~self.near_eff if (effusion and key == "consolidation" and self.near_eff is not None) else s4
        s6 = s5 & ~self.dep if posterior else s5
        return [s1, s2, s3, s4, s5, s6]

    def per_lobe(self, mask: np.ndarray) -> dict:
        c = np.bincount(self.lobe_idx[mask], minlength=len(self.short) + 1)
        return {n: round(float(c[i + 1]) * self.vml, 1) for i, n in enumerate(self.short)}

    def ml(self, mask: np.ndarray) -> float:
        return round(float(mask.sum()) * self.vml, 1)


def reference_masks(c: Case, holoct_run: Path, verdicts: dict, want_types: set) -> list[dict]:
    out = []
    for v in verdicts["verdicts"]:
        if v["case_id"] != c.cid or not (set(v["parsed"]["types"]) & want_types):
            continue
        idx = int(v["finding_id"].split("|")[-1])
        f = holoct_run / c.cid / "findings_grounded" / f"mask_{idx:03d}.nii.gz"
        if not f.exists():
            continue
        img = nib.load(str(f))
        if img.shape != c.ct.shape or not np.allclose(img.affine, c.ct.affine, atol=1e-3):
            out.append({"source": "HoloCT", "statement": v["statement"], "error": "mask not on the CT grid"})
            continue
        out.append({"source": "HoloCT", "statement": v["statement"], "file": str(f.relative_to(ROOT)), "mask": _to_ras(np.asanyarray(img.dataobj) > 0, c.aff)})
    return out


def overlay_figure(c: Case, key: str, st: list[np.ndarray], refs: list[dict], path: Path, title: str, focus: np.ndarray | None = None) -> list[int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    target = focus if focus is not None and focus.any() else st[0]
    zs = np.where(target.any(axis=(0, 1)))[0]
    per = target.sum(axis=(0, 1))
    if zs.size == 0:
        return []
    order = np.argsort(-per)[:400]
    picks = sorted({int(order[0]), int(np.percentile(zs, 30)), int(np.percentile(zs, 70))})
    cols = ["CT (window -160..240 HU)", "lung outline", "1 HU band in outline", "removed: edge erosion", "removed: vessels / airways", "removed: opening",
            "removed: next to fluid", "removed: posterior band", "6 measured region"] + [f"{r['source']} mask" for r in refs if "mask" in r]
    lay = [None, c.lung, st[0], st[0] & ~st[1], st[1] & ~st[2], st[2] & ~st[3], st[3] & ~st[4], st[4] & ~st[5], st[5]] + [r["mask"] for r in refs if "mask" in r]
    colours = [None, "#22d3ee", "#facc15", "#f97316", "#a855f7", "#ec4899", "#3b82f6", "#94a3b8", "#ef4444"] + ["#22c55e"] * len([r for r in refs if "mask" in r])
    # crop to the lungs' bounding box in-plane
    xy = np.argwhere(c.lung.any(axis=2))
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0) + 1
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(2.5 * len(cols), 2.6 * len(picks)), squeeze=False)
    disp = lambda a, k: a[x0:x1, y0:y1, k][::-1, ::-1].T        # RAS -> radiological display: left = patient Right, top = Anterior  # noqa: E731
    asp = c.zr[1] / c.zr[0]
    for r, k in enumerate(picks):
        for j, (name, layer, col) in enumerate(zip(cols, lay, colours)):
            a = axs[r][j]
            a.imshow(disp(c.hu_r, k), cmap="gray", vmin=-160, vmax=240, aspect=asp, interpolation="nearest")
            if layer is not None:
                m = disp(layer, k)
                a.imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([col]), alpha=0.6, aspect=asp, interpolation="nearest")
                ml = float(layer[:, :, k].sum()) * c.vml
                a.set_xlabel(f"{ml:.1f} mL on this image", fontsize=7)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(name, fontsize=8)
            if j == 0:
                a.set_ylabel(f"axial k={k}\n(image {c.lung.shape[2] - k} of {c.lung.shape[2]})", fontsize=8)
    fig.suptitle(title + "  |  patient's right is on the image left; top = front", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--pages-file", nargs="+", required=True)
    ap.add_argument("--config", default="configs/checklist_v3_4.json")
    ap.add_argument("--holoct-run", default="runs/P3_holoct_meeting4_20261120_182147")
    ap.add_argument("--verdicts", default="runs/A3_audit_verdicts_usability_20261120_192144/verdicts.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    (out / "density").mkdir(parents=True, exist_ok=True)
    ck = load_checklist_config(ROOT / args.config)
    runs = {}
    for pf in args.pages_file:
        for line in (ROOT / pf).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = ROOT / Path(line.strip().replace("\\", "/")).parent
                runs[json.loads((r / "evidence.json").read_text(encoding="utf-8"))["case_id"]] = r
    ver = json.loads((ROOT / args.verdicts).read_text(encoding="utf-8"))
    hrun = ROOT / args.holoct_run
    table, res = [], {}
    for cid in args.cases:
        print(f"[{cid}]", flush=True)
        c = Case(cid, runs[cid], ck)
        rec = {"provenance": {"ct_file": str(c.ct_path.relative_to(ROOT)), "ct_sha256": sha256_file(c.ct_path), "ct_shape": list(c.ct.shape),
                              "agent_run": c.run.name, "evidence_measured_with": c.ev.get("checklist_config"), "audit_config": ck["config_version"],
                              "inflation": c.rows.get("E96", {}).get("note"), "lung_volume_ml": c.rows.get("E95", {}).get("value")}}
        sm = c.seg / "seg_manifest.json"
        if sm.exists():
            rec["provenance"]["segmentation_ct_sha256"] = json.loads(sm.read_text()).get("ct_sha256")
        ha = hrun / cid / "answers.json"
        if ha.exists():
            rec["provenance"]["holoct_ct_sha256"] = json.loads(ha.read_text(encoding="utf-8")).get("ct_sha256")
        rec["provenance"]["same_file_everywhere"] = len({rec["provenance"]["ct_sha256"], rec["provenance"].get("segmentation_ct_sha256") or rec["provenance"]["ct_sha256"],
                                                         rec["provenance"].get("holoct_ct_sha256") or rec["provenance"]["ct_sha256"]}) == 1
        rec["lung_outline_ml"] = c.ml(c.lung)
        rec["lobe_ml"] = c.per_lobe(c.lung)
        for key, (_, row_id, adj_id) in BANDS.items():
            if blocked(c.noise, key):
                rec[key] = {"error": "not measured: image too grainy"}
                continue
            st = c.stages(key)
            stored = c.rows[row_id]["value"]
            got = round(float(st[5].sum()) * c.vml)
            k = {"stored_value_ml": stored, "recomputed_final_ml": got, "reproduces_stored_value": got == stored, "stages": []}
            if adj_id:
                adj = st[3] & ~st[4]
                k["next_to_fluid_ml"] = c.ml(adj)
                k["next_to_fluid_stored_ml"] = c.rows[adj_id]["value"]
            for name, m in zip(STAGES, st):
                row = {"case_id": cid, "band": key, "stage": name, "ml": c.ml(m), **{f"ml_{n}": v for n, v in c.per_lobe(m).items()}}
                table.append(row)
                k["stages"].append({"stage": name, "ml": row["ml"], "per_lobe": c.per_lobe(m)})
            abl = {}
            for name, kw in (("no lung-edge erosion", {"erosion": False}), ("no vessel / airway removal", {"vessels": False}), ("no opening", {"opening": False}),
                             ("no fluid-proximity removal", {"effusion": False}), ("no posterior-band removal", {"posterior": False}),
                             ("no exclusion at all", {"erosion": False, "vessels": False, "opening": False, "effusion": False, "posterior": False})):
                f = c.stages(key, **kw)[5]
                abl[name] = {"ml": c.ml(f), "per_lobe": c.per_lobe(f)}
            k["ablation_final_ml_with_one_step_switched_off"] = abl
            types = {"consolidation": {"consolidation"}, "ground_glass": {"ground_glass"}}[key]
            refs = reference_masks(c, hrun, ver, types)
            k["references"] = []
            for r in refs:
                if "mask" not in r:
                    k["references"].append(r)
                    continue
                rm = r["mask"]
                band = c.m[BANDS[key][0]]
                k["references"].append({
                    "source": r["source"], "statement": r["statement"], "file": r["file"], "ml": c.ml(rm), "per_lobe": c.per_lobe(rm),
                    "inside_lung_outline_pct": round(100 * float((rm & c.lung).sum()) / max(1, int(rm.sum())), 1),
                    "in_this_hu_band_pct": round(100 * float((rm & (c.hu_r >= band[0]) & (c.hu_r < band[1])).sum()) / max(1, int(rm.sum())), 1),
                    "median_hu": round(float(np.median(c.hu_r[rm]))) if rm.any() else None,
                    "overlap_with_each_stage_ml": {name: c.ml(m & rm) for name, m in zip(STAGES, st)},
                    "lost_at": {"outside the lung outline": c.ml(rm & ~c.lung), "in outline but outside this HU band": c.ml(rm & c.lung & ~st[0]),
                                "edge erosion": c.ml(rm & st[0] & ~st[1]), "vessel / airway removal": c.ml(rm & st[1] & ~st[2]), "opening": c.ml(rm & st[2] & ~st[3]),
                                "next to pleural fluid": c.ml(rm & st[3] & ~st[4]), "posterior band": c.ml(rm & st[4] & ~st[5]), "kept in the measured region": c.ml(rm & st[5])}})
            nf = c.seg / "lung_nodules" / "lung_nodules.nii.gz"
            if key == "consolidation" and nf.exists():
                nm = _to_ras(np.asanyarray(nib.load(str(nf)).dataobj) > 0, c.aff)
                lab, n = ndimage.label(nm, structure=np.ones((3, 3, 3)))
                if n:
                    sizes = np.bincount(lab.ravel())[1:]
                    big = lab == (int(sizes.argmax()) + 1)
                    k["largest_lung_nodules_component"] = {"ml": c.ml(big), "per_lobe": c.per_lobe(big), "inside_lung_outline_pct": round(100 * float((big & c.lung).sum()) / int(big.sum()), 1),
                                                           "kept_in_measured_region_ml": c.ml(big & st[5]), "in_band_in_outline_ml": c.ml(big & st[0])}
                    refs = refs + [{"source": "lung_nodules largest piece", "mask": big}]
            focus = next((r["mask"] for r in refs if "mask" in r), None)
            pic = out / "density" / f"{cid}_{key}_stages.png"
            k["overlay_slices_k"] = overlay_figure(c, key, st, refs, pic, f"{cid} - {key.replace('_', '-')} band {c.m[BANDS[key][0]]} HU: what each step removes", focus)
            k["overlay_file"] = str(pic.relative_to(ROOT))
            if key == "ground_glass":            # what kind of tissue is the measured ground-glass-range region made of?
                f = st[5]
                lab, n = ndimage.label(f)
                sizes = np.bincount(lab.ravel())[1:] * c.vml if n else np.array([])
                edge = c.lung & ~ndimage.binary_erosion(c.lung, iterations=int(round(5 / float(c.zr.min()))))
                near_v = ndimage.binary_dilation(c.ex_raw, iterations=int(round(3 / float(c.zr.min())))) if c.ex_raw.any() else np.zeros_like(f)
                tot = max(1, int(f.sum()))
                k["composition_of_measured_region"] = {
                    "pieces": int(n), "pieces_under_0.1_ml": int((sizes < 0.1).sum()), "ml_in_pieces_under_0.1_ml": round(float(sizes[sizes < 0.1].sum()), 1),
                    "largest_piece_ml": round(float(sizes.max()), 1) if n else 0, "ml_in_5_largest_pieces": round(float(np.sort(sizes)[-5:].sum()), 1) if n else 0,
                    "pct_within_5mm_of_lung_surface": round(100 * float((f & edge).sum()) / tot, 1), "pct_within_3mm_of_vessel_or_airway_outline": round(100 * float((f & near_v).sum()) / tot, 1),
                    "pct_in_lowest_third_of_lung_height": None, "median_hu": round(float(np.median(c.hu_r[f]))) if f.any() else None}
                zs = np.where(c.lung.any(axis=(0, 1)))[0]
                zc = zs.min() + (zs.max() - zs.min()) / 3
                k["composition_of_measured_region"]["pct_in_lowest_third_of_lung_height"] = round(100 * float(f[:, :, :int(zc)].sum()) / tot, 1)
            rec[key] = k
            print(f"  {key}: stored {stored} mL, recomputed {got} mL; stages {[s['ml'] for s in k['stages']]}", flush=True)
        res[cid] = rec
        del c
    with open(out / "density" / "stage_volumes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    (out / "density" / "density_audit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"-> {out / 'density'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
