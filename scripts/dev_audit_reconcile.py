"""Development audit, part 1 and 2 (core env): reconcile reviewer grades with every available tool output, and
replay a decision-policy change.

DEVELOPMENT DATA ONLY. The graded cases were opened on OPEN pages (the tool's answers were visible), so the grades
are not blinded and nothing here is a performance estimate.

Part 1 - for each graded case and condition one row with separate columns for: the reviewer's finding, confidence,
severity, location and comment (and where each location came from); the checklist answer the reviewer saw with its
exact decision provenance; HoloCT's yes/no answer, verbatim finding statements and mask volumes; specialist nodule
candidates; the report-derived label. Everything is read from run artifacts; anything that does not exist is written
as "not available". The grade files are read, never modified; their hashes and export metadata are recorded.
The reviewer-vs-checklist table is recomputed from the answers (the 'agreement' column of the files is not used).

Part 2 - replays a second checklist config (--policy-config) over the graded cases and over all development runs
and reports: incorrect negatives withdrawn, previously correct negatives left unresolved, and what changed.

Usage: python scripts/dev_audit_reconcile.py --grades runs/grades_received --holoct-run runs/P3_holoct_meeting4_20261120_182147
           --verdicts runs/A3_audit_verdicts_usability_20261120_192144/verdicts.json --nodules runs/E11_nodule_candidates_<ts>
           --base-config configs/checklist_v3_4.json --policy-config configs/checklist_v3_5.json
           --pages-file runs/test_logs/pages_test20b_v3.txt [more] [--out runs/D1_dev_audit_<ts>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.checklist import build_checklist, load_checklist_config  # noqa: E402
from lung_ct.audit.checks import TYPE_TO_CONDITION  # noqa: E402

EXCLUDE = ("pulmonary_embolism", "lung_density")   # needs contrast / overlaps ground-glass + consolidation
NA = "not available"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def authenticated_grade_files(folder: Path) -> list[Path]:
    """Grade files listed in AUTHENTICATED_EXPORTS.json with a matching SHA-256. Anything else in the folder stops the run."""
    mf = folder / "AUTHENTICATED_EXPORTS.json"
    if not mf.exists():
        raise SystemExit(f"{mf} is missing: grade files must be authenticated before they are analysed")
    listed = {x["file"]: x["sha256"] for x in json.loads(mf.read_text(encoding="utf-8"))["files"]}
    out = []
    for f in sorted(folder.glob("grades_*.csv")):
        if listed.get(f.name) != hashlib.sha256(f.read_bytes()).hexdigest():
            raise SystemExit(f"{f.name} is not an authenticated clinician export (not listed, or its content changed)")
        out.append(f)
    missing = [n for n in listed if not (folder / n).exists()]
    if missing:
        raise SystemExit(f"authenticated files missing from {folder}: {missing}")
    return out


def route(item: dict) -> str:
    """How the checklist reached its answer, from the stored basis text."""
    b, s = item.get("basis", ""), item["status"]
    if s == "cannot assess":
        return "cannot assess (scan type)"
    if b.startswith("lungs poorly inflated"):
        return "inflation gate"
    if b.startswith("measured"):
        return "measurement decides"
    if s == "absent" and "classifier low" in b:
        return "classifier fallback (low score -> absent)" + (" after an undecided measurement" if "grey zone" in b else "")
    if s == "cannot tell":
        return "abstains" + (": measurement undecided" if "grey zone" in b else ": no decisive measurement")
    return "other: " + b[:60]


def bucket(status: str) -> str:
    return {"present": "present", "absent": "absent"}.get(status, "uncertain")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grades", required=True, type=Path)
    ap.add_argument("--holoct-run", required=True, type=Path)
    ap.add_argument("--verdicts", required=True, type=Path)
    ap.add_argument("--nodules", required=True, type=Path)
    ap.add_argument("--base-config", default="configs/checklist_v3_4.json")
    ap.add_argument("--policy-config", default="configs/checklist_v3_5.json")
    ap.add_argument("--pages-file", nargs="+", required=True)
    ap.add_argument("--quarantine", default="runs/test_logs/test_generated_csv_NOT_from_reviewer")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "runs" / f"D1_dev_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    base, pol = load_checklist_config(ROOT / args.base_config), load_checklist_config(ROOT / args.policy_config)
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    runs = {}
    for pf in args.pages_file:
        for line in (ROOT / pf).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = ROOT / Path(line.strip().replace("\\", "/")).parent
                runs[json.loads((r / "evidence.json").read_text(encoding="utf-8"))["case_id"]] = r
    ver = json.loads((ROOT / args.verdicts).read_text(encoding="utf-8"))
    nod = list(csv.DictReader(open(ROOT / args.nodules / "candidates.csv", encoding="utf-8")))
    cond_types = {}
    for t, labs in TYPE_TO_CONDITION.items():
        for lab in labs:
            cond_types.setdefault(lab, set()).add(t)

    # ---------- grade files: read-only, hashed ----------
    manifest, grades = [], {}
    for f in authenticated_grade_files(ROOT / args.grades):
        rows = list(csv.DictReader(open(f, newline="", encoding="utf-8-sig")))
        r0 = rows[0]
        if any((r.get("checklist_version") or "").strip() or (r.get("pack") or "").strip() for r in rows):
            raise SystemExit(f"{f.name} was saved from a policy-labelled pack ({r0.get('checklist_version')}, pack {r0.get('pack')}). This audit was written for "
                             "grades made on v3_2 to v3_4 pages and counts 'present' as a confirmed answer; it must not read such a file. "
                             "Use scripts/collect_grades.py, which keeps policies apart and counts 'suggested' separately.")
        form = r0.get("form_version") or "grades_v1"
        m = re.match(r"(.+)_([a-z])_(\d+)$", r0["case_id"])
        manifest.append({"file": f.name, "sha256": sha256(f), "bytes": f.stat().st_size, "rows": len(rows), "reviewer": r0["reviewer"],
                         "export_time_utc": r0["saved_utc"], "case_id": r0["case_id"], "patient": m.group(1) if m else r0["case_id"],
                         "reconstruction": f"{m.group(2)}_{m.group(3)}" if m else NA, "checklist_version_shown": r0["checklist_config"],
                         "agent_run": r0["run"], "form_version": form,
                         "viewer_build": r0.get("viewer_build") or "viewer v1 (WebGL page; not recorded in a form-v1 file)",
                         "review_mode": ("staged" if r0.get("staged") == "yes" else "open page: tool answers visible while grading")
                                        + ("" if "staged" in r0 else " (form v1 has no 'staged' field; the v1 pages were open pages)"),
                         "location_fields": "recorded in the form (grades_v2)" if form == "grades_v2" else "not asked by this form; only free-text comments"})
        if r0["case_id"] in grades:
            raise SystemExit(f"two grade files for {r0['case_id']}: keep both and choose explicitly")
        grades[r0["case_id"]] = rows
    q = ROOT / args.quarantine
    history = json.loads((q / "PROVENANCE.json").read_text(encoding="utf-8")) if (q / "PROVENANCE.json").exists() else None
    auth = json.loads((ROOT / args.grades / "AUTHENTICATED_EXPORTS.json").read_text(encoding="utf-8"))

    # ---------- part 1: one row per case and condition ----------
    table = []
    cm_shown, cm_base = Counter(), Counter()
    traces = []
    for cid, rows in grades.items():
        run = runs[cid]
        shown_cfg = rows[0]["checklist_config"]
        shown = {i["key"]: i for i in json.loads((run / f"checklist_{shown_cfg}.json").read_text(encoding="utf-8"))["items"]}
        if any(i.get("headline") for i in shown.values()):
            raise SystemExit(f"{shown_cfg}: this audit script counts 'present' as a confirmed answer and was written for v3_2 to v3_5. "
                             "It must not be run on a checklist with 'suggested by segmentation' items (v3_8) without adding that category.")
        ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
        erows = {r["id"]: r for r in ev["rows"]}
        hd = ROOT / args.holoct_run / cid
        ans = {a["ct_rate_label"]: a for a in json.loads((hd / "answers.json").read_text(encoding="utf-8"))["answers"]} if (hd / "answers.json").exists() else None
        free = json.loads((hd / "findings" / "result.json").read_text(encoding="utf-8"))["text"] if (hd / "findings" / "result.json").exists() else None
        vs = [v for v in ver["verdicts"] if v["case_id"] == cid]
        labels = set(gt[cid]["labels"])
        ncand = [c for c in nod if c["case_id"] == cid]
        conds = {c["key"]: c for c in base["conditions"]}
        for r in rows:
            k = r["condition_key"]
            if k == "_overall":
                continue
            it = shown[k]
            lab = conds[k].get("ct_rate_label")
            # reviewer location and where it came from
            if r.get("form_version") == "grades_v2":
                loc_src = "form field"
                loc = r["location_where"] if r["location_status"] == "stated" else r["location_status"].replace("_", " ")
            else:
                loc, loc_src = "not asked (form v1)", "none"
            cl = re.findall(r"\b(RUL|RML|RLL|LUL|LLL)\b", r["comment"])
            comment_loc = " ".join(cl) if cl else ("see comment" if re.search(r"upper|lower|apical|apex|biapical|left|right|axillary|mediastin", r["comment"], re.I) else "")
            hstat = [v for v in vs if lab and set(v["parsed"]["types"]) & cond_types.get(lab, set())]
            table.append({
                "case_id": cid, "condition_key": k, "condition": r["condition"],
                "doctor_answer": r["doctor_answer"], "doctor_confidence": r["confidence"], "doctor_severity": r["severity"],
                "doctor_location": loc, "doctor_location_source": loc_src, "doctor_comment": r["comment"],
                "doctor_location_from_comment": comment_loc, "doctor_location_from_comment_source": "free-text comment" if comment_loc else "",
                "checklist_version_shown": shown_cfg, "checklist_answer_shown": it.get("display", it["status"]), "checklist_status": it["status"],
                "checklist_route": route(it), "checklist_basis": it.get("basis", ""), "checklist_measured_value": it.get("measured_value", ""),
                "checklist_classifier_p": it.get("classifier_p", ""), "checklist_evidence_ids": " ".join(it.get("evidence", [])),
                "checklist_location_text": it.get("location_text") or it.get("location") or "",
                "holoct_yes_no": (ans[lab]["answer"] if ans and lab in ans else NA) if lab else "no question for this condition",
                "holoct_statements_grounded": " || ".join(v["statement"] for v in hstat) if vs else NA,
                "holoct_mask_ml": " || ".join(f"{v['measurements'].get('volume_ml', 0):.1f}" for v in hstat) if vs else NA,
                "holoct_mask_lobes": " || ".join(",".join(f"{n}:{round(100 * s)}%" for n, s in sorted((v["measurements"].get("lobe_share") or {}).items(), key=lambda t: -t[1]) if s >= 0.05) for v in hstat) if vs else NA,
                "holoct_findings_text_ungrounded": free if free is not None else NA,
                "nodule_candidates": (f"{len(ncand)} outlined; equivalent-sphere mm: " + ", ".join(f"{c['lobe']} {c['equiv_diameter_mm']}" for c in ncand) if ncand else "0 outlined") if k == "lung_nodule" else "",
                "report_label": ("yes" if lab in labels else "no") if lab else "no label for this condition",
                "agreement_column_in_file": r["agreement"]})
            if k not in EXCLUDE and r["doctor_answer"] in ("present", "absent"):
                cm_shown[(r["doctor_answer"], bucket(it["status"]))] += 1
            if k in ("hiatal_hernia", "pleural_effusion", "medical_material", "atelectasis") and r["doctor_answer"] == "present" and it["status"] == "absent":
                cp = next((x for x in ev["rows"] if x.get("group") == "classifier" and x.get("ckey") == conds[k].get("classifier_key")), None)
                traces.append({"case_id": cid, "condition": k, "shown": it.get("display"), "checklist_version": shown_cfg, "route": route(it), "basis": it["basis"],
                               "measured_value": it.get("measured_value"), "unit": conds[k].get("unit"), "present_at_least": conds[k].get("present_at_least"),
                               "absent_below": conds[k].get("absent_below"), "classifier_row": cp["id"] if cp else None,
                               "classifier_source": (ev.get("classifier") or {}).get("model"), "classifier_p": cp["value"] if cp else None,
                               "classifier_low_threshold": (ev.get("classifier") or {}).get("low_threshold", 0.3),
                               "classifier_note": cp.get("note") if cp else None, "reader_comment": r["comment"],
                               "measuring_tool_exists": bool(conds[k].get("measured_by")), "measured_by": conds[k].get("measured_by")})
    with open(out / "reconciled_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # ---------- part 2: policy replay ----------
    def replay(case_ids):
        rows_ = []
        for cid in case_ids:
            run = runs[cid]
            ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
            rd = json.loads((run / "readers.json").read_text(encoding="utf-8")).get("parsed", {}) if (run / "readers.json").exists() else {}
            cl = ev.get("classifier") or {}
            a = build_checklist(ev, rd, base, cl.get("high_threshold", 0.7), cl.get("low_threshold", 0.3))["items"]
            b = build_checklist(ev, rd, pol, cl.get("high_threshold", 0.7), cl.get("low_threshold", 0.3))["items"]
            for x, y in zip(a, b):
                rows_.append((cid, x, y))
        return rows_
    dg = {(cid, r["condition_key"]): r["doctor_answer"] for cid, rows in grades.items() for r in rows}
    rep_g = replay(list(grades))
    cm_pol = Counter()
    withdrawn, unresolved, other = [], [], []
    for cid, x, y in rep_g:
        k = x["key"]
        d = dg.get((cid, k))
        if k in EXCLUDE or d not in ("present", "absent"):
            continue
        cm_base[(d, bucket(x["status"]))] += 1
        cm_pol[(d, bucket(y["status"]))] += 1
        if x["status"] != y["status"]:
            rec = {"case_id": cid, "condition": k, "reviewer": d, "before": x.get("display", x["status"]), "after": y.get("display", y["status"]),
                   "before_basis": x["basis"], "after_basis": y["basis"]}
            (withdrawn if d == "present" and x["status"] == "absent" else unresolved if d == "absent" and x["status"] == "absent" else other).append(rec)
    rep_all = replay(sorted(runs))
    # ---- complete transition table over all development runs, with explicit inclusion rules and balance checks ----
    cond_meta = {c["key"]: c for c in base["conditions"]}
    trans = {}
    for cid, x, y in rep_all:
        t = trans.setdefault(x["key"], Counter())
        t[(x["status"], y["status"])] += 1
    trans_rows = []
    for k in [c["key"] for c in base["conditions"]]:
        t = trans[k]
        lab = cond_meta[k].get("ct_rate_label")
        row = {"condition": k, "report_label": lab or "", "in_report_label_table": bool(lab),
               "in_reader_table": k not in EXCLUDE, "why_excluded_from_reader_table": {"pulmonary_embolism": "cannot be assessed without contrast",
                                                                                      "lung_density": "overlaps ground-glass and consolidation"}.get(k, ""),
               "answers": sum(t.values()), "unchanged": sum(n for (a, b), n in t.items() if a == b),
               "absent_to_cannot_tell": t[("absent", "cannot tell")],
               "other_changes": sum(n for (a, b), n in t.items() if a != b and (a, b) != ("absent", "cannot tell"))}
        for a in ("present", "absent", "cannot tell", "cannot assess"):
            row[f"v3_4_{a.replace(' ', '_')}"] = sum(n for (x_, _), n in t.items() if x_ == a)
            row[f"v3_5_{a.replace(' ', '_')}"] = sum(n for (_, y_), n in t.items() if y_ == a)
        trans_rows.append(row)
    n_runs = len(runs)
    tot_changed = sum(r["absent_to_cannot_tell"] + r["other_changes"] for r in trans_rows)
    labelled_changed = sum(r["absent_to_cannot_tell"] + r["other_changes"] for r in trans_rows if r["in_report_label_table"])
    unlabelled = [r["condition"] for r in trans_rows if not r["in_report_label_table"]]
    unlabelled_changed = sum(r["absent_to_cannot_tell"] + r["other_changes"] for r in trans_rows if not r["in_report_label_table"])
    # balance assertions: nothing is counted twice or dropped
    assert all(r["answers"] == n_runs for r in trans_rows), "every condition must have one answer per run"
    assert sum(r["answers"] for r in trans_rows) == len(rep_all) == n_runs * len(base["conditions"])
    assert all(r["unchanged"] + r["absent_to_cannot_tell"] + r["other_changes"] == r["answers"] for r in trans_rows)
    assert all(r["other_changes"] == 0 for r in trans_rows), "the policy may only move absent -> cannot tell"
    assert tot_changed == labelled_changed + unlabelled_changed
    for r in trans_rows:   # per condition: what leaves 'absent' arrives in 'cannot tell', the other states do not move
        assert r["v3_4_absent"] - r["v3_5_absent"] == r["absent_to_cannot_tell"] == r["v3_5_cannot_tell"] - r["v3_4_cannot_tell"]
        assert r["v3_4_present"] == r["v3_5_present"] and r["v3_4_cannot_assess"] == r["v3_5_cannot_assess"]
    with open(out / "policy_transition_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trans_rows[0].keys()))
        w.writeheader()
        w.writerows(trans_rows)
    ch_all = Counter()
    lab_cm = {"base": Counter(), "policy": Counter()}
    per_cond = Counter()
    n_label_answers = 0
    for cid, x, y in rep_all:
        if x["status"] != y["status"]:
            ch_all[(x["status"], y["status"])] += 1
            per_cond[x["key"]] += 1
        lab = x.get("ct_rate_label")
        if lab and cid in gt:
            n_label_answers += 1
            pos = lab in gt[cid]["labels"]
            for name, it in (("base", x), ("policy", y)):
                s = it["status"]
                lab_cm[name]["found" if s == "present" and pos else "extra" if s == "present" else "uncertain (cannot tell / cannot assess)" if s in ("cannot tell", "cannot assess")
                             else "missed (absent, label present)" if pos else "absent, label absent"] += 1
    assert sum(ch_all.values()) == tot_changed
    assert sum(lab_cm["base"].values()) == sum(lab_cm["policy"].values()) == n_label_answers
    lb, lp = lab_cm["base"], lab_cm["policy"]
    U, M, A_ = "uncertain (cannot tell / cannot assess)", "missed (absent, label present)", "absent, label absent"
    assert lp[U] - lb[U] == (lb[M] - lp[M]) + (lb[A_] - lp[A_]) == labelled_changed, "label table must balance with the labelled transitions"
    assert lb["found"] == lp["found"] and lb["extra"] == lp["extra"]
    fmt = lambda c: {f"reviewer {a} / checklist {b}": c[(a, b)] for a in ("present", "absent") for b in ("present", "absent", "uncertain")}  # noqa: E731
    res = {"created_utc": datetime.now(timezone.utc).isoformat(), "data_status": "DEVELOPMENT ONLY: open pages (not blinded), cases already inspected; no performance claim",
           "grade_files": manifest, "authentication": {"manifest": str((args.grades / "AUTHENTICATED_EXPORTS.json").as_posix()), "meaning": auth["meaning_of_authenticated"]},
           "withdrawn_comparisons": auth["withdrawn"], "test_generated_files_excluded": history,
           "abstention": {"base": {"config": base["config_version"], "uncertain_answers": cm_base[("present", "uncertain")] + cm_base[("absent", "uncertain")],
                                   "incorrect_negatives": cm_base[("present", "absent")], "of": sum(cm_base.values())},
                          "policy": {"config": pol["config_version"], "uncertain_answers": cm_pol[("present", "uncertain")] + cm_pol[("absent", "uncertain")],
                                     "incorrect_negatives": cm_pol[("present", "absent")], "of": sum(cm_pol.values())}},
           "excluded_conditions": EXCLUDE,
           "reviewer_vs_checklist_as_shown": fmt(cm_shown), "reviewer_vs_checklist_base_config": {"config": base["config_version"], **fmt(cm_base)},
           "reviewer_vs_checklist_policy_config": {"config": pol["config_version"], **fmt(cm_pol)},
           "unsupported_negative_traces": traces,
           "policy_replay_graded_cases": {"incorrect_negatives_withdrawn": withdrawn, "correct_negatives_now_unresolved": unresolved, "other_changes": other},
           "policy_replay_all_development_runs": {"runs": len(runs), "answers": len(rep_all), "status_changes": {f"{a} -> {b}": n for (a, b), n in ch_all.items()},
                                                  "changes_by_condition": dict(per_cond.most_common()),
                                                  "vs_report_labels_base": dict(lab_cm["base"]), "vs_report_labels_policy": dict(lab_cm["policy"]),
                                                  "reconciliation": {
                                                      "all_conditions": {"conditions": len(base["conditions"]), "answers": len(rep_all), "absent_to_cannot_tell": tot_changed},
                                                      "conditions_with_a_report_label": {"conditions": len(trans_rows) - len(unlabelled), "answers": n_label_answers, "absent_to_cannot_tell": labelled_changed,
                                                                                          "of_which_label_present (missed -> uncertain)": lb[M] - lp[M],
                                                                                          "of_which_label_absent (agreed-absent -> uncertain)": lb[A_] - lp[A_]},
                                                      "conditions_without_a_report_label": {"conditions": unlabelled, "answers": n_runs * len(unlabelled), "absent_to_cannot_tell": unlabelled_changed},
                                                      "identity": f"{tot_changed} = {labelled_changed} + {unlabelled_changed}",
                                                      "table": "policy_transition_table.csv", "assertions": "all balance checks passed at run time"},
                                                  "note": "Report labels are text-derived and the reviewer disagreed with many of them; shown for bookkeeping only."}}
    (out / "reconcile.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("reviewer_vs_checklist_as_shown", "reviewer_vs_checklist_base_config", "reviewer_vs_checklist_policy_config")}, indent=1))
    print("withdrawn", [(x["case_id"], x["condition"]) for x in withdrawn])
    print("unresolved", len(unresolved), Counter(x["condition"] for x in unresolved).most_common())
    print("other", [(x["case_id"], x["condition"], x["reviewer"], x["before"], x["after"]) for x in other])
    print("RECONCILIATION", json.dumps(res["policy_replay_all_development_runs"]["reconciliation"]))
    print("all runs", res["policy_replay_all_development_runs"]["status_changes"], res["policy_replay_all_development_runs"]["vs_report_labels_base"], res["policy_replay_all_development_runs"]["vs_report_labels_policy"])
    for t in traces:
        print("TRACE", t["case_id"], t["condition"], "|", t["route"], "|", t["basis"], "| classifier", t["classifier_source"], t["classifier_row"], t["classifier_p"], "| measured", t["measured_value"], t["unit"], "| tool exists", t["measuring_tool_exists"])
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
