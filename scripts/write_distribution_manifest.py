"""Write runs/DISTRIBUTION_MANIFEST.json: every reviewer-facing pack and review page, its status and file hashes (core env).

Nothing is deleted or moved. A superseded pack stays on disk exactly as it was sent or built; the manifest says what
replaced it and why. For the corrected viewer copies the script also verifies, file by file, that the viewer code and
the embedded scan data are byte-identical to the pages they replace and that the corrected wording is in place.

Usage: python scripts/write_distribution_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.wording import banned_hits  # noqa: E402

RUNS = ROOT / "runs"
PAYLOAD = re.compile(r'<script type="application/octet-stream"[^>]*>[^<]*</script>\n?')


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files(folder: Path, pattern: str = "*.html") -> list[dict]:
    return [{"file": p.relative_to(RUNS).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(folder.glob(pattern))]


def main() -> int:
    m3, m4 = RUNS / "M3_canvas_viewer_20261122", RUNS / "M4_canvas_viewer_corrected_20261122"
    checks = []
    for new in sorted(m4.glob("viewer_*_v4c.html")):
        old = m3 / new.name.replace("_v4c.html", "_v4.html")
        a, b = old.read_text(encoding="utf-8"), new.read_text(encoding="utf-8")
        k = "const META = "
        code_b = PAYLOAD.sub("", b)
        checks.append({"corrected": new.name, "replaces": old.relative_to(RUNS).as_posix(),
                       "viewer_script_identical": a[a.index(k):] == b[b.index(k):],
                       "embedded_scan_and_outline_data_identical": "".join(PAYLOAD.findall(a)) == "".join(PAYLOAD.findall(b)),
                       "old_wording_left_in_corrected_copy": banned_hits(code_b[:code_b.index(k)]),
                       "old_wording_in_superseded_page": banned_hits(PAYLOAD.sub("", a)[:PAYLOAD.sub("", a).index(k)])[:6],
                       "says_largest_inscribed_sphere": "largest inscribed sphere diameter" in code_b, "labelled_corrected_copy": "corrected wording copy" in code_b})
        del a, b, code_b
    bad = [c for c in checks if not (c["viewer_script_identical"] and c["embedded_scan_and_outline_data_identical"] and not c["old_wording_left_in_corrected_copy"]
                                     and c["says_largest_inscribed_sphere"] and c["labelled_corrected_copy"])]
    if bad or len(checks) != 4:
        raise SystemExit(f"corrected copies failed verification: {bad or 'expected 4 pages'}")
    z = RUNS / "M4_canvas_viewer_corrected_20261122.zip"
    tmp = z.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as f:
        for p in sorted(x for x in m4.iterdir() if x.is_file()):
            f.write(p, f"{m4.name}/{p.name}")
    with zipfile.ZipFile(tmp) as f:
        assert f.testzip() is None
    if z.exists() and sha(z) == sha(tmp):      # unchanged: leave the existing archive exactly as it is
        tmp.unlink()
    else:
        tmp.replace(z)
    policy_packs = []      # policy-labelled packs (scripts/build_policy_pack.py): one checklist version, named on every page
    for info_p in sorted(RUNS.glob("M*/PACK_INFO.json")):
        info = json.loads(info_p.read_text(encoding="utf-8"))
        folder = info_p.parent
        for pg in info["pages"]:
            if sha(folder / pg["file"]) != pg["sha256"]:
                raise SystemExit(f"{folder.name}/{pg['file']}: differs from PACK_INFO.json")
        pz = RUNS / f"{folder.name}.zip"
        ptmp = pz.with_suffix(".zip.tmp")
        with zipfile.ZipFile(ptmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as f:
            for q in sorted(x for x in folder.iterdir() if x.is_file()):
                f.write(q, f"{folder.name}/{q.name}")
        with zipfile.ZipFile(ptmp) as f:
            assert f.testzip() is None
        if pz.exists() and sha(pz) == sha(ptmp):
            ptmp.unlink()
        else:
            ptmp.replace(pz)
        policy_packs.append({"id": info["pack"], "folder": folder.name, "zip": {"file": pz.name, "bytes": pz.stat().st_size, "sha256": sha(pz)},
                             "status": "built, not sent", "policy": info["policy"],
                             "what": f"2D canvas viewer ({info['viewer']}), {info['policy']} answers; open development pages. Earlier grades shown read-only with the "
                                     "policy they were made on, not filled into the form; grades saved from these pages name the policy and the pack in every row",
                             "use": "review and discussion of the four development cases under the current answer policy; NOT for independent grading (tool answers visible)",
                             "note": "Does not supersede M4 by itself: whether to send it is a project decision. M4 and earlier packs are unchanged.",
                             "files": files(folder), "pack_info": f"{folder.name}/PACK_INFO.json"})
    d1 = RUNS / "D1_dev_audit_20261122"
    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "Nothing is deleted. 'superseded' = kept on disk unchanged, not to be sent or used for new review. 'current' = what to send.",
        "viewer_packs": policy_packs + [
            {"id": "M4", "folder": "M4_canvas_viewer_corrected_20261122", "zip": {"file": z.name, "bytes": z.stat().st_size, "sha256": sha(z)}, "status": "current",
             "policy": "checklist_v3_4",
             "what": "2D canvas viewer (checklist_viewer_v4_canvas2d), checklist_v3_4 answers, corrected wording copy; open pages with the reviewer's earlier grades shown",
             "use": "review and discussion of the four development cases; NOT for independent grading (tool answers visible)", "files": files(m4), "verified_against_M3": checks},
            {"id": "M3", "folder": "M3_canvas_viewer_20261122", "zip": {"file": "M3_canvas_viewer_20261122.zip", "bytes": (RUNS / "M3_canvas_viewer_20261122.zip").stat().st_size,
                                                                         "sha256": sha(RUNS / "M3_canvas_viewer_20261122.zip")},
             "status": "superseded", "superseded_by": "M4", "policy": "checklist_v3_4",
             "reason": "measurement panel showed the pericardial 'greatest thickness' with a small / moderate / large wording (it is the largest inscribed sphere of the automatic outline, "
                       "not a caliper measurement), and several texts asserted 'lungs poorly inflated' / 'shallow breath' without an independent assessment of inspiration. "
                       "Viewer code and scan data are identical to M4.",
             "note": "This is the pack the clinician reviewer used for valid_832 on 2026-11-22 (no crashes reported).", "files": files(m3)},
            {"id": "M2", "folder": "M2_location_first_20261122", "status": "superseded", "superseded_by": "M4", "policy": "checklist_v3_4",
             "reason": "WebGL viewer (NiiVue). On integrated graphics the graphics process grew by gigabytes per window change and died; the *_v3 pages are not a confirmed fix. "
                       "Also carries the wording superseded in M3.", "files": files(RUNS / "M2_location_first_20261122")},
            {"id": "M1", "folder": "M1_meeting_pack_20261119", "zip": {"file": "M1_meeting_pack_20261119.zip"}, "status": "superseded", "superseded_by": "M4",
             "policy": "checklist_v3_2 (valid_301, valid_541), checklist_v3_3 (valid_832, valid_988)",
             "reason": "first WebGL pages (viewer v1, grading form v1); crashed on the reviewer's laptop. Three of the four authenticated grade files were made with these pages.",
             "files": files(RUNS / "M1_meeting_pack_20261119", "viewer_*.html")}],
        "review_pages": [
            {"id": "development_adjudication_package", "status": "current", "build": "dev_adjudication_package_v1",
             "policy": "checklist_v3_4 answers plus the reviewer's earlier grades made on v3_2 / v3_3 pages",
             "what": "one staged page: retained dense regions, nodule candidates and missed nodules, pericardial outline; development adjudication of previously inspected cases",
             "send_order": "send this page first; send clinician_questions.md only to be read after step 1 is locked", "files": files(d1 / "review_package")},
            {"id": "separate_review_pages", "status": "superseded", "superseded_by": "development_adjudication_package",
             "reason": "not staged: no lock before tool information; tool measurements (outline volume, equivalent-sphere diameter) were visible from the start; three separate files",
             "files": files(d1 / "candidates") + files(d1 / "retained_regions")}],
        "grades": {"authenticated_exports": "grades_received/AUTHENTICATED_EXPORTS.json", "excluded_test_generated": "test_logs/test_generated_csv_NOT_from_reviewer/PROVENANCE.json",
                   "policy_of_each_review": "every grades CSV names the policy it was made on in its column checklist_config (valid_301: checklist_v3_2, valid_541: checklist_v3_2, valid_832: checklist_v3_4, valid_988: checklist_v3_3). "
                                            "Files saved from a policy-labelled pack also carry checklist_version, pack and "
                                            "agent_current_assessment. A grade is only ever compared with the answers of the policy named in its own file."}}
    prev = RUNS / "DISTRIBUTION_MANIFEST.json"
    if prev.exists():      # 'answer_policy' (which pages show which answer policy) is recorded by hand; keep it across rebuilds
        man["answer_policy"] = json.loads(prev.read_text(encoding="utf-8")).get("answer_policy")
    (RUNS / "DISTRIBUTION_MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    for c in checks:
        print(c["corrected"], "| viewer identical:", c["viewer_script_identical"], "| data identical:", c["embedded_scan_and_outline_data_identical"],
              "| old wording left:", c["old_wording_left_in_corrected_copy"], "| old wording in M3 page:", c["old_wording_in_superseded_page"])
    print(f"zip {z.name} {z.stat().st_size / 1e6:.0f} MB; manifest -> {RUNS / 'DISTRIBUTION_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
