"""Documentation/layout-only revision v1.1 of a clinician review package (core env).

Applies to a package built by scripts/build_review_package.py. It does NOT re-render images,
re-copy CTs, rerun inference or touch any clinical judgement. It:
  1. records a SHA-256 inventory of every file before the revision
  2. reorganises the package into staged handoffs (moves, no content change):
       handoff_1_blinded/      View 1 only (CT viewers, CT copies, View 1 sheet + docs)
       handoff_2_comparison/   View 2, model-input contact sheets, View 2 sheet + docs,
                               view1_revisions.csv
       handoff_3_airway/       central-airway landmark review
       analyst_only/           build manifest, unblinding key, View 1 lock record,
                               revision log, inventories, superseded v1.0 documents
  3. rewrites reviewer documentation and data dictionaries (lock rule, pending approvals,
     not_applicable vs indeterminate, multi-value error types, omission definition)
  4. adds the verbatim model prompt to every View 2 comparison page (never View 1)
  5. adds `visible_in_model_input` to the View 2 scoring sheet (blank)
  6. updates relative links for the new layout and verifies every link resolves
  7. records a second inventory and proves every non-documentation file is byte-identical

Usage:
    python scripts/revise_review_package.py --package runs/P6_clinician_review_v1_<ts>
"""
import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVISION = "v1.1"

ERROR_TYPES = ["none", "incorrect_finding", "incorrect_laterality", "incorrect_distribution",
               "unsupported_claim", "omission", "incorrect_input_description", "other"]
VISIBLE = ["yes", "no", "uncertain", "not_checked"]

NA_VS_INDET = (
    "not_applicable = the question does not apply because of an earlier answer for the same "
    "case (e.g. laterality and distribution when opacity_presence = absent). "
    "indeterminate = the question applies but cannot be answered from the images (e.g. "
    "opacities present but their laterality cannot be decided). Consistency rules: "
    "opacity_presence = absent -> all distribution fields not_applicable; opacity_presence = "
    "present -> distribution fields must not be not_applicable; opacity_presence = "
    "indeterminate -> distribution fields indeterminate; assessability = not_assessable -> "
    "opacity_presence and distribution fields indeterminate.")
DIST_PENDING = (
    "PENDING clinician/professor approval: the operational definitions of the zones "
    "(upper/middle/lower; peripheral/central) and of 'predominant' have not been approved. "
    "Until approved, use your usual clinical interpretation and state it in comments.")
REF_PENDING = (
    "PENDING clinician/professor approval: which reference is primary for evaluating the model "
    "- the supplied COVID-19 infection annotation or the reader's own View 1 assessment - has "
    "not been decided. Both are available; judge the model output against the images and "
    "record the basis of your judgement in comments.")
MULTI_FORMAT = (
    "One or more codes from: " + " | ".join(ERROR_TYPES) + ". Multiple codes: separate with a "
    "semicolon, no spaces, any order (e.g. unsupported_claim;incorrect_laterality). 'none' must "
    "be used alone. Leave blank only if the item was not scored.")
OMISSION_DEF = (
    "An omission is a finding visible in the CT that falls within what the model was asked to "
    "report by its actual prompt (shown on each View 2 page: pulmonary findings; presence, "
    "absence or indeterminacy of pulmonary opacities; their broad distribution; limitations "
    "of coverage or assessability) and that the response does not mention. Findings outside "
    "the prompt's scope (e.g. extrapulmonary findings, aetiology, chronicity, interval change, "
    "quantitative measurements - which the prompt told the model not to provide) are not "
    "omissions; record them in comments.")
VISIBLE_DEF = (
    "For each omission (and any missed finding noted elsewhere): is the missed finding visible "
    "in the model's 85-slice input (model_input_contact_sheets/R##_model_input.png)? "
    + " | ".join(VISIBLE) + ". Use not_checked if you did not check the contact sheet. "
    "Leave blank for rows without a missed finding.")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(pkg: Path) -> dict:
    return {str(p.relative_to(pkg)).replace("\\", "/"): sha256(p)
            for p in sorted(pkg.rglob("*")) if p.is_file()}


def git_state() -> str:
    c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                       text=True).stdout.strip()
    d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                       text=True).stdout.strip()
    return c + ("+dirty" if d else "")


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def check_links(base: Path) -> list[str]:
    """Every relative href/src and every viewer image-set directory must exist."""
    bad = []
    for f in base.rglob("*.html"):
        txt = f.read_text(encoding="utf-8")
        for m in re.finditer(r"""(?:href|src)=['"]([^'"#]+)['"]""", txt):
            t = m.group(1)
            if t.startswith(("data:", "http:", "https:")):
                continue
            if not (f.parent / t).exists():
                bad.append(f"{f.relative_to(base)} -> {t}")
        m = re.search(r"SETS=(\[\[.*?\]\])", txt)
        if m:
            for s, _ in json.loads(m.group(1)):
                if not (f.parent / s).is_dir():
                    bad.append(f"{f.relative_to(base)} -> image set {s}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", required=True, type=Path)
    args = ap.parse_args()
    pkg = ROOT / args.package
    if (pkg / "analyst_only").exists():
        raise SystemExit("package already revised (analyst_only/ exists)")
    commit = git_state()
    if commit.endswith("+dirty"):
        raise SystemExit("commit the revision script first: working tree is dirty")

    manifest = json.loads((pkg / "package_manifest.json").read_text())
    report = json.loads((ROOT / manifest["inputs"]["report"] / "run_manifest.json").read_text())
    vlm_path = report["provenance"]["inputs"]["vlm_run"]["path"]
    vlm = json.loads((ROOT / vlm_path / "run_manifest.json").read_text())
    prompt = vlm["config"]["prompt"]
    prompt_sha = vlm["provenance"]["prompt_sha256"]
    rep_cfg = vlm["config"]["representation"]

    inv_before = inventory(pkg)

    h1, h2, h3, an = (pkg / d for d in ("handoff_1_blinded", "handoff_2_comparison",
                                        "handoff_3_airway", "analyst_only"))
    for d in (h1, h2, h3, an, an / "superseded_v1.0"):
        d.mkdir()
    moves = {"view1_independent": h1, "view2_comparison": h2,
             "model_input_contact_sheets": h2, "central_airway_review": h3}
    for name, dest in moves.items():
        shutil.move(str(pkg / name), str(dest / name))
    shutil.move(str(pkg / "package_manifest.json"), str(an / "package_manifest.json"))
    shutil.move(str(h2 / "view2_comparison" / "unblinding_key.csv"), str(an / "unblinding_key.csv"))
    for name in ("README_reviewer.md", "data_dictionary.csv"):
        shutil.move(str(pkg / name), str(an / "superseded_v1.0" / name))
    sheet2 = h2 / "view2_comparison" / "scoring_view2_comparison.csv"
    shutil.copy2(sheet2, an / "superseded_v1.0" / "scoring_view2_comparison.csv")

    # ---- View 2 scoring sheet: add visible_in_model_input (blank)
    with open(sheet2, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
    i = cols.index("error_type") + 1
    cols = cols[:i] + ["visible_in_model_input"] + cols[i:]
    for r in rows:
        r["visible_in_model_input"] = ""
    write_csv(sheet2, rows, cols)

    # ---- View 1 revision log template (handoff 2) and lock record (analyst only)
    write_csv(h2 / "view2_comparison" / "view1_revisions.csv", [],
              ["review_code", "field", "view1_value_as_submitted", "revised_value", "reason",
               "stage_when_revised", "reviewer_id", "date"])
    write_csv(an / "view1_lock_record.csv", [],
              ["reviewer_id", "file_name", "sha256", "received_date", "received_by",
               "handoff_2_released_date"])

    # ---- View 2 HTML: prompt block, pending-reference note, links for the new layout
    pblock = (
        "<h3>Model prompt (verbatim, prompt " + html.escape(prompt["version"]) + ", sha256 "
        + prompt_sha[:12] + ")</h3><p><small>Shown here, never in View 1. Judge omissions "
        "against what this prompt asked for. Message layout: instruction text, then "
        f"{rep_cfg['max_slices']} images each followed by a 'SLICE n' label, then the query "
        "text; one user turn; greedy decoding.</small></p>"
        "<p><b>Instruction</b></p><pre>" + html.escape(prompt["instruction"]) + "</pre>"
        "<p><b>Query</b></p><pre>" + html.escape(prompt["query"].strip("\n")) + "</pre>")
    note = ("<div style='background:#fff8e1;border:1px solid #f9a825;padding:6px;margin:6px 0'>"
            "<b>Pending approval.</b> " + html.escape(REF_PENDING) + " " + html.escape(DIST_PENDING)
            + "</div>")
    for f in (h2 / "view2_comparison").rglob("comparison.html"):
        t = f.read_text(encoding="utf-8")
        t = t.replace("'../../view1_independent/", "'../../../handoff_1_blinded/view1_independent/")
        assert "<h3>Model output (MedGemma 1.5 4B)</h3>" in t
        t = t.replace("<h3>Automated flags", note + "<h3>Automated flags", 1)
        t = t.replace("<h3>Model output (MedGemma 1.5 4B)</h3>",
                      pblock + "<h3>Model output (MedGemma 1.5 4B)</h3>", 1)
        f.write_text(t, encoding="utf-8")
    for f in (h2 / "view2_comparison").rglob("annotation_viewer.html"):
        t = f.read_text(encoding="utf-8")
        t = t.replace('"../../view1_independent/', '"../../../handoff_1_blinded/view1_independent/')
        f.write_text(t, encoding="utf-8")
    idx = h2 / "view2_comparison" / "index.html"
    t = idx.read_text(encoding="utf-8")
    t = t.replace("<table>", note + "<table>", 1)
    idx.write_text(t, encoding="utf-8")

    # ---- data dictionaries (split so handoff 1 carries nothing about the comparison stage)
    d1 = [
        ("review_code", "neutral code; do not attempt to identify the case"),
        ("assessability", "assessable | limited | not_assessable"),
        ("opacity_presence", "present | absent | indeterminate (pulmonary opacities)"),
        ("laterality", "bilateral | right | left | not_applicable | indeterminate"),
        ("craniocaudal_distribution", "upper_predominant | middle_predominant | "
         "lower_predominant | no_predominance | not_applicable | indeterminate. " + DIST_PENDING),
        ("axial_distribution", "peripheral | central | mixed | not_applicable | indeterminate. "
         + DIST_PENDING),
        ("not_applicable vs indeterminate", NA_VS_INDET),
        ("other_findings", "free text"),
        ("image_quality_comments", "free text (coverage, motion, noise, reconstruction)"),
        ("comments", "free text"),
        ("reviewer_id / review_date / minutes_spent", "initials / YYYY-MM-DD / integer"),
    ]
    write_csv(h1 / "data_dictionary_view1.csv",
              [{"column": c, "allowed_values_or_meaning": v} for c, v in d1],
              ["column", "allowed_values_or_meaning"])
    d2 = [
        ("review_code / case_id", "pre-filled"),
        ("automated_flags", "pre-filled; exploratory rules (review_flag_rules_v1) written after "
         "inspecting this pilot, not prospectively defined criteria; not confirmed errors"),
        ("item", "one row per item: assessability, opacity_presence, laterality, "
         "craniocaudal_distribution, axial_distribution, description, limitations, "
         "unsupported_claims, omissions, overall"),
        ("model_output", "pre-filled: the model's value or text for that item"),
        ("reference_annotation", "pre-filled where applicable: supplied COVID-19 infection "
         "annotation (not an exhaustive opacity annotation). " + REF_PENDING),
        ("agreement", "agree | partially_agree | disagree | cannot_judge"),
        ("error_type", MULTI_FORMAT),
        ("visible_in_model_input", VISIBLE_DEF),
        ("unsupported_claims row", "comments: statements in the response not supported by the "
         "images"),
        ("omissions row", OMISSION_DEF),
        ("not_applicable vs indeterminate", NA_VS_INDET),
        ("distribution fields", DIST_PENDING),
        ("comments / reviewer_id / review_date", "free text / initials / YYYY-MM-DD"),
        ("view1_revisions.csv", "every change to a View 1 answer made after submitting the "
         "View 1 file: review_code, field, value as submitted, revised value, reason, stage "
         "(view2 | airway | other), reviewer_id, date. Never edit the submitted View 1 file."),
    ]
    write_csv(h2 / "data_dictionary_view2.csv",
              [{"column": c, "allowed_values_or_meaning": v} for c, v in d2],
              ["column", "allowed_values_or_meaning"])

    # ---- reviewer documentation
    n = len(list((h1 / "view1_independent").glob("R*/ct_viewer.html")))
    (h1 / "README_view1.md").write_text(f"""# Handoff 1 - independent assessment (blinded)

Research use only. {n} CT examinations under neutral codes (R01-R{n:02d}) in randomised order.

1. Open `view1_independent/index.html`. Each case shows the full examination (all axial
   slices, lung and mediastinal windows). For multiplanar review open
   `view1_independent/R##/R##_ct.nii.gz` in 3D Slicer or ITK-SNAP.
2. Record your assessment of every case in `view1_independent/scoring_view1_independent.csv`.
   Allowed values: `data_dictionary_view1.csv`.
3. **When all cases are done, save the file unchanged as
   `scoring_view1_independent_<reviewer_id>_<YYYYMMDD>.csv` and send it to the analyst.**
   The analyst records its SHA-256 before releasing the next handoff. Do not edit this file
   afterwards; any later change is recorded separately in the next stage.

Definitions
- not_applicable vs indeterminate: {NA_VS_INDET}
- Distribution fields: {DIST_PENDING}

A comparison stage follows in a separate handoff.
""", encoding="utf-8")
    (h2 / "README_view2.md").write_text(f"""# Handoff 2 - comparison

Open only after your View 1 file has been submitted. Place this folder next to
`handoff_1_blinded/` (same parent folder) so the CT links work.

1. Open `view2_comparison/index.html`. Each case page shows: the verbatim model prompt, the
   model response (verbatim and parsed), the reference infection annotation (scrollable
   overlay on the full CT), automated flags, and links to the exact 85-slice model input
   (`model_input_contact_sheets/`) and to the plain CT.
2. Score every row of `view2_comparison/scoring_view2_comparison.csv`
   (`data_dictionary_view2.csv`). All judgement columns are blank by design.
3. **Do not edit your submitted View 1 file.** If seeing the comparison changes a View 1
   answer, record it in `view2_comparison/view1_revisions.csv` (value as submitted, revised
   value, reason, stage).

Definitions and conventions
- Primary reference: {REF_PENDING}
- Distribution fields: {DIST_PENDING}
- not_applicable vs indeterminate: {NA_VS_INDET}
- error_type: {MULTI_FORMAT}
- Omissions: {OMISSION_DEF}
- visible_in_model_input: {VISIBLE_DEF}
- Automated flags (review_flag_rules_v1) are EXPLORATORY rules written after inspecting this
  pilot, not prospectively defined criteria. `input_description_error` = the model described
  its supplied input incorrectly (e.g. "single slice" when 85 images were supplied);
  `review_flag` = prioritise review, not a confirmed error (the infection annotation is not
  exhaustive); `processing_sensitivity` = output changed with an image-processor detail.
""", encoding="utf-8")
    (h3 / "README_handoff3.md").write_text(
        "# Handoff 3 - central-airway landmark review\n\nRelease after Views 1 and 2 are "
        "complete. See `central_airway_review/README.md`. The subset was selected by the "
        "analyst and does not validate the whole cohort.\n", encoding="utf-8")

    inv_after = inventory(pkg)
    # every pre-revision file must be unchanged (by content) unless it is a documented doc
    doc_changed = {"view2_comparison/scoring_view2_comparison.csv", "view2_comparison/index.html"}
    doc_changed |= {k for k in inv_before if k.endswith(("comparison.html",
                                                          "annotation_viewer.html"))}
    after_hashes = set(inv_after.values())
    unchanged = [k for k in inv_before if k not in doc_changed and inv_before[k] in after_hashes]
    missing = [k for k in inv_before if k not in doc_changed and inv_before[k] not in after_hashes]
    if missing:
        raise SystemExit(f"non-documentation files changed or lost: {missing[:5]}")
    bad_links = check_links(pkg)
    if bad_links:
        raise SystemExit(f"broken links: {bad_links[:10]}")
    leaks = [str(p.relative_to(pkg)) for p in h1.rglob("*")
             if p.is_file() and p.suffix in (".html", ".csv", ".md") and re.search(
                 r"(?i)describe the pulmonary findings|medgemma|corona|covid|infection|"
                 r"annotation|flag|case_id|unblinding", p.read_text(encoding="utf-8"))]
    if leaks:
        raise SystemExit(f"blinded handoff leaks: {leaks}")
    forbidden = [p for p in ("package_manifest.json", "unblinding_key.csv",
                             "central_airway_review") if list(h1.rglob(p))]
    if forbidden:
        raise SystemExit(f"handoff 1 contains {forbidden}")

    (an / "file_inventory_v1.0_build.json").write_text(json.dumps(inv_before, indent=1))
    (an / "file_inventory_v1.1_revised.json").write_text(json.dumps(inv_after, indent=1))
    revision = {
        "revision": REVISION, "applied_utc": datetime.now(timezone.utc).isoformat(),
        "revision_commit": commit, "build_commit": manifest["git_commit"],
        "documentation_only": True, "inference_rerun": False, "judgements_populated": False,
        "files_unchanged_by_content": len(unchanged),
        "documentation_files_modified": sorted(doc_changed),
        "prompt_shown_in_view2": {"version": prompt["version"], "sha256": prompt_sha},
        "vlm_run": vlm_path, "vlm_run_git_commit": vlm["provenance"]["git_commit"],
        "links_checked": True, "blinded_handoff_scan": "passed",
    }
    (an / "revision_v1.1.json").write_text(json.dumps(revision, indent=2))
    (an / "REVISION_LOG.md").write_text(f"""# Package revision log

## v1.0 - build
- Built by `scripts/build_review_package.py` at commit `{manifest['git_commit']}` (the
  **build commit**, recorded in `package_manifest.json`, which is preserved unchanged here).
  All images, viewers, CT copies, model-input sheets, airway figures and the original
  scoring templates were generated at this commit.
- Original documents are kept in `superseded_v1.0/`.

## v1.1 - documentation/layout revision ({revision['applied_utc'][:10]})
- Applied by `scripts/revise_review_package.py` at commit `{commit}` (the **revision
  commit**). The final repository commit recording this revision (STATUS update) follows it;
  both are listed in docs/STATUS.md.
- Why two commits: the build manifest documents how the content was generated and is not
  rewritten; later documentation changes are recorded here with their own commit, so the
  generated content stays traceable to the code that produced it.
- No inference rerun; no clinical judgement populated; no image, CT copy or viewer stack
  re-rendered. {len(unchanged)} pre-revision files are byte-identical after the revision
  (inventories: `file_inventory_v1.0_build.json`, `file_inventory_v1.1_revised.json`); paths
  changed only by the move into handoff folders.
- Modified documentation/template files: View 2 `comparison.html` (verbatim prompt, pending
  approval note, links), `annotation_viewer.html` (links), View 2 `index.html` (pending note),
  `scoring_view2_comparison.csv` (added blank `visible_in_model_input`).
- New: handoff READMEs, split data dictionaries, `view1_revisions.csv`, `view1_lock_record.csv`.
- Staged handoffs: 1 blinded (View 1 only) -> analyst records SHA-256 of the submitted View 1
  file in `view1_lock_record.csv` -> 2 comparison -> 3 airway. The build manifest, case key
  and airway subset are not in handoff 1.
- Provenance note: the MedGemma run shown in View 2 (`{vlm_path}`) was produced from
  `{vlm['provenance']['git_commit']}` (Phase 3 code before its commit); inference was not
  rerun.
""", encoding="utf-8")
    (an / "README_analyst.md").write_text("""# Analyst-only material (do not send to reviewers)

- `package_manifest.json`: v1.0 build provenance (build commit, inputs, seed, rule version).
- `unblinding_key.csv`: review code <-> case id, CT hashes.
- `view1_lock_record.csv`: when a View 1 file arrives, record its file name and SHA-256
  (`certutil -hashfile <file> SHA256` or `sha256sum`), then release handoff 2.
- `REVISION_LOG.md`, `revision_v1.1.json`, inventories, `superseded_v1.0/`.

Release order: handoff_1_blinded -> (View 1 file received and hashed) ->
handoff_2_comparison -> handoff_3_airway.
""", encoding="utf-8")
    (pkg / "PACKAGE_LAYOUT.md").write_text("""# Package layout (analyst copy)

Send each folder as a separate handoff, in order:
1. `handoff_1_blinded/` - View 1 only
2. `handoff_2_comparison/` - only after the View 1 file is received and hashed
3. `handoff_3_airway/` - after Views 1 and 2
Never send `analyst_only/`. Reviewers place handoff folders side by side in one parent folder.
""", encoding="utf-8")
    print(f"revised {pkg.name}: {len(unchanged)} files unchanged, "
          f"{len(doc_changed)} documentation files modified; links ok; handoff 1 scan ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
