"""Pipeline step 5: prompt for the report writer, citation checks, and a rule-based fallback.

Pure Python (json/re only) so the GPU environment can import it without numpy/nibabel.
The writer must cite evidence ids for every finding. validate_report() checks that every
cited id exists and that every number in a finding appears in one of its cited rows; it
flags, it never repairs.
"""
from __future__ import annotations

import json
import re

FIELD_LABEL = {"laterality": "Side", "craniocaudal": "Zone", "axial": "Peripheral / central",
               "opacity_presence": "Opacities"}


def _fmt_value(r: dict) -> str:
    v = r["value"]
    if isinstance(v, dict):
        v = ", ".join(f"{k} {x}" for k, x in v.items())
    return f"{v} {r['unit'] or ''}".strip() if v is not None else "not available"


def evidence_lines(ev: dict) -> list[str]:
    out = []
    for r in ev["rows"]:
        note = f"; {r['note']}" if r.get("note") and len(r["note"]) < 120 else ""
        rel = r["reliability"].split(" - ")[0]
        out.append(f"[{r['id']}] {r['name']}: {_fmt_value(r)}{note} (reliability: {rel})")
    return out


def build_prompt(ev: dict, cc: dict, readers: dict) -> str:
    rd = []
    for n, p in readers.items():
        if p is None:
            rd.append(f"- {n}: no valid answer")
        else:
            d = p.get("distribution", {})
            rd.append(f"- {n}: opacities {p.get('opacity_presence')}; side {d.get('laterality')}; "
                      f"zone {d.get('craniocaudal')}; peripheral/central {d.get('axial')}. "
                      f"Text: \"{p.get('description', '')}\"")
    conf = [f"- {i['reader']} / {i['field']}: {i['status']}"
            + (f" (reader {i.get('reader_value')}, evidence {i.get('evidence_value')})" if 'reader_value' in i else "")
            + (f" - {i['detail']}" if i.get("detail") else "")
            for i in cc["items"] if i["status"] != "agree"]
    dec = [f"- {FIELD_LABEL[k]}: {v['value']}" + (f" ({v['note']})" if v.get("note") else "")
           + f" [basis: {v['basis']}; evidence {', '.join(v['evidence'])}]" for k, v in cc["decisions"].items()]
    return (
        "You are writing a research (not clinical) summary of one chest CT for a radiologist to review.\n"
        "Use ONLY the evidence rows, reader outputs and cross-check below. Do not add findings that "
        "are not listed. Every finding must cite the evidence ids it relies on, and any number you "
        "write must be copied from a cited row. Where readers and measurements disagree, say so. "
        "If something cannot be decided, say 'cannot tell'.\n"
        "Be brief: at most 8 findings, each one sentence that summarises related rows (do NOT list "
        "every row). Classifier probabilities are uncalibrated: mention them only as 'classifier "
        "suggests', never as confirmed findings. Keep thinking short.\n\n"
        "EVIDENCE TABLE (measurements from the CT and predicted masks):\n" + "\n".join(evidence_lines(ev))
        + "\n\nAI READERS (looked at the images, no measurements):\n" + "\n".join(rd)
        + "\n\nCROSS-CHECK (disagreements only):\n" + ("\n".join(conf) or "- none")
        + "\n\nDECISIONS BY RULE (measurements decide WHERE; readers decide WHAT):\n" + "\n".join(dec)
        + "\n\nReturn only a JSON object:\n"
        "{\n  \"impression\": \"<one or two sentences>\",\n"
        "  \"findings\": [{\"text\": \"<one finding>\", \"evidence\": [\"E..\", ...]}],\n"
        "  \"disagreements\": [\"<what the reviewer should check>\"],\n"
        "  \"limitations\": [\"<limitation>\"]\n}"
    )


_NUM = re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)")


def validate_report(obj: object, ev: dict) -> dict:
    errs, flags = [], []
    if not isinstance(obj, dict):
        return {"valid": False, "errors": ["top level is not an object"], "flags": []}
    for k, typ in (("impression", str), ("findings", list), ("disagreements", list), ("limitations", list)):
        if not isinstance(obj.get(k), typ):
            errs.append(f"{k} missing or not a {typ.__name__}")
    rows = {r["id"]: r for r in ev["rows"]}
    for i, f in enumerate(obj.get("findings") or []):
        if not isinstance(f, dict) or not isinstance(f.get("text"), str):
            errs.append(f"finding {i} malformed")
            continue
        cited = [c for c in (f.get("evidence") or []) if isinstance(c, str)]
        bad = [c for c in cited if c not in rows]
        if not cited:
            flags.append({"finding": i, "flag": "no evidence cited"})
        if bad:
            flags.append({"finding": i, "flag": f"cites unknown ids {bad}"})
        pool = " ".join(json.dumps(rows[c]) for c in cited if c in rows)
        nums = [n for n in _NUM.findall(f["text"]) if not re.fullmatch(r"[0-5]", n)]
        missing = [n for n in nums if n not in pool]
        if missing:
            flags.append({"finding": i, "flag": f"numbers not found in cited rows: {missing}"})
    return {"valid": not errs, "errors": errs, "flags": flags}


def rule_based_report(ev: dict, cc: dict, readers: dict) -> dict:
    rows = {r["id"]: r for r in ev["rows"]}
    d = cc["decisions"]
    lat = d["laterality"]
    findings = [
        {"text": f"Opacities {d['opacity_presence']['value']} according to the AI readers; measurements "
                 f"find {rows['E10']['value']} mL of dense lung tissue "
                 f"(left {rows['E11']['value']} mL, right {rows['E12']['value']} mL).",
         "evidence": ["E10", "E11", "E12"]},
        {"text": f"Side: {lat.get('note') or lat['value']} ({rows['E13']['value']}% of dense tissue in the right lung).",
         "evidence": ["E13"]},
        {"text": f"Zone: {d['craniocaudal']['value'].replace('_', ' ')} "
                 f"(share by zone: {_fmt_value(rows['E14'])}).", "evidence": ["E14"]},
        {"text": f"Peripheral / central: {d['axial']['value']} ({rows['E15']['value']}% in the outer third).",
         "evidence": ["E15"]},
        {"text": f"Proxy severity score {rows['E25']['value']} of 25.", "evidence": ["E25"]},
        {"text": f"Emphysema measure: {rows['E30']['value']}% of lung below -950 HU.", "evidence": ["E30"]},
    ]
    cl = sorted((r for r in ev["rows"] if r["group"] == "classifier" and r["value"] is not None),
                key=lambda r: -r["value"])
    if cl:
        top = cl[:3]
        findings.append({"text": "Finding classifier (uncalibrated) highest scores: " + ", ".join(
            f"{r['name'].replace(' (classifier probability)', '')} {r['value']}" for r in top) + ".",
            "evidence": [r["id"] for r in top]})
    dis = [f"{i['reader']} {FIELD_LABEL.get(i['field'], i['field'])}: "
           + (f"said {i.get('reader_value')}, measurements say {i.get('evidence_value')}" if "reader_value" in i else "")
           + (f" - {i['detail']}" if i.get("detail") else "")
           for i in cc["items"] if i["status"] in ("conflict", "no_valid_output")]
    return {"impression": f"{d['opacity_presence']['value'].capitalize()} lung opacities, "
                          f"{lat.get('note') or lat['value']}, {d['craniocaudal']['value'].replace('_', ' ')}, "
                          f"{d['axial']['value']} (measurement-based; research use only).",
            "findings": findings, "disagreements": dis,
            "limitations": ["Dense-lung proxy, not a lesion segmentation; under-estimates faint disease.",
                            ("Finding classifier scores are uncalibrated zero-shot probabilities." if cl
                             else "Finding classifier (step 2) not run for this case."),
                            "Thresholds tuned on a 10-case COVID pilot that includes this case."]}
