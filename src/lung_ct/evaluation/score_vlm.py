"""Score structured VLM answers against reference answers, field by field.

Works for any model whose parsed output follows the lung_ct.vlm.prompting schema.
Each field gets one outcome:
  agree            model value == reference value
  disagree         model committed to a different value (incl. not_applicable when the
                   reference says opacities are present)
  indeterminate    model answered "indeterminate"
  no_valid_output  no schema-valid response for this case (failure is counted, never dropped)

Two rates are reported so abstaining is visible:
  agreement_all        agree / all cases
  agreement_committed  agree / (agree + disagree)
"""
from __future__ import annotations

FIELDS = ("opacity_presence", "laterality", "craniocaudal", "axial")
OUTCOMES = ("agree", "disagree", "indeterminate", "no_valid_output")


def _get(d: dict, field: str):
    return d.get(field) if field == "opacity_presence" else d.get("distribution", {}).get(field)


def score_case(parsed: dict | None, reference: dict) -> dict:
    ref = reference["answers"]
    out = {}
    for f in FIELDS:
        rv = _get(ref, f)
        if parsed is None:
            out[f] = {"outcome": "no_valid_output", "model": None, "reference": rv}
            continue
        mv = _get(parsed, f)
        if mv == "indeterminate":
            oc = "indeterminate"
        elif mv == rv:
            oc = "agree"
        else:
            oc = "disagree"
        out[f] = {"outcome": oc, "model": mv, "reference": rv}
    return out


def summarise(case_scores: list[dict]) -> dict:
    summ = {}
    n = len(case_scores)
    for f in FIELDS:
        counts = {o: sum(1 for c in case_scores if c[f]["outcome"] == o) for o in OUTCOMES}
        committed = counts["agree"] + counts["disagree"]
        summ[f] = {"n_cases": n, **counts,
                   "agreement_all": counts["agree"] / n if n else None,
                   "agreement_committed": counts["agree"] / committed if committed else None}
    return summ
