"""Reader prompt v2: the AI readers answer a fixed condition checklist instead of the COVID-style
'opacity' question. Strict schema, no coercion; anything else is a recorded failure."""
from __future__ import annotations

READER_CONDITIONS = {  # key -> plain description shown to the model
    "ground_glass": "ground-glass: hazy increased lung density through which vessels are still visible",
    "consolidation": "consolidation: lung airspace filled in so that vessels are hidden",
    "atelectasis": "atelectasis: collapsed or airless lung (linear, segmental or lobar)",
    "lung_nodule": "lung nodule: round focal lesion up to 3 cm",
    "emphysema": "emphysema: abnormally dark lung areas with destroyed walls",
    "bronchiectasis": "bronchiectasis: airways wider than the artery next to them or not tapering",
    "peribronchial_thickening": "peribronchial thickening: thickened airway walls",
    "interlobular_septal_thickening": "interlobular septal thickening: fine lines outlining lung lobules",
    "mosaic_attenuation": "mosaic attenuation: patchwork of darker and brighter lung regions",
    "fibrotic_sequela": "fibrotic (scarring) changes: bands, reticulation, traction, volume loss",
    "pleural_effusion": "pleural effusion: fluid between lung and chest wall",
    "pericardial_effusion": "pericardial effusion: fluid around the heart",
    "cardiomegaly": "cardiomegaly: enlarged heart",
    "arterial_calcification": "aortic wall calcification: bright calcium in the aorta wall",
    "coronary_calcification": "coronary artery calcification: calcium along the heart's arteries",
    "lymphadenopathy": "enlarged lymph nodes in the mediastinum or hila (short axis over 1 cm)",
    "hiatal_hernia": "hiatal hernia: stomach pushed up through the diaphragm",
    "medical_material": "medical material: devices, wires, stents, tubes or surgical clips",
}
STATUS = {"present", "absent", "cannot_tell"}
SIDE = {"right", "left", "both", "not_applicable"}
ASSESS = {"assessable", "limited", "not_assessable"}
TOP = {"assessability", "conditions", "limitations"}


def build_query() -> str:
    lines = "\n".join(f"- \"{k}\" = {v}" for k, v in READER_CONDITIONS.items())
    example = ",\n    ".join(f"\"{k}\": {{\"status\": \"...\", \"side\": \"...\"}}" for k in READER_CONDITIONS)
    return (
        "\n\nBased only on the visual evidence in the slices above, answer a checklist. For EACH "
        "condition below say \"present\", \"absent\" or \"cannot_tell\" (use cannot_tell when the "
        "slices do not let you decide, for example small findings between sampled slices). For a "
        "present finding give the side (\"right\", \"left\", \"both\"); otherwise \"not_applicable\". "
        "The patient's right is on the image left. Do not invent measurements. Do not use the word "
        "'opacity'.\n\nConditions:\n" + lines +
        "\n\nReturn only a JSON object with exactly these keys:\n{\n  \"assessability\": \"assessable\" | "
        "\"limited\" | \"not_assessable\",\n  \"conditions\": {\n    " + example +
        "\n  },\n  \"limitations\": [\"<limitation>\", ...]\n}"
    )


def validate_checklist(obj: object) -> list[str]:
    if not isinstance(obj, dict):
        return [f"top level is {type(obj).__name__}, expected object"]
    errs = []
    if set(obj) != TOP:
        errs.append(f"keys {sorted(obj)} != {sorted(TOP)}")
    if obj.get("assessability") not in ASSESS:
        errs.append(f"assessability={obj.get('assessability')!r}")
    c = obj.get("conditions")
    if not isinstance(c, dict):
        return errs + ["conditions is not an object"]
    missing = set(READER_CONDITIONS) - set(c)
    extra = set(c) - set(READER_CONDITIONS)
    if missing:
        errs.append(f"missing conditions: {sorted(missing)}")
    if extra:
        errs.append(f"unexpected conditions: {sorted(extra)}")
    for k, v in c.items():
        if k not in READER_CONDITIONS:
            continue
        if not isinstance(v, dict) or v.get("status") not in STATUS:
            errs.append(f"{k}: bad status {v!r}")
        elif "side" in v and v["side"] not in SIDE:
            errs.append(f"{k}: bad side {v['side']!r}")
    lim = obj.get("limitations")
    if not (isinstance(lim, list) and all(isinstance(x, str) for x in lim)):
        errs.append("limitations must be a list of strings")
    return errs


def checklist_warnings(obj: dict) -> list[str]:
    w = []
    for k, v in obj["conditions"].items():
        if v.get("status") != "present" and v.get("side") not in (None, "not_applicable"):
            w.append(f"{k}: {v['status']} but side {v['side']}")
    if obj["assessability"] == "not_assessable" and any(v["status"] != "cannot_tell" for v in obj["conditions"].values()):
        w.append("not_assessable but some conditions decided")
    if all(v["status"] == "absent" for v in obj["conditions"].values()):
        w.append("every condition absent (possible default answer)")
    return w
