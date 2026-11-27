"""Prompt test for COLIPRI (colipri env): the same scans scored with several frozen prompt sets, compared by ranking.

Follows docs/PROMPT_TEST_TEMPLATE.md; the plan of the first use is docs/prompt_tests/PT01_colipri_prompts_PLAN.md.
Everything runs locally. Each scan is encoded ONCE; prompt sets differ only in the text embeddings, so the scores of
set A are the scores of scripts/agent_classify.py (same model call, same prompts).

  score for "present" = softmax over (negative, positive) of image_embedding . text_embedding / temperature

Prompt sets come from a config (default configs/colipri_prompt_sets_v1.json):
  A / B / C  lists of positive and negative sentences per condition; several sentences are averaged
  D          "native": mean embedding of report Findings sections of OTHER patients (rule in the config)
No cut-off is used. Reference: CT-RATE report labels in data/raw/ct_rate/cases.json.

Usage:
  environments/colipri/Scripts/python.exe scripts/prompt_test_colipri.py --prompt-sets configs/colipri_prompt_sets_v1.json
  ... --reuse runs/PT01_colipri_prompts_<time>     (re-analyse stored embeddings; no model is loaded)
Writes runs/PT01_colipri_prompts_<time>/{image_embeddings.npz,text_embeddings.npz,scores.csv,native_reports.json,summary.json,summary.md,run_manifest.json}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "colipri" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patient_of(scan: str) -> str:
    return scan.rsplit("_", 2)[0]                     # valid_301_a_2 -> valid_301


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def side_embedding(emb: np.ndarray) -> np.ndarray:
    """Several sentence embeddings of one side -> one: normalise each, average, normalise the average."""
    return unit(unit(np.atleast_2d(emb)).mean(axis=0))


def p_present(img: np.ndarray, neg: np.ndarray, pos: np.ndarray, temperature: float) -> np.ndarray:
    """img [n, d] normalised; neg / pos [d]. Softmax over the pair, as colipri's classify()."""
    d = (img @ pos - img @ neg) / temperature
    return 1.0 / (1.0 + np.exp(-d))


def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Probability that a positive scan scores above a negative one; ties count half."""
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    s = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def average_precision(scores: np.ndarray, y: np.ndarray) -> float | None:
    if y.sum() == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    yy = y[order]
    hits = np.cumsum(yy)
    return float((hits[yy == 1] / (np.flatnonzero(yy == 1) + 1)).mean())


def macro(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray], idx: np.ndarray, fn=auroc) -> float | None:
    vals = [fn(scores[k][idx], labels[k][idx]) for k in labels]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def analyse(scores: dict[str, dict[str, np.ndarray]], labels: dict[str, np.ndarray], patients: list[str], an: dict, baseline: str = "A") -> dict:
    """scores: set -> condition -> [n]; labels: condition -> [n] 0/1 (only conditions with a reference)."""
    n = len(patients)
    allidx = np.arange(n)
    per = {k: {"positive_scans": int(labels[k].sum()), "negative_scans": int(n - labels[k].sum()),
               "auroc": {s: auroc(scores[s][k], labels[k]) for s in scores if k in scores[s]},
               "average_precision": {s: average_precision(scores[s][k], labels[k]) for s in scores if k in scores[s]}} for k in labels}
    for k, v in per.items():
        v["interpretable"] = v["positive_scans"] >= an["min_positive_scans_to_interpret"]
    uniq = sorted(set(patients))
    by = {p: np.flatnonzero(np.array(patients) == p) for p in uniq}
    rng = np.random.default_rng(an["bootstrap"]["seed"])
    draws = {s: [] for s in scores if s != baseline}
    for _ in range(an["bootstrap"]["resamples"]):
        idx = np.concatenate([by[p] for p in rng.choice(uniq, len(uniq), replace=True)])
        base = macro({k: scores[baseline][k] for k in labels}, labels, idx)
        for s in draws:
            ks = {k: labels[k] for k in labels if k in scores[s]}
            m = macro({k: scores[s][k] for k in ks}, ks, idx)
            b = base if len(ks) == len(labels) else macro({k: scores[baseline][k] for k in ks}, ks, idx)
            if m is not None and b is not None:
                draws[s].append(m - b)
    lo, hi = an["bootstrap"]["interval"]
    sets = {}
    for s in scores:
        ks = {k: labels[k] for k in labels if k in scores[s]}
        m = macro({k: scores[s][k] for k in ks}, ks, allidx)
        row = {"conditions": len(ks), "macro_auroc": m, "macro_average_precision": macro({k: scores[s][k] for k in ks}, ks, allidx, average_precision)}
        if s != baseline:
            b = macro({k: scores[baseline][k] for k in ks}, ks, allidx)
            worse = sorted(k for k in ks if per[k]["interpretable"] and per[k]["auroc"][s] < per[k]["auroc"][baseline] - 0.05)
            ci = [float(np.percentile(draws[s], lo)), float(np.percentile(draws[s], hi))]
            row.update(difference_to_baseline=m - b, interval_95=ci, interval_above_zero=ci[0] > 0,
                       conditions_worse_by_more_than_0_05=worse, rule_met=bool(ci[0] > 0 and not worse))
        sets[s] = row
    winners = [s for s in sets if sets[s].get("rule_met")]
    chosen = max(winners, key=lambda s: sets[s]["macro_auroc"]) if winners else baseline
    return {"scans": n, "patients": len(uniq), "sets": sets, "per_condition": per, "rule": an["rule"], "rule_met_by": winners,
            "decision": (f"set {chosen} is the proposed method (confirmation on untouched scans needed before adoption)" if winners
                         else f"set {baseline} stays: no set met the pre-stated rule")}


def native_reports(cfg: dict, findings: list[dict], test_patients: set[str]) -> dict:
    """50 positive and 50 negative Findings sections per condition from patients NOT in the test scans. Returns condition -> lists of texts and names."""
    nat = cfg["native"]
    cache = ROOT / "data" / "raw" / "ct_rate" / "hf_cache"
    rep = sorted(cache.glob("**/radiology_text_reports/validation_reports.csv"))[0]
    lab = sorted(cache.glob("**/multi_abnormality_labels/valid_predicted_labels.csv"))[0]
    with open(rep, newline="", encoding="utf-8") as f:
        text = {r["VolumeName"].split(".")[0]: (r["Findings_EN"] or "").strip() for r in csv.DictReader(f)}
    with open(lab, newline="", encoding="utf-8") as f:
        rows = {r["VolumeName"].split(".")[0]: r for r in csv.DictReader(f)}
    first = {}
    for name in sorted(rows):
        p = patient_of(name)
        if p not in test_patients and p not in first and text.get(name):
            first[p] = name
    pool = sorted(first.values())
    rng = np.random.default_rng(nat["seed"])
    out = {"source": {"reports": str(rep.relative_to(ROOT)), "labels": str(lab.relative_to(ROOT)), "reports_sha256": sha256_file(rep), "labels_sha256": sha256_file(lab)},
           "pool_patients": len(pool), "excluded_test_patients": len(test_patients), "conditions": {}}
    for f in findings:
        if f["key"] in nat["not_available_for"] or not f.get("ct_rate_label"):
            continue
        col = f["label"]                               # the CT-RATE label name; a missing column stops the run
        pos = [n for n in pool if rows[n][col] == "1"]
        neg = [n for n in pool if rows[n][col] == "0"]
        ps = sorted(rng.choice(pos, min(nat["n_positive"], len(pos)), replace=False).tolist())
        ns = sorted(rng.choice(neg, min(nat["n_negative"], len(neg)), replace=False).tolist())
        out["conditions"][f["key"]] = {"available_positive": len(pos), "available_negative": len(neg), "positive": ps, "negative": ns,
                                       "positive_text": [text[n] for n in ps], "negative_text": [text[n] for n in ns]}
    return out


def load_model(mc: dict):
    import torch
    from colipri import get_model, get_processor
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    weights = Path(hf_hub_download(mc["model"]["repo_id"], mc["model"]["filename"], revision=mc["model"]["revision"]))
    model = get_model(pretrained=False)
    model.softmax_temperature = model.softmax_temperature.float()        # see scripts/agent_classify.py
    ckpt = load_file(str(weights))
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {list(missing)[:5]}, unexpected {list(unexpected)[:5]}")
    state = model.state_dict()
    if [k for k, v in ckpt.items() if not torch.equal(state[k].cpu(), v.to(state[k].dtype))]:
        raise RuntimeError("weights differ from the file after loading")
    return model.cuda().eval(), get_processor(), weights


def embed_text(model, processor, sentences: list[str]) -> tuple[np.ndarray, list[int]]:
    import torch
    out, cut = [], []
    for i in range(0, len(sentences), 16):
        batch = sentences[i:i + 16]
        ids, mask = processor.process_text(batch)
        full = processor._text_tokenizer(batch, truncation=False, padding=False)["input_ids"]
        cut += [int(len(x) > ids.shape[1]) for x in full]
        with torch.no_grad():
            out.append(model.encode_text(ids.to(model.device), mask.to(model.device)).float().cpu().numpy())
    return np.concatenate(out), cut


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt-sets", default="configs/colipri_prompt_sets_v1.json")
    ap.add_argument("--reuse", default=None, help="run folder with stored embeddings: analysis only")
    ap.add_argument("--label", default="PT01_colipri_prompts")
    ap.add_argument("--scan-list", default=None, metavar="FILE",
                    help="scans stored outside the repository: the file written by anatomy_contrast_check.py --write-scan-list")
    ap.add_argument("--scores-only", action="store_true",
                    help="set A scores only (scores.csv without a label column, embeddings, manifest); no label is read, nothing is compared")
    args = ap.parse_args()
    cfg_path = ROOT / args.prompt_sets
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    mc = json.loads((ROOT / cfg["model_config"]).read_text(encoding="utf-8"))
    if args.scores_only and (args.reuse or not args.scan_list):
        raise SystemExit("--scores-only needs --scan-list and cannot be combined with --reuse")
    if args.scan_list:
        sl = json.loads(Path(args.scan_list).read_text(encoding="utf-8"))
        ct_of = {s: Path(p) for s, p in sl["scans"].items()}
        gt = {} if args.scores_only else json.loads(Path(sl["cases_file"]).read_text(encoding="utf-8"))["cases"]
        scans = sorted(ct_of)
    else:
        gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
        scans = sorted(c for c in gt if (ROOT / "data/raw/ct_rate/volumes" / f"{c}.nii.gz").exists())
        ct_of = {c: ROOT / "data/raw/ct_rate/volumes" / f"{c}.nii.gz" for c in scans}
    patients = [patient_of(s) for s in scans]
    findings = cfg["findings"]
    run = ROOT / "runs" / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    manifest = {"run_id": run.name, "plan": cfg["plan"], "prompt_sets": args.prompt_sets, "prompt_sets_sha256": sha256_file(cfg_path),
                "model_config": cfg["model_config"], "model": f"{mc['model']['repo_id']}@{mc['model']['revision']}",
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                "started_utc": datetime.now(timezone.utc).isoformat(), "scans": scans, "failed": [], "reused_from": args.reuse, "scan_list": args.scan_list, "scores_only": args.scores_only}

    if args.reuse:
        src = ROOT / args.reuse
        z = np.load(src / "image_embeddings.npz")
        img, names, temperature = z["emb"], [str(x) for x in z["scans"]], float(z["temperature"])
        t = np.load(src / "text_embeddings.npz")
        text = {k: t[k] for k in t.files}
        native = json.loads((src / "native_reports.json").read_text(encoding="utf-8"))
    else:
        import torch
        model, processor, weights = load_model(mc)
        temperature = float(model.softmax_temperature)
        manifest.update(weights_sha256=sha256_file(weights), gpu=torch.cuda.get_device_name(0))
        embs, names = [], []
        for cid in scans:
            try:
                imgs = processor.to_images_batch(processor.process_images(str(ct_of[cid]))).to(model.device)
                with torch.no_grad():
                    e = None
                    for prec in ("bf16", "fp32"):                       # same order as scripts/agent_classify.py
                        if prec == "bf16":
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                e = model.encode_image(imgs, project=True, pool=True, normalize=True)
                        else:
                            e = model.encode_image(imgs.float(), project=True, pool=True, normalize=True)
                        e = e.float()
                        if bool(torch.isfinite(e).all()):
                            break
                if not bool(torch.isfinite(e).all()):
                    raise RuntimeError("non-finite image embedding")
                embs.append(e[0].cpu().numpy())
                names.append(cid)
                print(f"[{cid}] ok", flush=True)
            except Exception as ex:  # reported, never dropped
                manifest["failed"].append({"scan": cid, "error": f"{type(ex).__name__}: {ex}"})
                print(f"[{cid}] FAILED {ex}", flush=True)
        img = np.stack(embs)
        np.savez(run / "image_embeddings.npz", emb=img, scans=np.array(names), temperature=temperature)
        if args.scores_only:                                 # the current short prompts only; no report text, no label, no comparison
            with open(run / "scores.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["scan", "patient", "condition", "A"])
                for f in findings:
                    neg, pos = (side_embedding(embed_text(model, processor, f["A"][side])[0]) for side in ("negative", "positive"))
                    for sname, pv in zip(names, p_present(img, neg, pos, temperature)):
                        w.writerow([sname, patient_of(sname), f["key"], f"{pv:.6f}"])
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (run / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            print(f"scores only: {len(names)} scans, {len(manifest['failed'])} failed -> {run}")
            return 0
        native = native_reports(cfg, findings, set(patient_of(s) for s in names))
        text, cut_total = {}, 0
        for f in findings:
            for s in ("A", "B", "C"):
                for side in ("negative", "positive"):
                    e, _ = embed_text(model, processor, f[s][side])
                    text[f"{s}|{f['key']}|{side}"] = side_embedding(e)
            nat = native["conditions"].get(f["key"])
            if nat:
                for side in ("negative", "positive"):
                    e, cut = embed_text(model, processor, nat[f"{side}_text"])
                    nat[f"{side}_truncated_at_512_tokens"] = int(sum(cut))
                    cut_total += sum(cut)
                    text[f"D|{f['key']}|{side}"] = side_embedding(e)
        np.savez(run / "text_embeddings.npz", **text)
        manifest["native_reports_truncated"] = int(cut_total)
    slim = {**native, "conditions": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_text")} for k, v in native["conditions"].items()}}
    (run / "native_reports.json").write_text(json.dumps(native, indent=1), encoding="utf-8")
    manifest["native"] = slim

    order = [names.index(s) for s in names]
    pats = [patient_of(s) for s in names]
    scores: dict[str, dict[str, np.ndarray]] = {}
    for key in text:
        s, k, side = key.split("|")
        if side == "positive":
            scores.setdefault(s, {})[k] = p_present(img[order], text[f"{s}|{k}|negative"], text[key], temperature)
    lab_of = {f["key"]: f["label"] for f in findings if f.get("ct_rate_label") and f["key"] not in cfg["native"]["not_available_for"]}
    labels = {k: np.array([int(lab in gt[s]["labels"]) for s in names]) for k, lab in lab_of.items()}
    with open(run / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scan", "patient", "condition", "report_label", *sorted(scores)])
        for i, sname in enumerate(names):
            for f in findings:
                k = f["key"]
                w.writerow([sname, pats[i], k, labels[k][i] if k in labels else ""] + [f"{scores[s][k][i]:.6f}" if k in scores[s] else "" for s in sorted(scores)])
    res = analyse(scores, labels, pats, cfg["analysis"])
    res["label_of_condition"] = lab_of
    res["temperature"] = temperature
    (run / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    S = sorted(scores)
    md = [f"# {run.name}: COLIPRI prompt sets on {res['scans']} scans ({res['patients']} patients)", "",
          "Development scans; reference = CT-RATE report labels (a mention in the report). A choice of method, not a performance figure.", "",
          "| Prompt set | Conditions | Macro AUROC | Difference to A | 95 % interval | Rule met |", "|---|---|---|---|---|---|"]
    for s in S:
        r = res["sets"][s]
        md.append(f"| {s} | {r['conditions']} | {r['macro_auroc']:.3f} | " + (f"{r['difference_to_baseline']:+.3f} | {r['interval_95'][0]:+.3f} to {r['interval_95'][1]:+.3f} | {'yes' if r['rule_met'] else 'no'} |"
                                                                              if s != "A" else "baseline | | |"))
    md += ["", "| Condition | Positive scans | " + " | ".join(S) + " | Note |", "|---|---|" + "---|" * (len(S) + 1)]
    for k, v in res["per_condition"].items():
        md.append(f"| {k} | {v['positive_scans']} | " + " | ".join(f"{v['auroc'][s]:.2f}" if v["auroc"].get(s) is not None else "-" for s in S)
                  + f" | {'' if v['interpretable'] else 'too few to interpret'} |")
    md += ["", f"Rule: {res['rule']}", "", f"**Decision: {res['decision']}**"]
    (run / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {run}")
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
