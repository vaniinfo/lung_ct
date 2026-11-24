"""Grounded agent demo, step 5 (vlm env): MedGemma writes the final report from the evidence.

Text only: the model sees the evidence table, the readers' answers and the cross-check, not the
images. Every finding must cite evidence ids; validate_report() flags uncited findings, unknown
ids and numbers that do not appear in the cited rows. If the output is not valid JSON in the
required shape, the rule-based report is used and that is recorded.

Writes into the agent run directory: report_raw.txt, final_report.json
Usage:
  environments/vlm/Scripts/python.exe scripts/agent_write_report.py --run runs/A1_agent_<case>_<time>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from lung_ct.agent.report import validate_report  # noqa: E402
from lung_ct.vlm.prompting import extract_json  # noqa: E402

MODEL_CACHE = ROOT / "models" / "medgemma" / "hub"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--config", default="configs/medgemma_ct_v1.json")
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
    rin = json.loads((run / "report_input.json").read_text(encoding="utf-8"))
    prompt = rin["prompt"]
    if rin.get("kind") == "checklist":
        from lung_ct.agent.checklist import load_checklist_config, validate_checklist_report
        ck = load_checklist_config(ROOT / rin["checklist_config"])
        chk = json.loads((run / "checklist.json").read_text(encoding="utf-8"))
        check = lambda o, e: validate_checklist_report(o, e, chk, ck, validate_report)  # noqa: E731
    else:
        check = validate_report
    fallback = json.loads((run / "report_rule_based.json").read_text(encoding="utf-8"))
    mcfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))["model"]

    import torch
    import transformers
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(mcfg["model_id"], revision=mcfg["revision"], cache_dir=str(MODEL_CACHE)))
    processor = transformers.AutoProcessor.from_pretrained(local, use_fast=mcfg["use_fast_processor"])
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        local, dtype=getattr(torch, mcfg["dtype"]), device_map=mcfg["device_map"])
    model.eval()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    text = processor.decode(out[0, n_in:], skip_special_tokens=True)
    (run / "report_raw.txt").write_text(text, encoding="utf-8")

    # MedGemma 1.5 may emit a thinking block (<unused94>thought ... <unused95>) before its answer
    thinking, answer = None, text
    if "<unused95>" in text:
        thinking, answer = text.split("<unused95>", 1)
    elif text.lstrip().startswith("<unused94>"):
        thinking, answer = text, ""  # thinking never finished: no answer
    obj, method, err = extract_json(answer)
    val = check(obj, ev) if obj is not None else {"valid": False, "errors": [err], "flags": []}
    rec = {"writer": f"{mcfg['model_id']}@{mcfg['revision'][:8]} (text only, greedy)",
           "input_tokens": int(n_in), "output_tokens": int(out.shape[1] - n_in),
           "seconds": round(time.time() - t0, 1), "extraction_method": method, "validation": val,
           "thinking_chars": len(thinking) if thinking else 0,
           "truncated": int(out.shape[1] - n_in) >= args.max_new_tokens}
    if val["valid"]:
        rec.update(source="medgemma", report=obj)
    else:
        rec.update(source="rule_based_fallback", report=fallback,
                   note="MedGemma output was not a valid report; rule-based report used")
    (run / "final_report.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"step 5: source={rec['source']} valid={val['valid']} flags={len(val['flags'])} -> {run / 'final_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
