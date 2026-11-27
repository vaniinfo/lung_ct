"""Prompt test for NV-Reason-CT (nvreason env): the same scans asked with several frozen prompt sets. Collects answers only.

Follows docs/PROMPT_TEST_TEMPLATE.md; plan: docs/prompt_tests/PT02_nvreason_prompts_PLAN.md. Everything runs locally.
This script never reads a label. Scoring is done afterwards by scripts/prompt_test_yesno.py (core env).

Speed: an ordinary call makes the model read the whole scan again for every question (about 5 s). Here the scan and
the text before the question are processed ONCE per scan and kept; each question starts from a copy of that state and
is fed one token at a time, exactly as generation feeds tokens. On the first scan and every 20th, three baseline
questions are also asked the ordinary way: the answer text must be identical, otherwise the run stops.

Per scan, in this order: the yes / no and location sets, then set D (report first) last, because D extends the kept
state with the model's own report.

Usage:
  environments/nvreason/Scripts/python.exe scripts/prompt_test_nvreason.py --sets A,A2,A3,C,E
  ... --resume runs/PT02_nvreason_prompts_<time> --sets B,D        (adds sets to the same run; finished scan-sets are skipped)
Writes runs/PT02_nvreason_prompts_<time>/{run_manifest.json,answers.jsonl,reports/<scan>.txt,equivalence.jsonl,failures.jsonl}
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

ABSENT_WORDS = ("none", "no", "nowhere", "absent", "not")


def after_thinking(text: str) -> str:
    return text.partition("</think>")[2] if "</think>" in text else text


def read_yes_no(text: str) -> str | None:
    w = re.sub(r"[^a-z]", " ", after_thinking(text).lower()).split()
    return {"yes": "present", "no": "absent"}.get(w[0]) if w else None


def read_location(text: str) -> str | None:
    """Location-first answer: a first word that denies the finding -> absent; empty -> unreadable; anything else -> present."""
    w = re.sub(r"[^a-z]", " ", after_thinking(text).lower()).split()
    if not w:
        return None
    return "absent" if w[0] in ABSENT_WORDS else "present"


READERS = {"yes_no": read_yes_no, "location": read_location}


def question(set_cfg: dict, cond: dict) -> str:
    return set_cfg["template"].format(phrase=cond["phrase"], definition=cond["definition"])


def common_prefix_len(rows: list[list[int]]) -> int:
    n = min(len(r) for r in rows)
    for i in range(n):
        if any(r[i] != rows[0][i] for r in rows):
            return i
    return n


CT_PATHS: dict[str, str] = {}      # scan -> volume path, for scans stored outside the repository (--scan-list)


def ct_path(scan: str) -> Path:
    return Path(CT_PATHS[scan]) if scan in CT_PATHS else ROOT / "data/raw/ct_rate/volumes" / f"{scan}.nii.gz"


def verified_scans() -> tuple[dict, list]:
    """Scan -> QC run, for local CT-RATE scans that are HU-valid with left / right verified from anatomy; and the scans left out."""
    gt = json.loads((ROOT / "data/raw/ct_rate/cases.json").read_text(encoding="utf-8"))["cases"]
    scans = sorted(c for c in gt if (ROOT / "data/raw/ct_rate/volumes" / f"{c}.nii.gz").exists())
    ok = {}
    for p in sorted((ROOT / "runs").glob("P1_qc_verified_*/run_manifest.json")):
        for c in json.loads(p.read_text())["cases"]:
            if c["status"] != "failed" and c["ct"]["hu_valid"] and c.get("laterality", {}).get("verified"):
                ok[c["case_id"]] = p.parent.name
    return {s: ok[s] for s in scans if s in ok}, [s for s in scans if s not in ok]


class Session:
    """One scan: the state after the scan (and the text before the question) is kept; questions start from a copy."""

    def __init__(self, processor, model, ct_path: Path, fixed: dict):
        import torch
        self.torch, self.p, self.m, self.ct, self.fixed = torch, processor, model, str(ct_path), fixed
        ids = [processor.tokenizer.encode(w, add_special_tokens=False) for w in ("Yes", "No")]
        if any(len(i) != 1 for i in ids):
            raise RuntimeError("'Yes' / 'No' are not single tokens")
        self.yes_id, self.no_id = ids[0][0], ids[1][0]
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.cache, self.ids, self.rope = None, None, None      # kept state, the token ids it covers, MRoPE offset

    def tokens(self, messages: list, thinking: bool):
        prompt = self.p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return self.p(text=prompt, images3d=[self.ct], anatomy_region=self.fixed["anatomy_region"], return_tensors="pt").to(self.m.device)

    def start(self, inputs) -> None:
        """Process the scan and the text up to the end of the image block, once."""
        torch, ids = self.torch, inputs.input_ids
        pos = (ids[0] == self.m.config.image_token_id).nonzero()[:, 0]
        n = int(pos[-1]) + 2                                     # image tokens and the token that closes the image block
        pre = {k: (v[:, :n] if k in ("input_ids", "attention_mask", "mm_token_type_ids") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.m.model(**pre, use_cache=True)
        self.cache, self.ids, self.rope = out.past_key_values, ids[0, :n].tolist(), self.m.model.rope_deltas.clone()

    def extend(self, new_ids: list[int]) -> None:
        """Feed more tokens into the kept state itself (no copy): used for the report conversation of set D."""
        torch = self.torch
        self.m.model.rope_deltas = self.rope
        with torch.inference_mode():
            for t in new_ids:
                self.m.model(input_ids=torch.tensor([[t]], device=self.m.device), past_key_values=self.cache, use_cache=True)
        self.ids = self.ids + list(new_ids)

    def ask(self, messages: list, thinking: bool, max_new_tokens: int, score: bool, in_place: bool = False) -> dict:
        torch = self.torch
        inputs = self.tokens(messages, thinking)
        if self.cache is None:
            self.start(inputs)
        ids = inputs.input_ids[0].tolist()
        n = len(self.ids)
        if ids[:n] != self.ids:
            raise RuntimeError("the prompt does not start with the tokens already processed for this scan")
        t0 = time.time()
        cache = self.cache if in_place else copy.deepcopy(self.cache)
        self.m.model.rope_deltas = self.rope
        new, first = [], None
        with torch.inference_mode():
            for t in ids[n:-1]:
                self.m.model(input_ids=torch.tensor([[t]], device=self.m.device), past_key_values=cache, use_cache=True)
            cur = ids[-1]
            for _ in range(max_new_tokens):
                o = self.m.model(input_ids=torch.tensor([[cur]], device=self.m.device), past_key_values=cache, use_cache=True)
                logits = self.m.lm_head(o.last_hidden_state[:, -1])[0]
                if first is None:
                    first = torch.softmax(logits.float(), 0)
                cur = int(logits.argmax())                       # sampling off: the most likely token
                new.append(cur)
                if cur in self.eos:
                    break
        text = self.p.batch_decode(torch.tensor([new]), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        rec = {"raw": text, "output_tokens": len(new), "input_tokens": len(ids), "seconds": round(time.time() - t0, 2),
               "truncated": bool(len(new) >= max_new_tokens and new[-1] not in self.eos)}
        if score:        # how the first answer token is split between "Yes" and "No"; uncalibrated, not a probability of disease
            y, no = float(first[self.yes_id]), float(first[self.no_id])
            rec["score_yes"] = round(y / (y + no), 4) if y + no > 0 else None
        if in_place:
            self.ids = ids + new
        del cache
        return rec


def user(text: str, with_image: bool = False) -> dict:
    return {"role": "user", "content": ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": text}]}


def run_scan(scan: str, sets: list[str], cfg: dict, processor, model, run: Path, check: bool) -> tuple[list[dict], list[dict]]:
    import run_nvreason as nv
    ct = ct_path(scan)
    s = Session(processor, model, ct, cfg["fixed"])
    rows, eq = [], []
    for name in [x for x in sets if x != "D"]:
        sc = cfg["sets"][name]
        for c in cfg["conditions"]:
            q = question(sc, c)
            a = s.ask([user(q, True)], sc["thinking"], sc["max_new_tokens"], sc["score"])
            rows.append({"scan": scan, "set": name, "key": c["key"], "prompt": q, "answer": READERS[sc["read"]](a["raw"]), **a})
    if check:            # the ordinary call (scan read again for the question) must give the same text
        ref = nv.Asker(processor, model, ct, cfg["fixed"]["anatomy_region"], cfg["fixed"]["do_sample"])
        sc = cfg["sets"]["A"]
        for c in cfg["conditions"][:3]:
            q = question(sc, c)
            r = ref.ask(q, sc["thinking"], sc["max_new_tokens"], score_yes_no=True)
            mine = next((x for x in rows if x["set"] == "A" and x["key"] == c["key"]), None) or s.ask([user(q, True)], sc["thinking"], sc["max_new_tokens"], True)
            eq.append({"scan": scan, "key": c["key"], "ordinary": r["raw"], "kept_state": mine["raw"], "same_text": r["raw"] == mine["raw"],
                       "score_ordinary": r["score"]["p_yes_of_yes_no"], "score_kept_state": mine.get("score_yes"), "seconds_ordinary": r["seconds"], "seconds_kept_state": mine["seconds"]})
    if "D" in sets:      # last: the kept state is extended with the report conversation, so nothing else can use it afterwards
        sc = cfg["sets"]["D"]
        first = [user(sc["report_prompt"], True)]
        r = s.ask(first, sc["report_thinking"], sc["report_max_new_tokens"], False)
        report = after_thinking(r["raw"]).strip()
        (run / "reports").mkdir(exist_ok=True)
        (run / "reports" / f"{scan}.txt").write_text(r["raw"], encoding="utf-8")
        rows.append({"scan": scan, "set": "D", "key": "_report", "prompt": sc["report_prompt"], "answer": None, **{k: v for k, v in r.items() if k != "raw"}, "raw": f"[{len(report)} characters, see reports/{scan}.txt]"})
        if r["truncated"] or not report:
            raise RuntimeError("set D: the report reached the token limit or is empty; no follow-up question was asked")
        convo = first + [{"role": "assistant", "content": [{"type": "text", "text": report}]}]
        toks = [s.tokens(convo + [user(question(sc, c))], sc["thinking"]).input_ids[0].tolist() for c in cfg["conditions"]]
        n0, shared = len(s.ids), common_prefix_len(toks)
        if toks[0][:n0] != s.ids:
            raise RuntimeError("set D: the report conversation does not start with the tokens already processed")
        s.extend(toks[0][n0:shared])                             # the report, fed once into the kept state
        for c, t in zip(cfg["conditions"], toks):
            q = question(sc, c)
            a = s.ask(convo + [user(q)], sc["thinking"], sc["max_new_tokens"], sc["score"])
            rows.append({"scan": scan, "set": "D", "key": c["key"], "prompt": q, "answer": READERS[sc["read"]](a["raw"]), **a})
    return rows, eq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt-sets", default="configs/nvreason_prompt_sets_v1.json")
    ap.add_argument("--sets", required=True, help="comma-separated set names from the config, e.g. A,A2,A3,C,E")
    ap.add_argument("--resume", default=None, help="existing run folder: add sets / continue; finished scan-sets are skipped")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--label", default="PT02_nvreason_prompts")
    ap.add_argument("--scan-list", default=None, metavar="FILE",
                    help="scans stored outside the repository: the file written by anatomy_contrast_check.py --write-scan-list (HU-valid, left / right verified)")
    args = ap.parse_args()
    import run_nvreason as nv
    cfg_path = ROOT / args.prompt_sets
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    sets = args.sets.split(",")
    if any(x not in cfg["sets"] for x in sets):
        raise SystemExit(f"unknown set in {sets}")
    mc = json.loads((ROOT / cfg["model_config"]).read_text(encoding="utf-8"))
    if args.scan_list:
        sl = json.loads(Path(args.scan_list).read_text(encoding="utf-8"))
        CT_PATHS.update(sl["scans"])
        ok, left_out = {s: sl["qc_verified_run"] for s in sl["scans"]}, [x["scan"] for x in sl["left_out"]]
    else:
        ok, left_out = verified_scans()
    scans = [s for s in (args.cases or list(ok)) if s in ok]
    run = ROOT / args.resume if args.resume else ROOT / "runs" / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    mpath = run / "run_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"run_id": run.name, "plan": cfg["plan"], "prompt_sets": args.prompt_sets, "prompt_sets_sha256": nv.sha256_file(cfg_path),
                                                                      "model": f"{mc['model']['model_id']}@{mc['model']['revision']}", "scans": scans, "scan_qc_run": {s: ok[s] for s in scans},
                                                                      "left_out": [{"scan": s, "reason": "not HU-valid with left / right verified from anatomy in any QC run"} for s in left_out], "scan_list": args.scan_list, "sessions": []}
    if manifest["prompt_sets_sha256"] != nv.sha256_file(cfg_path):
        raise SystemExit("the prompt-set file changed since this run started: that is a new test, use a new run folder")
    done = set()
    if (run / "answers.jsonl").exists():
        for line in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["scan"], r["set"]))
    import torch
    local = nv.snapshot_dir(mc["model"])
    processor = nv.load_processor(local)
    model, info = nv.load_model(mc["model"], local)
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "sets": sets, "git_commit": nv.git_commit(), "gpu": torch.cuda.get_device_name(0), "weights_sha256": info["weights_sha256"]}
    n_done = 0
    for i, scan in enumerate(scans):
        todo = [x for x in sets if (scan, x) not in done]
        if not todo:
            continue
        t0 = time.time()
        try:
            rows, eq = run_scan(scan, todo, cfg, processor, model, run, check=(n_done % 20 == 0))
        except Exception as e:  # reported, never dropped
            with open(run / "failures.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"scan": scan, "sets": todo, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}) + "\n")
            print(f"[{scan}] FAILED {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        with open(run / "answers.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if eq:
            with open(run / "equivalence.jsonl", "a", encoding="utf-8") as f:
                for r in eq:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not all(r["same_text"] for r in eq):
                print(f"[{scan}] STOP: the kept-state answers differ from the ordinary call: {eq}", flush=True)
                return 3
        n_done += 1
        unread = sum(r["answer"] is None and r["key"] != "_report" for r in rows)
        print(f"[{i + 1}/{len(scans)} {scan}] {','.join(todo)}: {len(rows)} answers, {unread} unreadable, {time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        torch.cuda.empty_cache()
    session["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["sessions"].append(session)
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"done -> {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
