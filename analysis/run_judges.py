"""Homework 5: judge inputs, label split, development runs, and the final test.

Mode: policy_claim_without_citation (chosen in Part A).

Functions:
    prepare_inputs()            Part B. One judge input record per labelled conversation (user turns and assistant
                                replies only), saved to analysis/state/hw5_trace_inputs.json. Offline.
    split_data(mode)            Part B. 20/40/40 train/dev/test via split_labels (seed 7), saved to
                                analysis/state/splits.json. Offline. Refuses to run twice for the same mode.
    run_development(mode, path) Part C. Register a prompt (once), run it on dev, save analysis/report/dev-<id>.json.
                                Paid (gpt-4o-mini-2024-07-18).
    run_test(judge_id)          Part D. Freeze once, run on test, save analysis/report/test-<id>.json. Paid.
    recompute_metrics(id, split) Recompute TP/FN/TN/FP, TPR, TNR and Wilson intervals from the saved files alone.
                                Offline; used to check the reports (and live in the video).

Usage (from the repository root):
    uv run python -m analysis.run_judges prepare
    uv run python -m analysis.run_judges split
    uv run python -m analysis.run_judges dev --prompt analysis/prompts/policy_claim_without_citation-v2.txt
    uv run python -m analysis.run_judges test --judge policy_claim_without_citation-v2
    uv run python -m analysis.run_judges recompute --judge policy_claim_without_citation-v2 --split test
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "analysis" / "state"
MODE = "policy_claim_without_citation"
INPUTS = STATE / "hw5_trace_inputs.json"
LABELS = STATE / "hw5_labels" / f"{MODE}.jsonl"
HW4_CACHE = ROOT / "analysis" / "review_app" / "cache" / "conversations.json"

# Every judge run reads the saved inputs, never a fresh Langfuse fetch, so all prompt versions see the same text.
os.environ.setdefault("CARTWHEEL_JUDGE_TRACE_SOURCE", str(INPUTS))


def _labelled_ids(mode: str = MODE) -> list[str]:
    rows = [json.loads(x) for x in LABELS.read_text().splitlines() if x.strip()]
    live = {r["trace_id"]: r for r in rows if not r.get("superseded_by")}
    return sorted(live)


def prepare_inputs() -> list[dict[str, Any]]:
    """Write one judge input per labelled conversation: user turns and assistant replies, in order.

    Student decision (Part B, option a): the judge sees only what the user saw. No tool calls or results (their
    policy_ids lists could be mistaken for citations; D is decided from the replies), no interim model text, no
    labels, notes, scenario ids, or answer keys.
    """
    convs = {c["cid"]: c for c in json.loads(HW4_CACHE.read_text())["conversations"]}
    ids = _labelled_ids()
    missing = [cid for cid in ids if cid not in convs]
    if missing:
        raise SystemExit(f"labelled conversations missing from the HW4 cache: {missing[:5]}")

    records = []
    for cid in ids:
        trace = []
        for turn in convs[cid]["turns"]:
            trace.append({"role": "user", "text": turn["user"]})
            trace.append({"role": "assistant", "text": turn["reply"] or ""})
        records.append({"trace_id": cid, "trace": trace})

    # Leak checks: one record per label, only the two allowed fields, only user/assistant messages.
    assert len(records) == len(ids) == len({r["trace_id"] for r in records}), "one input per label"
    assert all(set(r) == {"trace_id", "trace"} for r in records), "no extra fields (labels, metadata) in inputs"
    assert all(m["role"] in ("user", "assistant") and set(m) == {"role", "text"} for r in records for m in r["trace"])
    INPUTS.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {INPUTS.relative_to(ROOT)}: {len(records)} records, "
          f"{sum(len(r['trace']) // 2 for r in records)} turns")
    return records


def split_data(mode: str = MODE) -> dict[str, list[str]]:
    """Split the HW5 labels 20/40/40 (seed 7) once; splits.json must not change during development."""
    from analysis.helpers import split_labels

    existing = json.loads((STATE / "splits.json").read_text()) if (STATE / "splits.json").exists() else {}
    if mode in existing:
        raise SystemExit(f"splits.json already has a split for {mode}; it must stay unchanged during development")
    records = json.loads(INPUTS.read_text())
    splits = split_labels(
        mode,
        fractions=(0.20, 0.40, 0.40),
        seed=7,
        min_per_class=10,
        eligible_trace_ids=[record["trace_id"] for record in records],
    )
    pass_ids = {json.loads(x)["trace_id"] for x in LABELS.read_text().splitlines()
                if x.strip() and json.loads(x)["label"] == 1}
    for name in ("train", "dev", "test"):
        ids = splits[name]
        p = sum(t in pass_ids for t in ids)
        print(f"{name:<6} n={len(ids):>3}  Pass={p:>3}  Fail={len(ids) - p:>3}")
    return splits


JUDGE_MODEL = "gpt-4o-mini-2024-07-18"  # handout: gpt-4o-mini, same model for dev and test; pinned snapshot (student decision)


def _registered(mode: str, prompt_text: str, model: str) -> str | None:
    """Judge id already registered for exactly this prompt text and model (so a rerun resumes, not re-registers)."""
    from analysis.helpers import tools

    history = json.loads((STATE / "judges" / f"_history_{mode}.json").read_text()) \
        if (STATE / "judges" / f"_history_{mode}.json").exists() else {"versions": []}
    h = tools._prompt_hash(prompt_text, model)
    return next((v["judge_id"] for v in history["versions"] if v["prompt_hash"] == h), None)


def run_development(mode: str, prompt_path: str | Path, judge_model: str = JUDGE_MODEL) -> dict[str, Any]:
    """Register the prompt (once), run it on the dev split, score it, and save analysis/report/dev-<judge_id>.json.

    Paid: one judge call per dev conversation not yet cached. Cached predictions are reused on a rerun.
    """
    from analysis.helpers import judge_alignment, register_judge, run_judge

    prompt_text = Path(prompt_path).read_text()
    judge_id = _registered(mode, prompt_text, judge_model)
    if judge_id is None:
        judge_id = register_judge(mode=mode, prompt_text=prompt_text, judge_model=judge_model)["judge_id"]
    run_judge(judge_id, split="dev", batch_size=10)
    development = judge_alignment(judge_id, split="dev")
    report = {"judge_id": judge_id, "mode": mode, "split": "dev", "model": judge_model,
              "prompt_path": str(Path(prompt_path).relative_to(ROOT) if Path(prompt_path).is_absolute() else prompt_path),
              **development}
    out = ROOT / "analysis" / "report" / f"dev-{judge_id}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{judge_id}: TP={development['tp']} FN={development['fn']} TN={development['tn']} FP={development['fp']}  "
          f"TPR={development['tpr']:.2f} {development['tpr_interval']}  TNR={development['tnr']:.2f} "
          f"{development['tnr_interval']}  -> {out.relative_to(ROOT)}")
    return report


def run_test(judge_id: str) -> dict[str, Any]:
    """Freeze the chosen judge, run it once on the test split, score it, and save analysis/report/test-<judge_id>.json.

    Paid: one judge call per test conversation. Freeze happens once; on a rerun after an interruption the judge is
    already frozen and only missing predictions are paid for.
    """
    from analysis.helpers import freeze_judge, judge_alignment, run_judge, tools

    if tools._load_judge(judge_id).get("status") != "frozen":
        freeze_judge(judge_id)  # do this once for the selected version
    run_judge(judge_id, split="test", batch_size=10)
    test = judge_alignment(judge_id, split="test")
    judge = tools._load_judge(judge_id)
    report = {"judge_id": judge_id, "mode": judge["mode"], "split": "test", "model": judge["model"],
              "pass_count": test["tp"] + test["fn"], "fail_count": test["tn"] + test["fp"], **test}
    out = ROOT / "analysis" / "report" / f"test-{judge_id}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{judge_id} TEST: Pass={report['pass_count']} Fail={report['fail_count']}  TP={test['tp']} FN={test['fn']} "
          f"TN={test['tn']} FP={test['fp']}  TPR={test['tpr']:.2f} {test['tpr_interval']}  "
          f"TNR={test['tnr']:.2f} {test['tnr_interval']}  -> {out.relative_to(ROOT)}")
    return report


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    import math

    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(centre - margin, 4), round(centre + margin, 4))


def recompute_metrics(judge_id: str, split: str = "test") -> dict[str, Any]:
    """Recompute the confusion counts, TPR, TNR and Wilson intervals from the saved files alone (no helper, no model).

    Reads the judge's cached predictions (analysis/state/judges/<judge_id>.json), the split
    (analysis/state/splits.json) and the HW5 labels. Pass is the positive class: 1 = Pass, 0 = Fail.
    """
    judge = json.loads((STATE / "judges" / f"{judge_id}.json").read_text())
    predictions = judge["predictions"][judge["prompt_hash"]]
    ids = json.loads((STATE / "splits.json").read_text())[judge["mode"]][split]
    labels: dict[str, int] = {}
    for line in (STATE / "hw5_labels" / f"{judge['mode']}.jsonl").read_text().splitlines():
        row = json.loads(line) if line.strip() else None
        if row and not row.get("superseded_by"):
            labels[row["trace_id"]] = int(row["label"])

    tp = sum(labels[t] == 1 and predictions[t] == 1 for t in ids)
    fn = sum(labels[t] == 1 and predictions[t] == 0 for t in ids)
    tn = sum(labels[t] == 0 and predictions[t] == 0 for t in ids)
    fp = sum(labels[t] == 0 and predictions[t] == 1 for t in ids)
    out = {"judge_id": judge_id, "split": split, "pass_count": tp + fn, "fail_count": tn + fp,
           "tp": tp, "fn": fn, "tn": tn, "fp": fp,
           "tpr": tp / (tp + fn), "tpr_interval": _wilson(tp, tp + fn),
           "tnr": tn / (tn + fp), "tnr_interval": _wilson(tn, tn + fp)}
    print(f"{judge_id} {split} (recomputed from saved files): Pass={out['pass_count']} Fail={out['fail_count']}  "
          f"TP={tp} FN={fn} TN={tn} FP={fp}  TPR={out['tpr']:.3f} {out['tpr_interval']}  "
          f"TNR={out['tnr']:.3f} {out['tnr_interval']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "split", "dev", "test", "recompute"])
    ap.add_argument("--split", default="test", help="split for 'recompute' (default test)")
    ap.add_argument("--judge", help="judge id for 'test'")
    ap.add_argument("--prompt", help="prompt file for 'dev'")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare_inputs()
    elif args.command == "split":
        split_data(MODE)
    elif args.command == "dev":
        run_development(MODE, args.prompt or ROOT / "analysis" / "prompts" / f"{MODE}-v0.txt")
    elif args.command == "test":
        if not args.judge:
            raise SystemExit("--judge is required for 'test'")
        run_test(args.judge)
    else:
        recompute_metrics(args.judge or f"{MODE}-v2", args.split)


if __name__ == "__main__":
    main()
