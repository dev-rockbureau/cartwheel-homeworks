"""Add a review batch to ``analysis/state/sample_manifest.json`` (Part B). Offline, deterministic, no model.

Methods:
    uniform  N conversations drawn uniformly at random from those not yet in the sample
    cluster  N cluster representatives (nearest to each k-means centroid in the cache's map projection),
             spread across clusters round-robin, from those not yet in the sample
    stratified  N conversations spread evenly across the values of one scenario-tuple field (--field, an INPUT
             dimension from scenarios/support_scenarios.jsonl, never an outcome), random within each value
    depth    N conversations from the depth searches in DEPTH_SEARCHES: each search selects the SETUP in which a
             draft mode could occur (scenario input fields or trace structure), never a predicted failure, and
             picks at random within it, so both positives and close negatives come back

Every pick is a conversation id (first Langfuse trace id). A conversation never appears in two batches.

Usage (from the repository root):
    uv run python -m analysis.review_app.make_sample uniform-1 uniform 15 --seed 20261110
    uv run python -m analysis.review_app.make_sample cluster-1 cluster 15
    uv run python -m analysis.review_app.make_sample dimension-2 stratified 30 --field intent --seed 20261110
    uv run python -m analysis.review_app.make_sample depth-3 depth 25 --seed 20261110
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache" / "conversations.json"
MANIFEST = ROOT / "analysis" / "state" / "sample_manifest.json"
SCENARIOS = ROOT / "scenarios" / "support_scenarios.jsonl"


def _order_store_ids() -> dict[int, int]:
    """order id -> store id from the seed database (read-only). An input fact about the record, not agent behaviour."""
    import sqlite3

    db = sqlite3.connect(f"file:{ROOT / 'data' / 'cartwheel.db'}?mode=ro", uri=True)
    try:
        return dict(db.execute("select id, store_id from orders"))
    finally:
        db.close()


def _fee_store_ids() -> set[int]:
    from seed.generate import CORE_STORES  # (name, category, window override, restocking opt-in)
    return {i for i, (_n, _c, _o, optin) in enumerate(CORE_STORES, start=1) if optin}


# Depth searches for the batch-3 draft modes (student-approved split, 2026-11-10). Each selects a SETUP, not an outcome.
# (label, mode, n, predicate(conv, tuple))
def depth_searches() -> list[tuple[str, str, int, Any]]:
    fee_stores, order_store = _fee_store_ids(), _order_store_ids()
    return [
        ("record state: duplicate product title or two matching orders", "ambiguous_match_presented_as_unique", 4,
         lambda c, t: t.get("record_state") in ("product_duplicate_title", "ambiguous_two_matching_orders")),
        ("shipped-not-delivered, last day / one day past window, past dispute window, or dispute intent",
         "policy_rule_misapplied", 4,
         lambda c, t: t.get("record_state") in ("order_shipped", "order_past_dispute_window", "order_last_day_of_window",
                                                "order_one_day_past_window") or t.get("intent") == "dispute"),
        ("intent: out_of_scope", "out_of_scope_request_answered", 4, lambda c, t: t.get("intent") == "out_of_scope"),
        ("difficulty: correction_across_turns", "correction_not_reconciled", 4,
         lambda c, t: t.get("difficulty") == "correction_across_turns"),
        ("return/refund question about an order at a store with a restocking fee", "omitted_restocking_fee", 3,
         lambda c, t: t.get("intent") in ("return_eligibility", "refund")
         and order_store.get(t.get("order_id")) in fee_stores),
        ("intent: policy_question", "policy_question_answered_without_lookup / policy_claim_without_citation", 3,
         lambda c, t: t.get("intent") == "policy_question"),
        ("intent: account_change or dispute", "required_escalation_not_made", 3,
         lambda c, t: t.get("intent") in ("account_change", "dispute")),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("method", choices=["uniform", "cluster", "stratified", "depth"])
    ap.add_argument("n", type=int)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--field", help="scenario tuple field for --method stratified (e.g. intent)")
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text())
    convs = {c["cid"]: c for c in cache["conversations"]}
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest.setdefault("population", {"description": "HW3 final run, support-* conversations (sessions)",
                                       "source": cache["source"], "conversations": len(convs),
                                       "traces": cache["trace_count"]})
    manifest.setdefault("unit", "conversation (session); cid = first Langfuse trace id")
    batches = manifest.setdefault("batches", [])
    if any(b["name"] == args.name for b in batches):
        raise SystemExit(f"batch {args.name} already exists")
    taken = {p["cid"] for b in batches for p in b["picks"]}
    # practice conversations were already read (and coded) outside the sample, so they cannot be drawn again
    ann_file = ROOT / "analysis" / "state" / "annotations.json"
    if ann_file.exists():
        taken |= {a["trace_id"] for a in json.loads(ann_file.read_text()).get("annotations", [])
                  if a.get("batch") == "practice"}
    pool = sorted(cid for cid in convs if cid not in taken)

    if args.method == "depth":
        tuples = {}
        for line in SCENARIOS.read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                tuples[s["id"]] = s["tuple"]
        rng = random.Random(args.seed)
        picks, searches, chosen = [], [], set()
        for label, mode, k, pred in depth_searches():
            matches = [cid for cid in pool if cid not in chosen and pred(convs[cid], tuples[convs[cid]["scenario_id"]])]
            got = rng.sample(matches, min(k, len(matches)))
            chosen.update(got)
            picks += [{"cid": c, "reason": f"depth search for {mode}: {label}"} for c in got]
            searches.append({"mode": mode, "setup": label, "requested": k, "pool": len(matches), "picked": len(got)})
        if len(picks) < args.n:
            raise SystemExit(f"depth searches returned only {len(picks)} of {args.n}: {searches}")
        detail = {"seed": args.seed, "searches": searches,
                  "rule": "each search selects the setup where a mode could occur, never a predicted failure; random within it"}
    elif args.method == "stratified":
        if not args.field:
            raise SystemExit("--field is required for stratified")
        tuples = {}
        for line in SCENARIOS.read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                tuples[s["id"]] = s["tuple"]
        strata: dict[str, list[str]] = {}
        for cid in pool:
            strata.setdefault(str(tuples[convs[cid]["scenario_id"]].get(args.field)), []).append(cid)
        values = sorted(strata)
        per, extra = divmod(args.n, len(values))
        rng = random.Random(args.seed)
        picks = []
        for i, v in enumerate(values):
            k = min(per + (1 if i < extra else 0), len(strata[v]))
            for c in rng.sample(strata[v], k):
                picks.append({"cid": c, "reason": f"{args.field}={v} (stratified random pick)"})
        detail = {"seed": args.seed, "field": args.field,
                  "per_value": {v: sum(p["reason"].startswith(f"{args.field}={v} ") for p in picks) for v in values}}
    elif args.method == "uniform":
        chosen = random.Random(args.seed).sample(pool, args.n)
        picks = [{"cid": c, "reason": "uniform random pick"} for c in chosen]
        detail = {"seed": args.seed}
    else:
        nodes = [n for n in cache["graph"]["nodes"] if n["cid"] in set(pool)]
        by_cluster: dict[int, list[dict]] = {}
        for n in sorted(nodes, key=lambda n: (n["dist_to_centroid"], n["cid"])):
            by_cluster.setdefault(n["cluster"], []).append(n)
        order = sorted(by_cluster, key=lambda c: -len(by_cluster[c]))  # largest clusters first in each round
        picks, rank = [], 0
        while len(picks) < args.n and any(len(v) > rank for v in by_cluster.values()):
            for c in order:
                if len(picks) < args.n and len(by_cluster[c]) > rank:
                    n = by_cluster[c][rank]
                    picks.append({"cid": n["cid"], "reason": f"cluster {c} representative (#{rank + 1} nearest centroid)"})
            rank += 1
        detail = {"k": len(cache["graph"]["clusters"]), "features": cache["graph"]["features"],
                  "cluster_sizes": {c["id"]: c["size"] for c in cache["graph"]["clusters"]}}

    batches.append({"name": args.name, "method": args.method, "n": len(picks), **detail,
                    "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "picks": picks})
    all_ids = [p["cid"] for b in batches for p in b["picks"]]
    assert len(all_ids) == len(set(all_ids)), "a conversation appears in two batches"
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{args.name}: {len(picks)} picks ({args.method}); sample now {len(all_ids)} distinct conversations")
    for p in picks:
        c = convs[p["cid"]]
        print(f"  {c['scenario_id']}  {c['role']:<8} {c['group']:<9} turns={c['features']['turns']} "
              f"tools={c['features']['tool_calls']}  {p['reason']}")
    rows = [convs[p["cid"]] for p in picks]
    print("  roles:", dict(Counter(c["role"] for c in rows)), " groups:", dict(Counter(c["group"] for c in rows)))


if __name__ == "__main__":
    main()
