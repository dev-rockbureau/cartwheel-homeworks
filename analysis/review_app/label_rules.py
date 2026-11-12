"""Part E label proposals, groups 1 and 2. Offline, deterministic, no model. Writes PROPOSALS, never labels.

Group 1  the student's Part D decisions: a mode's positives -> proposed Fail, close negatives -> proposed Pass.
Group 2  setup absent: the situation a mode is about never occurs in the conversation -> proposed Pass. One rule per
         mode, approved by the student (2026-11-11). Rules are deliberately conservative: anything they cannot rule
         out is left for group 3 (read and judged).
Group 3  everything else; proposed after reading (added to label_proposals.json separately) and verified one by one.

A proposal becomes a label only when the student accepts it in the review app (labels/<mode>.jsonl + Langfuse score).

Usage (from the repository root):
    uv run python -m analysis.review_app.label_rules            # dry run: counts per mode and group
    uv run python -m analysis.review_app.label_rules --write    # add pending group 1-2 proposals (existing ones kept)
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache" / "conversations.json"
STATE = ROOT / "analysis" / "state"
SCENARIOS = ROOT / "scenarios" / "support_scenarios.jsonl"

ORDER_TOOLS = {"find_order", "get_order", "list_my_orders", "check_refund_eligibility", "issue_refund", "cancel_order"}
FEE_STORE = re.compile(r"Cascade Audio|Second Stitch|\"store_id\": (5|15)\b", re.I)
ELIGIBILITY = re.compile(r"eligib|return window|can (still )?(be )?return|able to return|refund|days (left|remaining)|"
                         r"deadline|within \d+ days", re.I)
POLICY_WORDS = re.compile(r"\b\d+[- ]days?\b|window|threshold|fee|policy|restock|cancel|refund|return|eligib|dispute|"
                          r"escalat|permission|allowed|only|cannot|can't|merchant|payout|approval|account settings", re.I)
POLICY_QUESTION = re.compile(r"policy|rules?\b|allowed|fee|how long|how many days|cutoff|threshold|"
                             r"can (a|merchants?|i) |what happens", re.I)

# Plain-language rule per mode (shown as the evidence of each group-2 proposal).
RULES = {
    "eligibility_without_order_lookup": "intent is not return/refund and no reply mentions eligibility, a return window, a deadline, or a refund",
    "reply_contradicts_tool_result": "no order tool was called, so no order fact could be contradicted",
    "omitted_restocking_fee": "no fee store (Cascade Audio, Second Stitch Apparel; store 5 or 15) appears in the conversation",
    "policy_claim_without_citation": "no reply contains a policy-like word (days, window, fee, policy, refund, return, cancel, dispute, escalate, permission, only, cannot, merchant, approval, ...)",
    "ambiguous_match_presented_as_unique": "no result had 2+ records, no search used limit=1, and the scenario input is not a duplicate-title or two-matching-orders case",
    "required_escalation_not_made": "intent is not account change, account info, or dispute, and the answer key does not expect escalation",
    "policy_question_answered_without_lookup": "intent is not a policy or account question and the user never asks about a policy, rule, fee, cutoff, 'how long', or 'what happens'",
    "absence_claimed_after_bad_search_arguments": "no lookup returned an empty or failed result",
}


def _steps(conv: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for t in conv["turns"] for s in t["steps"]]


def setup_present(conv: dict[str, Any], scenario: dict[str, Any]) -> dict[str, bool]:
    tup, expected = scenario["tuple"], scenario.get("expected")
    steps = _steps(conv)
    replies = " ".join(t["reply"] or "" for t in conv["turns"])
    user = " ".join(t["user"] for t in conv["turns"])

    def multi(r: Any) -> bool:
        return isinstance(r, dict) and len(r.get("orders") or r.get("products") or []) >= 2

    def empty(r: Any) -> bool:
        return isinstance(r, dict) and (r.get("ok") is False or r.get("orders") == [] or r.get("products") == [])

    return {
        "eligibility_without_order_lookup": tup["intent"] in ("return_eligibility", "refund") or bool(ELIGIBILITY.search(replies)),
        "reply_contradicts_tool_result": any(s["name"] in ORDER_TOOLS for s in steps),
        "omitted_restocking_fee": bool(FEE_STORE.search(json.dumps(conv))),
        "policy_claim_without_citation": bool(POLICY_WORDS.search(replies)),
        "ambiguous_match_presented_as_unique": tup.get("record_state") in ("product_duplicate_title", "ambiguous_two_matching_orders")
        or any(multi(s.get("result")) or s["arguments"].get("limit") == 1 for s in steps),
        "required_escalation_not_made": tup["intent"] in ("account_change", "dispute", "account_info")
        or "escalat" in json.dumps(expected).lower(),
        "policy_question_answered_without_lookup": tup["intent"] in ("policy_question", "account_info", "account_change")
        or bool(POLICY_QUESTION.search(user)),
        "absence_claimed_after_bad_search_arguments": any(empty(s.get("result")) for s in steps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    convs = {c["cid"]: c for c in json.loads(CACHE.read_text())["conversations"]}
    scenarios = {json.loads(x)["id"]: json.loads(x) for x in SCENARIOS.read_text().splitlines() if x.strip()}
    manifest = json.loads((STATE / "sample_manifest.json").read_text())
    review_set = [p["cid"] for b in manifest["batches"] for p in b["picks"]]
    modes = [m for m in json.loads((STATE / "patterns.json").read_text())["modes"] if m["status"] == "confirmed"]
    assert set(RULES) == {m["name"] for m in modes}, "every confirmed mode needs exactly one approved rule"

    path = STATE / "label_proposals.json"
    props = json.loads(path.read_text()) if path.exists() else []
    have = {(p["mode"], p["cid"]) for p in props}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new: list[dict[str, Any]] = []
    for m in modes:
        pos, cn = set(m["example_trace_ids"]), set(m["close_negative_trace_ids"])
        counts = {1: 0, 2: 0, 3: 0}
        for cid in review_set:
            conv = convs[cid]
            sid = conv["scenario_id"]
            if cid in pos or cid in cn:
                group, label = 1, ("fail" if cid in pos else "pass")
                evidence = f"Part D: {'positive' if cid in pos else 'close negative'} of {m['name']} (your decision)"
            elif not setup_present(conv, scenarios[sid])[m["name"]]:
                group, label, evidence = 2, "pass", f"setup absent: {RULES[m['name']]}"
            else:
                counts[3] += 1
                continue
            counts[group] += 1
            if (m["name"], cid) not in have:
                new.append({"id": f"lp-{m['name']}-{sid[-4:]}", "cid": cid, "scenario_id": sid, "mode": m["name"],
                            "label": label, "group": group, "confidence": "high", "evidence": evidence,
                            "source": "label_rules.py" if group == 2 else "patterns.json (Part D)",
                            "created_at": now, "status": "pending", "decided_at": None})
        print(f"{m['name']:<44} g1 {counts[1]:>3}  g2 {counts[2]:>3}  g3 {counts[3]:>3}")
    print(f"{len(new)} new group 1-2 proposals for {len(review_set)} conversations x {len(modes)} modes")
    if args.write:
        path.write_text(json.dumps(props + new, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)} ({len(props) + len(new)} proposals)")
    else:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
