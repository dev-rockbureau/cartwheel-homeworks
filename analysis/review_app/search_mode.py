"""Part D search for additional instances of reply_contradicts_tool_result (B). Offline, deterministic, no model.

The signals are a RETRIEVAL step, never a label: every candidate goes to analysis/state/suggestions.json as a pending
AI suggestion, and the student accepts, revises, or rejects it in the review app.

Signals (checked per turn; the first firing turn of a conversation becomes the candidate):
    S1 wrong order    the reply names an order id from the tool results whose product title does not match what the
                      user named, while an order whose title the user named is in the same results
    S2 false absence  the reply says it cannot find / does not see the order, while the tool results contain an order
                      whose title the user named
    S3 queued vs done issue_refund returned queued_for_approval, but the reply says processed / completed / issued and
                      never says queued, pending, or approval

"The user named a title" means every word of the product title appears in the user's messages so far (lower-cased).
Titles come from the seed database (read-only); the agent's tools return product ids only.

v2 (after the student revised the boundary of reply_contradicts_tool_result) adds what v1 could not retrieve:
    typo tolerance    one title word may be misspelled in the user's text (similarity >= 0.8), e.g. "Rustc Beanie"
    S4 wrong store    the reply presents one order but names a store the user named that is not that order's store
                      (e.g. "Store: Trailhead Supply" on a Northwind Books order)
Conversations that already have a suggestion from an earlier version are skipped.

Usage (from the repository root):
    uv run python -m analysis.review_app.search_mode            # dry run: counts only, writes nothing
    uv run python -m analysis.review_app.search_mode --write    # append suggestions + add unreviewed hits as a batch
    uv run python -m analysis.review_app.search_mode --version v2 [--write]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache" / "conversations.json"
STATE = ROOT / "analysis" / "state"
MODE = "reply_contradicts_tool_result"

ABSENT = re.compile(r"(don['’]t see|do not see|couldn['’]t find|could not find|can['’]t find|cannot find|unable to find|"
                    r"none of them|no (recent )?orders? for|not in your (recent )?order|isn['’]t in your|not listed|appear to be missing)", re.I)
DONE = re.compile(r"(successfully (processed|issued|completed|refunded)|has been (processed|issued|completed|refunded)|"
                  r"refund (is|was) (processed|complete|completed|issued))", re.I)
QUEUED = re.compile(r"(queued|pending|approval|review)", re.I)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def title_named(title: str, user_words: set[str], typo: bool) -> bool:
    """Every title word appears in the user's text; with typo=True one word may be misspelled."""
    missing = [w for w in words(title) if w not in user_words]
    if not missing:
        return True
    return typo and len(missing) == 1 and any(difflib.SequenceMatcher(None, missing[0], u).ratio() >= 0.8 for u in user_words)


def orders_in(result) -> list[dict]:
    if not isinstance(result, dict):
        return []
    out = [o for o in result.get("orders") or [] if isinstance(o, dict)]
    if isinstance(result.get("order"), dict):
        out.append(result["order"])
    return [o for o in out if "order_id" in o and "product_id" in o]


def line_with(reply: str, needle: str) -> str:
    """The reply line containing needle, so the app can highlight an exact substring."""
    for line in reply.splitlines():
        if needle.lower() in line.lower():
            return line.strip()
    return reply.strip().splitlines()[0]


def scan(conv: dict, titles: dict[int, str], stores: dict[int, str] | None = None, v2: bool = False) -> dict | None:
    user_so_far, seen_orders, queued = "", [], False
    for n, turn in enumerate(conv["turns"], start=1):
        user_so_far += " " + turn["user"]
        uw = words(user_so_far)
        for step in turn["steps"]:
            seen_orders += orders_in(step.get("result"))
            r = step.get("result") or {}
            if step["name"] == "issue_refund" and isinstance(r, dict) and r.get("status") == "queued_for_approval":
                queued = True
        reply = turn["reply"] or ""
        named = {o["order_id"] for o in seen_orders if titles.get(o["product_id"]) and title_named(titles[o["product_id"]], uw, v2)}
        by_id = {o["order_id"]: o for o in seen_orders}
        label = lambda ids: ", ".join(f"{j} ({titles.get(by_id[j]['product_id'])})" for j in sorted(ids))
        mentioned = [int(x) for x in re.findall(r"\b(\d{2,5})\b", reply) if int(x) in by_id]
        wrong = [i for i in mentioned if i not in named]
        if named and wrong:
            i = wrong[0]
            return {"signal": "S1 wrong order", "turn": n, "quote": line_with(reply, str(i)),
                    "detail": f"reply names order {i} ({titles.get(by_id[i]['product_id'])}); the user named "
                              f"{label(named)}"}
        m = ABSENT.search(reply)
        if m and named:
            return {"signal": "S2 false absence", "turn": n, "quote": line_with(reply, m.group(0)),
                    "detail": f"reply says '{m.group(0)}'; results contain "
                              f"{label(named)}"}
        if v2 and stores:
            distinct = list(dict.fromkeys(mentioned))
            low_reply, low_user = reply.lower(), user_so_far.lower()
            if len(distinct) == 1:
                o = by_id[distinct[0]]
                wrong_store = [nm for sid, nm in stores.items()
                               if sid != o.get("store_id") and nm.lower() in low_reply and nm.lower() in low_user]
                if wrong_store:
                    return {"signal": "S4 wrong store", "turn": n, "quote": line_with(reply, wrong_store[0]),
                            "detail": f"reply presents order {o['order_id']} ({titles.get(o['product_id'])}, "
                                      f"{stores.get(o.get('store_id'))}) under the store the user named, {wrong_store[0]}"}
        d = DONE.search(reply)
        if queued and d and not QUEUED.search(reply):
            return {"signal": "S3 queued vs done", "turn": n, "quote": line_with(reply, d.group(0)),
                    "detail": f"issue_refund returned queued_for_approval; reply says '{d.group(0)}'"}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="append suggestions and add unreviewed hits as a search batch")
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    v2 = args.version == "v2"
    source = f"D4 search {args.version}"
    batch_name = "search-B" if not v2 else "search-B-v2"

    db = sqlite3.connect(f"file:{ROOT / 'data' / 'cartwheel.db'}?mode=ro", uri=True)
    titles = {i: t for i, t in db.execute("select id, title from products")}
    stores = {i: n for i, n in db.execute("select id, name from stores")}
    db.close()
    convs = json.loads(CACHE.read_text())["conversations"]
    manifest = json.loads((STATE / "sample_manifest.json").read_text())
    reviewed = {p["cid"] for b in manifest["batches"] for p in b["picks"]}
    patterns = json.loads((STATE / "patterns.json").read_text())
    already = {cid for m in patterns["modes"] if m["name"] == MODE for cid in m["example_trace_ids"]}
    practice = {a["trace_id"] for a in json.loads((STATE / "annotations.json").read_text())["annotations"]
                if a.get("batch") == "practice"}
    sugg_file = STATE / "suggestions.json"
    suggs = json.loads(sugg_file.read_text()) if sugg_file.exists() else []
    suggested = {s["cid"] for s in suggs if s.get("mode") == MODE}  # decided or pending from an earlier version

    hits = []
    for c in convs:
        if c["cid"] in already or c["cid"] in practice or c["cid"] in suggested:
            continue
        h = scan(c, titles, stores, v2)
        if h:
            hits.append((c, h))

    by_signal: dict[str, list[str]] = {}
    for c, h in hits:
        tag = "reviewed" if c["cid"] in reviewed else "unreviewed"
        by_signal.setdefault(h["signal"], []).append(f"{c['scenario_id'][-4:]}({tag})")
    print(f"{source}: {len(hits)} candidates from {len(convs)} conversations "
          f"(excluded: {len(already)} already in {MODE}, {len(suggested)} already suggested, practice {len(practice)})")
    for s, ids in sorted(by_signal.items()):
        print(f"  {s}: {len(ids)}  {' '.join(ids)}")
    if not args.write:
        print("dry run: nothing written")
        return

    have = {(s["cid"], s.get("source")) for s in suggs}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    added, new_picks = 0, []
    for c, h in hits:
        if (c["cid"], source) in have:
            continue
        suggs.append({"id": f"sug-d4{args.version}-{c['scenario_id'][-4:]}", "cid": c["cid"], "msg_key": f"t{h['turn']}.reply",
                      "quote": h["quote"], "mode": MODE,
                      "note": f"(B) {h['detail']}.",
                      "rationale": f"retrieval signal {h['signal']}; a hypothesis to check, not a label",
                      "source": source, "created_at": now, "status": "pending", "decision_reason": None,
                      "decided_at": None, "annotation_id": None})
        added += 1
        if c["cid"] not in reviewed:
            new_picks.append({"cid": c["cid"], "reason": f"{source}: {h['signal']} (retrieval signal, not a label)"})
    sugg_file.write_text(json.dumps(suggs, indent=2) + "\n")
    if new_picks:
        manifest["batches"].append({"name": batch_name, "method": "search", "n": len(new_picks), "mode": MODE,
                                    "source": source, "rule": f"deterministic retrieval signals {'S1-S4 with typo tolerance' if v2 else 'S1-S3'} (see search_mode.py); "
                                    "every candidate reviewed by the student, accepted or rejected in suggestions.json",
                                    "selected_at": now, "picks": new_picks})
        ids = [p["cid"] for b in manifest["batches"] for p in b["picks"]]
        assert len(ids) == len(set(ids)), "a conversation appears in two batches"
        (STATE / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {added} suggestions; {len(new_picks)} unreviewed conversations added as batch {batch_name}")


if __name__ == "__main__":
    main()
