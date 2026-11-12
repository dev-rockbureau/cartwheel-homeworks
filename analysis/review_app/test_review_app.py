"""Offline tests for the HW4 review app (no Langfuse, no model calls).

Run: uv run pytest analysis/review_app/test_review_app.py -q
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from analysis.review_app import load_traces as lt


def _trace(tid, ts, session, scenario, msgs, output=None):
    return {"trace_id": tid, "timestamp": ts, "meta": {"scenario_id": scenario, "session_id": session, "role": "shopper"},
            "trace": msgs, "features": {"tokens": 10}, "models": ["m"], "output": output}


TURN1 = [
    {"role": "user", "text": "refund my planner"},
    {"role": "assistant", "text": "let me look"},
    {"role": "tool_call", "name": "list_my_orders", "arguments": {}},
    {"role": "tool_result", "name": "list_my_orders", "content": {"ok": True, "count": 1}},
    {"role": "assistant", "text": "it is eligible"},
    {"role": "tool_call", "name": "issue_refund", "arguments": {"order_id": 1}},
    {"role": "tool_result", "name": "issue_refund", "content": {"ok": True, "status": "auto_approved"}},
    {"role": "assistant", "text": "Refund done."},
]
TURN2 = [{"role": "user", "text": "again"}, {"role": "assistant", "text": "Already done."}]


def test_turn_pairs_calls_with_results_and_marks_interim():
    t = lt.build_turn(_trace("a", "1", "s", "support-1", TURN1))
    assert t["user"] == "refund my planner"
    assert [s["name"] for s in t["steps"]] == ["list_my_orders", "issue_refund"]
    assert t["steps"][0]["interim"] == ["let me look"] and t["steps"][1]["interim"] == ["it is eligible"]
    assert t["steps"][1]["result"]["status"] == "auto_approved"
    assert t["reply"] == "Refund done."  # only the last assistant message is the reply the user saw


def test_parallel_calls_pair_by_name_in_order():
    msgs = [{"role": "user", "text": "q"},
            {"role": "tool_call", "name": "get_order", "arguments": {"order_id": 1}},
            {"role": "tool_call", "name": "get_order", "arguments": {"order_id": 2}},
            {"role": "tool_result", "name": "get_order", "content": {"order": {"order_id": 1}}},
            {"role": "tool_result", "name": "get_order", "content": {"order": {"order_id": 2}}},
            {"role": "assistant", "text": "done"}]
    t = lt.build_turn(_trace("a", "1", "s", "support-1", msgs))
    assert [s["arguments"]["order_id"] for s in t["steps"]] == [1, 2]
    assert [s["result"]["order"]["order_id"] for s in t["steps"]] == [1, 2]


def test_conversations_group_by_session_in_time_order_and_skip_non_support():
    traces = [_trace("t2", "2026-01-01T00:00:02Z", "S1", "support-0001", TURN2),
              _trace("t1", "2026-01-01T00:00:01Z", "S1", "support-0001", TURN1),
              _trace("p1", "2026-01-01T00:00:03Z", "S2", "pilot-0001", TURN2)]
    convs = lt.build_conversations(traces, {"support-0001": {"scenario_group": "coverage"}})
    assert len(convs) == 1
    c = convs[0]
    assert c["cid"] == "t1" and c["trace_ids"] == ["t1", "t2"] and c["group"] == "coverage"
    assert c["features"]["turns"] == 2 and c["features"]["tool_calls"] == 2 and c["features"]["interim_messages"] == 2


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """Server on a free port with a temporary state dir and a two-conversation cache."""
    from analysis.review_app import server as srv

    convs = lt.build_conversations(
        [_trace("t1", "1", "S1", "support-0029", TURN1), _trace("t3", "3", "S3", "support-0233", TURN2)], {})
    cache = {"source": "test", "conversations": convs, "graph": {"nodes": [], "clusters": [], "features": []}}
    state = tmp_path / "state"
    (state / "labels").mkdir(parents=True)
    (state / "annotations.json").write_text('{"annotations": []}')
    monkeypatch.setenv("REVIEW_STATE_DIR", str(state))
    monkeypatch.setenv("REVIEW_NO_LANGFUSE", "1")
    monkeypatch.setenv("REVIEW_BACKUP_DIR", str(tmp_path / "backup"))
    monkeypatch.setattr(srv, "_cache", cache)
    monkeypatch.setattr(srv, "CACHE_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(srv, "APP_DIR", srv.APP_DIR)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_answer_key_is_locked_until_a_human_note_exists(app):
    base, _ = app
    assert _get(f"{base}/api/answer-key?cid=t1")[0] == 403
    note = {"id": "ann-1", "trace_id": "t1", "author": "human", "kind": "failure", "note": "x", "msg_key": "t1.reply"}
    assert _post(f"{base}/api/annotations", {"annotations": [note]})[0] == 200
    status, key = _get(f"{base}/api/answer-key?cid=t1")
    assert status == 200 and key["scenario_id"] == "support-0029" and "expected" in key


def test_labels_are_append_only_with_supersede_and_failure_is_one(app):
    base, state = app
    s1, r1 = _post(f"{base}/api/labels", {"cid": "t1", "mode": "unconfirmed_write", "label": "fail", "evidence": "q"})
    s2, r2 = _post(f"{base}/api/labels", {"cid": "t1", "mode": "unconfirmed_write", "label": "pass"})
    assert s1 == s2 == 200 and r1["langfuse_score_written"] is False
    rows = [json.loads(x) for x in (state / "labels" / "unconfirmed_write.jsonl").read_text().splitlines()]
    assert [r["label"] for r in rows] == [1, 0]
    assert rows[0]["superseded_by"] == r2["label_id"] and "superseded_by" not in rows[1]
    assert rows[1]["trace_ids"] == ["t1"] and rows[1]["session_id"] == "S1"
    live = _get(f"{base}/api/labels")[1]
    assert [r["label"] for r in live["unconfirmed_write"]] == [0]


def test_bulk_labels_save_all_or_nothing(app):
    base, state = app
    bad = {"labels": [{"cid": "t1", "mode": "some_mode", "label": "pass"}, {"cid": "nope", "mode": "some_mode", "label": "fail"}]}
    assert _post(f"{base}/api/labels", bad)[0] == 400
    assert not (state / "labels" / "some_mode.jsonl").exists()  # nothing written when one item is invalid
    ok = {"labels": [{"cid": "t1", "mode": "some_mode", "label": "pass", "evidence": "setup absent"},
                     {"cid": "t3", "mode": "some_mode", "label": "fail"}]}
    status, out = _post(f"{base}/api/labels", ok)
    assert status == 200 and out["count"] == 2 and out["langfuse_scores_written"] == 0
    live = {r["trace_id"]: r["label"] for r in _get(f"{base}/api/labels")[1]["some_mode"]}
    assert live == {"t1": 0, "t3": 1}


def test_label_records_how_it_was_accepted(app):
    base, state = app
    assert _post(f"{base}/api/labels", {"cid": "t1", "mode": "m_one", "label": "pass"})[0] == 200
    bulk = {"labels": [{"cid": "t3", "mode": "m_one", "label": "pass", "accepted_via": "bulk_after_spot_check"}]}
    assert _post(f"{base}/api/labels", bulk)[0] == 200
    assert _post(f"{base}/api/labels", {"cid": "t1", "mode": "m_one", "label": "pass", "accepted_via": "guess"})[0] == 400
    rows = {r["trace_id"]: r["accepted_via"] for r in
            (json.loads(x) for x in (state / "labels" / "m_one.jsonl").read_text().splitlines())}
    assert rows == {"t1": "individual", "t3": "bulk_after_spot_check"}


def test_label_proposals_round_trip(app):
    base, _ = app
    assert _get(f"{base}/api/label_proposals") == (200, [])
    props = [{"id": "lp-1", "cid": "t1", "mode": "some_mode", "label": "pass", "group": 2, "status": "pending"}]
    assert _post(f"{base}/api/label_proposals", props)[0] == 200
    assert _get(f"{base}/api/label_proposals")[1] == props


def test_label_validation(app):
    base, _ = app
    assert _post(f"{base}/api/labels", {"cid": "t1", "mode": "Bad Name", "label": "fail"})[0] == 400
    assert _post(f"{base}/api/labels", {"cid": "nope", "mode": "ok_mode", "label": "fail"})[0] == 400
    assert _post(f"{base}/api/labels", {"cid": "t1", "mode": "ok_mode", "label": "maybe"})[0] == 400
    assert _post(f"{base}/api/annotations", [1, 2])[0] == 400


def test_samples_follow_manifest_batch_order(app):
    base, state = app
    (state / "sample_manifest.json").write_text(json.dumps({"batches": [
        {"name": "uniform", "picks": [{"cid": "t3", "reason": "random"}]},
        {"name": "cluster", "picks": [{"cid": "t1", "reason": "cluster 0"}, {"cid": "t3", "reason": "dup"}]}]}))
    samples = _get(f"{base}/api/samples")[1]
    assert [(s["cid"], s["batch"]) for s in samples] == [("t3", "uniform"), ("t1", "cluster")]


def test_policy_lists_overrides_and_spec_ids(app):
    base, _ = app
    p = _get(f"{base}/api/policy")[1]
    assert p["facts"]["refund_auto_approve_threshold_usd"] == 100
    assert {s["name"] for s in p["stores"]} >= {"Saltbox Pantry", "Northwind Books"}
    assert {"RESP-2", "ESC-1", "AUTH-1"} <= {r["id"] for r in p["spec"]}
    # store ids as returned by tools (Cascade Audio = 5, Atlas Stationery = 20 in the HW3 traces)
    assert p["store_names"]["5"] == "Cascade Audio" and p["store_names"]["20"] == "Atlas Stationery"
    assert len(p["store_names"]) == 20
    # product ids as returned by find_order / list_my_orders (support-0239: order 1313; support-0238: order 5741)
    assert p["product_names"]["484"] == "Handmade Hot Sauce" and p["product_names"]["275"] == "Compact Journal"
    assert len(p["product_names"]) == 800


def test_practice_notes_do_not_unlock_the_key(app):
    base, _ = app
    note = {"id": "ann-p", "trace_id": "t1", "author": "human", "kind": "failure", "note": "x", "batch": "practice"}
    assert _post(f"{base}/api/annotations", {"annotations": [note]})[0] == 200
    assert _get(f"{base}/api/answer-key?cid=t1")[0] == 403
