"""HW4 review app server (Python standard library only).

Adapted from the reference ``analysis/server.py`` (same file-backed JSON API, so the skill's agent loop still
works), with changes made after inspecting the Cartwheel traces:

* conversations are grouped by session (one Langfuse trace per user turn) and pre-built by ``load_traces.py``;
* free-text annotations never write Langfuse scores; only explicit Pass/Fail labels do (``POST /api/labels``);
* labels are kept append-only per mode in ``analysis/state/labels/<mode>.jsonl`` (1 = failure present);
* the HW3 answer key for a conversation is served only after it has a human note.

Run from the repository root:
    uv run python -m analysis.review_app.load_traces     # once, builds the cache from Langfuse
    uv run python -m analysis.review_app.server          # http://localhost:8030

Environment:
    REVIEW_STATE_DIR   state directory (default analysis/state); tests point it at a temporary copy
    REVIEW_BACKUP_DIR  where the rolling annotations backup goes (default analysis/review_app/cache)
    REVIEW_NO_LANGFUSE set to 1 to skip Langfuse score writes (labels then save locally with langfuse=false)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[1]
CACHE_FILE = APP_DIR / "cache" / "conversations.json"
SCENARIOS = ROOT / "scenarios" / "support_scenarios.jsonl"
LOCK = threading.Lock()
MODE_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
# How a label was accepted (Part E provenance): one cell at a time, or in bulk from agent proposals.
ACCEPTED_VIA = {"individual", "bulk_group1", "bulk_group2", "bulk_after_spot_check"}


def state_dir() -> Path:
    return Path(os.environ.get("REVIEW_STATE_DIR", ROOT / "analysis" / "state"))


FILES = {  # whole-document endpoints (GET returns, POST replaces), shared with the agent
    "/api/annotations": ("annotations.json", {"annotations": []}),
    "/api/patterns": ("patterns.json", {"modes": []}),
    "/api/suggestions": ("suggestions.json", []),
    # Part E: agent-proposed labels; a proposal becomes a label only when the student accepts it (POST /api/labels)
    "/api/label_proposals": ("label_proposals.json", []),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else default
    except FileNotFoundError:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- data

_cache: dict[str, Any] | None = None


def cache() -> dict[str, Any]:
    global _cache
    if _cache is None:
        if not CACHE_FILE.exists():
            raise FileNotFoundError("no conversation cache; run: uv run python -m analysis.review_app.load_traces")
        _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return _cache


def conversations_by_cid() -> dict[str, dict[str, Any]]:
    return {c["cid"]: c for c in cache()["conversations"]}


def annotations() -> list[dict[str, Any]]:
    data = read_json(state_dir() / "annotations.json", {"annotations": []})
    return data.get("annotations", []) if isinstance(data, dict) else data


def manifest() -> dict[str, Any]:
    return read_json(state_dir() / "sample_manifest.json", {"batches": []})


def samples() -> list[dict[str, Any]]:
    """Sample conversations in batch order, from ``sample_manifest.json`` (written in Part B)."""
    out, seen = [], set()
    for b in manifest().get("batches", []):
        for pick in b.get("picks", []):
            cid = pick.get("cid") or pick.get("trace_id")
            if cid in seen:
                continue
            seen.add(cid)
            out.append({"cid": cid, "batch": b.get("name"), "reason": pick.get("reason", "")})
    return out


def policy() -> dict[str, Any]:
    """Facts a reviewer needs beside the trace: facts.yaml numbers, store overrides, SPEC requirement ids."""
    import yaml

    facts = yaml.safe_load((ROOT / "facts.yaml").read_text())
    from seed.generate import CORE_STORES

    stores = [{"id": i, "name": n, "return_window_days": o if o is not None else facts["return_window_days"],
               "override": o is not None, "restocking_fee_opt_in": r}
              for i, (n, _cat, o, r) in enumerate(CORE_STORES, start=1)]
    spec = []
    for line in (ROOT / "SPEC.md").read_text().splitlines():
        m = re.search(r"\*\*([A-Z]+-\d+)\.\*\*\s*(.*)", line)
        if m:
            spec.append({"id": m.group(1), "text": m.group(2).strip()})
    keys = ["return_window_days", "refund_auto_approve_threshold_usd", "cancel_cutoff", "dispute_window_days",
            "restocking_fee_max_percent", "restocking_fee_opened_items_only", "refund_method",
            "refund_processing_days_min", "refund_processing_days_max", "support_escalation_sla_hours"]
    return {"world_date": "2026-07-01", "facts": {k: facts.get(k) for k in keys},
            "stores": [s for s in stores if s["override"] or s["restocking_fee_opt_in"]], "spec": spec,
            # every store id -> name, so tool results that carry only store_id can be read (e.g. find_order)
            "store_names": {str(s["id"]): s["name"] for s in stores},
            # every product id -> title (find_order / list_my_orders return product_id only)
            "product_names": _product_names()}


def _product_names() -> dict[str, str]:
    """product id -> title from the seed database, opened read-only. Titles repeat across stores; keep the id."""
    import sqlite3

    db = sqlite3.connect(f"file:{ROOT / 'data' / 'cartwheel.db'}?mode=ro", uri=True)
    try:
        return {str(i): t for i, t in db.execute("select id, title from products")}
    finally:
        db.close()


def answer_key(cid: str) -> tuple[int, dict[str, Any]]:
    conv = conversations_by_cid().get(cid)
    if conv is None:
        return 404, {"error": f"unknown conversation {cid}"}
    if not any(a.get("trace_id") == cid and a.get("author") == "human" and a.get("batch") != "practice"
               for a in annotations()):
        return 403, {"error": "locked: write a note or 'no failure observed' for this conversation first"}
    for line in SCENARIOS.read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            if s["id"] == conv["scenario_id"]:
                return 200, {"scenario_id": s["id"], "group": s["scenario_group"], "tuple": s["tuple"],
                             "expected": s["expected"], "data_quality_case_id": s.get("data_quality_case_id")}
    return 404, {"error": "scenario not found"}


# ---------------------------------------------------------------- labels

def labels_dir() -> Path:
    return state_dir() / "labels"


def read_labels() -> dict[str, list[dict[str, Any]]]:
    """Live label per (mode, conversation): superseded records dropped, last write wins."""
    out: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(labels_dir().glob("*.jsonl")):
        live: dict[str, dict[str, Any]] = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("source") == "human_hw4" and not r.get("superseded_by"):
                    live[r["trace_id"]] = r
        if live:
            out[p.stem] = list(live.values())
    return out


def write_langfuse_score(cid: str, mode: str, value: int, comment: str | None, flush: bool = True) -> bool:
    if os.environ.get("REVIEW_NO_LANGFUSE") == "1":
        return False
    from observability.instrument import load_env

    load_env()
    from analysis.helpers import langfuse_io

    if not langfuse_io.is_configured():
        return False
    lf = langfuse_io._client()
    # A deterministic score id makes a relabel overwrite the earlier score instead of adding a second one.
    lf.create_score(name=mode, value=int(value), trace_id=cid, data_type="NUMERIC", comment=comment,
                    score_id=f"hw4-{mode}-{cid}")
    if flush:
        lf.flush()
    return True


def flush_langfuse() -> None:
    if os.environ.get("REVIEW_NO_LANGFUSE") == "1":
        return
    from analysis.helpers import langfuse_io

    if langfuse_io.is_configured():
        langfuse_io._client().flush()


def save_label(body: dict[str, Any], flush: bool = True) -> tuple[int, dict[str, Any]]:
    mode, cid, label = body.get("mode"), body.get("cid"), body.get("label")
    if not isinstance(mode, str) or not MODE_NAME.match(mode):
        return 400, {"error": "mode must be a snake_case name"}
    conv = conversations_by_cid().get(cid)
    if conv is None:
        return 400, {"error": f"unknown conversation {cid}"}
    if label not in ("pass", "fail"):
        return 400, {"error": "label must be 'pass' or 'fail'"}
    value = 1 if label == "fail" else 0  # stored convention: 1 = failure present
    evidence = (body.get("evidence") or "").strip() or None
    via = body.get("accepted_via") or "individual"
    if via not in ACCEPTED_VIA:
        return 400, {"error": f"accepted_via must be one of {sorted(ACCEPTED_VIA)}"}
    path = labels_dir() / f"{mode}.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    prior = [r for r in rows if r.get("trace_id") == cid and not r.get("superseded_by")]
    label_id = f"{cid}#{sum(r.get('trace_id') == cid for r in rows)}"
    try:
        synced = write_langfuse_score(cid, mode, value, evidence, flush=flush)
        error = None
    except Exception as exc:  # keep the local record; report that the canonical write failed
        synced, error = False, f"{type(exc).__name__}: {exc}"
    for r in prior:
        r["superseded_by"] = label_id
    rows.append({"trace_id": cid, "label": value, "source": "human_hw4", "ts": now(), "label_id": label_id,
                 "mode": mode, "session_id": conv["session_id"], "trace_ids": conv["trace_ids"],
                 "scenario_id": conv["scenario_id"], "evidence": evidence, "accepted_via": via, "langfuse_score": synced})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    status = 502 if error else 200
    return status, {"ok": error is None, "label_id": label_id, "langfuse_score_written": synced,
                    "error": error, "cached_locally": True}


def save_labels(items: list[Any]) -> tuple[int, dict[str, Any]]:
    """Bulk accept: validate all first (nothing is written if one is invalid), then save each, one Langfuse flush."""
    if not items:
        return 400, {"error": "expected a non-empty list of labels"}
    convs = conversations_by_cid()
    for it in items:
        if (not isinstance(it, dict) or not isinstance(it.get("mode"), str) or not MODE_NAME.match(it["mode"])
                or it.get("cid") not in convs or it.get("label") not in ("pass", "fail")
                or (it.get("accepted_via") or "individual") not in ACCEPTED_VIA):
            return 400, {"error": f"invalid label in batch: {it!r}"}
    results = [save_label(it, flush=False)[1] for it in items]
    try:
        flush_langfuse()
    except Exception as exc:  # labels are on disk; report that the Langfuse flush failed
        return 502, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "results": results}
    errors = [r for r in results if not r.get("ok")]
    return (502 if errors else 200), {"ok": not errors, "count": len(results), "results": results,
                                      "langfuse_scores_written": sum(bool(r.get("langfuse_score_written")) for r in results)}


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002 - quiet server
        pass

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Any:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"null")
        except json.JSONDecodeError:
            return None

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                body = (APP_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/conversations":
                c = cache()
                self._json({"source": c["source"], "fallback_reason": c.get("fallback_reason"),
                            "conversations": c["conversations"],
                            "langfuse_host": os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
                            "langfuse_project": os.environ.get("LANGFUSE_PROJECT_ID", "cartwheel-dev")})
            elif url.path == "/api/graph":
                self._json(cache()["graph"])
            elif url.path == "/api/samples":
                self._json(samples())
            elif url.path == "/api/manifest":
                self._json(manifest())
            elif url.path == "/api/policy":
                self._json(policy())
            elif url.path == "/api/answer-key":
                status, data = answer_key((q.get("cid") or [""])[0])
                self._json(data, status)
            elif url.path == "/api/labels":
                self._json(read_labels())
            elif url.path in FILES:
                name, default = FILES[url.path]
                self._json(read_json(state_dir() / name, default))
            else:
                self._json({"error": f"not found: {url.path}"}, 404)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, 503)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        data = self._body()
        if data is None:
            self._json({"error": "expected a JSON body"}, 400)
            return
        with LOCK:
            if path == "/api/labels":
                status, out = save_labels(data["labels"]) if isinstance(data, dict) and "labels" in data else save_label(data)
                self._json(out, status)
            elif path in FILES:
                name, _default = FILES[path]
                target = state_dir() / name
                if path == "/api/annotations":
                    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
                        self._json({"error": "expected {\"annotations\": [...]}"}, 400)
                        return
                    if target.exists():  # rolling backup so a bad client write can be undone
                        backup = Path(os.environ.get("REVIEW_BACKUP_DIR", APP_DIR / "cache")) / "annotations.backup.json"
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(target, backup)
                write_json(target, data)
                count = len(data.get("annotations", data.get("modes", []))) if isinstance(data, dict) else len(data)
                self._json({"ok": True, "count": count})
            else:
                self._json({"error": f"cannot POST to {path}"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8030)
    args = ap.parse_args()
    cache()  # fail fast if the cache is missing
    print(f"review app: http://localhost:{args.port}  (state: {state_dir()})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
