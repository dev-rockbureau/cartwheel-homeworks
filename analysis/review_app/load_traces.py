"""Build the review app's conversation cache from Langfuse (canonical) or the HW3 export (fallback).

One record per conversation (= one Cartwheel session). Cartwheel writes one Langfuse trace per user turn, so the
traces of a session are grouped by ``session_id`` and ordered by timestamp. Inside each turn, tool calls are paired
with their results into numbered steps, and assistant text is split into the reply the user saw (the last assistant
message of the turn) and the model's interim text between tool calls (never shown to the user).

Usage (from the repository root):
    uv run python -m analysis.review_app.load_traces                 # Langfuse, falls back to the export
    uv run python -m analysis.review_app.load_traces --source export # force the HW3 export
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from analysis.helpers.normalization import normalize_trace

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_FILE = CACHE_DIR / "conversations.json"
EXPORT = ROOT / "traces" / "support_traces.json"
SCENARIOS = ROOT / "scenarios" / "support_scenarios.jsonl"
PREFIX = "support-"


# ---------------------------------------------------------------- turn structure

def build_turn(trace: dict[str, Any], reply_text: str | None = None) -> dict[str, Any]:
    """Split one normalized trace (one user turn) into user text, numbered steps, and the reply."""
    msgs = trace.get("trace") or []
    users = [m.get("text", "") for m in msgs if m.get("role") == "user"]
    assistant_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    reply_i = assistant_idx[-1] if assistant_idx else None
    if reply_text is not None and reply_i is not None and msgs[reply_i].get("text", "").strip() != reply_text.strip():
        # The trace output is what the user saw; trust it over message order.
        match = [i for i in assistant_idx if msgs[i].get("text", "").strip() == reply_text.strip()]
        reply_i = match[-1] if match else reply_i

    steps: list[dict[str, Any]] = []
    pending_interim: list[str] = []
    trailing_interim: list[str] = []
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "assistant" and i != reply_i:
            pending_interim.append(m.get("text", ""))
        elif role == "tool_call":
            steps.append({"n": len(steps) + 1, "interim": pending_interim, "name": m.get("name"),
                          "arguments": m.get("arguments"), "result": None, "has_result": False})
            pending_interim = []
        elif role == "tool_result":
            open_steps = [s for s in steps if s["name"] == m.get("name") and not s["has_result"]]
            if open_steps:
                open_steps[0]["result"], open_steps[0]["has_result"] = m.get("content"), True
            else:  # result without a recorded call: keep it visible as its own step
                steps.append({"n": len(steps) + 1, "interim": pending_interim, "name": m.get("name"),
                              "arguments": None, "result": m.get("content"), "has_result": True})
                pending_interim = []
    trailing_interim = pending_interim
    return {
        "trace_id": trace["trace_id"],
        "timestamp": trace.get("timestamp"),
        "user": "\n\n".join(users),
        "steps": steps,
        "interim_after_steps": trailing_interim,
        "reply": msgs[reply_i].get("text", "") if reply_i is not None else "",
        "models": trace.get("models", []),
    }


def _output_text(trace: dict[str, Any]) -> str | None:
    """The reply text recorded as the trace output, when present."""
    out = trace.get("output")
    if isinstance(out, str):
        return out
    if isinstance(out, list):  # [{"parts": [{"content": ...}], "role": "assistant"}]
        texts = [p.get("content", "") for o in out if isinstance(o, dict) for p in o.get("parts", [])
                 if isinstance(p, dict) and p.get("type", "text") == "text"]
        return "\n".join(texts) if texts else None
    return None


def build_conversations(traces: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Group normalized ``support-*`` traces by session into conversation records."""
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        sid = t.get("meta", {}).get("scenario_id") or ""
        if not sid.startswith(PREFIX):
            continue
        by_session[t["meta"].get("session_id") or t["trace_id"]].append(t)

    convs = []
    for session_id, group in by_session.items():
        group.sort(key=lambda t: (t.get("timestamp") or "", t["trace_id"]))
        turns = [build_turn(t, _output_text(t)) for t in group]
        first = group[0]
        scenario_id = first["meta"]["scenario_id"]
        sc = scenarios.get(scenario_id, {})
        tool_calls = sum(len(t["steps"]) for t in turns)
        convs.append({
            "cid": first["trace_id"],  # the conversation's first trace id = the id labels and scores use
            "session_id": session_id,
            "scenario_id": scenario_id,
            "trace_ids": [t["trace_id"] for t in group],
            "role": first["meta"].get("role"),
            "group": sc.get("scenario_group"),
            "models": sorted({m for t in turns for m in t["models"]}),
            "features": {
                "turns": len(turns),
                "tool_calls": tool_calls,
                "distinct_tools": len({s["name"] for t in turns for s in t["steps"]}),
                "tokens": sum(int(t.get("features", {}).get("tokens") or 0) for t in group),
                "interim_messages": sum(len(s["interim"]) for t in turns for s in t["steps"])
                + sum(len(t["interim_after_steps"]) for t in turns),
            },
            "turns": turns,
        })
    convs.sort(key=lambda c: c["scenario_id"])
    _add_outlier_flags(convs)
    return convs


def _add_outlier_flags(convs: list[dict[str, Any]]) -> None:
    """Header badges only for clear outliers (top or bottom 5% on a structural feature)."""
    labels = {"tool_calls": "tool calls", "tokens": "words", "interim_messages": "interim messages"}
    for key, word in labels.items():
        vals = sorted(c["features"][key] for c in convs)
        n = len(vals)
        if n < 20:
            continue
        hi, lo = vals[int(n * 0.95)], vals[int(n * 0.05)]
        for c in convs:
            v = c["features"][key]
            share = sum(x < v for x in vals) / n
            if v > hi and v > 0:
                c.setdefault("flags", []).append(f"{v} {word} (more than {round(share * 100)}%)")
            elif v < lo:
                c.setdefault("flags", []).append(f"{v} {word} (fewer than {round((1 - share) * 100)}%)")


# ---------------------------------------------------------------- map projection

def project(convs: list[dict[str, Any]], k: int = 8, seed: int = 7) -> dict[str, Any]:
    """2D PCA of standardized structural features plus k-means clusters, for the map view."""
    import numpy as np

    keys = ["turns", "tool_calls", "distinct_tools", "tokens", "interim_messages"]
    X = np.array([[float(c["features"][k_]) for k_ in keys] for c in convs])
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    xy = X @ vt[:2].T
    rng = random.Random(seed)
    centers = X[rng.sample(range(len(X)), k)]
    for _ in range(50):
        assign = ((X[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[assign == j].mean(0) if (assign == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    dist = ((X - centers[assign]) ** 2).sum(-1)
    nodes = [{"cid": c["cid"], "scenario_id": c["scenario_id"], "role": c["role"], "group": c["group"],
              "x": float(xy[i, 0]), "y": float(xy[i, 1]), "cluster": int(assign[i]),
              "dist_to_centroid": float(dist[i])} for i, c in enumerate(convs)]
    clusters = [{"id": j, "size": int((assign == j).sum()),
                 "centroid_features": {k_: round(float(v), 2) for k_, v in zip(keys, centers[j])}} for j in range(k)]
    return {"features": keys, "nodes": nodes, "clusters": clusters,
            "explained": [float(x) for x in (vt[:2] @ np.cov(X.T) @ vt[:2].T).diagonal()]}


# ---------------------------------------------------------------- sources

def load_scenarios(path: Path = SCENARIOS) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def from_langfuse() -> list[dict[str, Any]]:
    from observability.instrument import load_env

    load_env()
    from analysis.helpers.langfuse_io import fetch_traces, is_configured

    if not is_configured():
        raise RuntimeError("Langfuse is not configured (LANGFUSE_* in .env)")
    return fetch_traces(limit=5000)


def from_export(path: Path = EXPORT) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return [normalize_trace(t) for t in data["traces"]]


def build_cache(source: str = "langfuse") -> dict[str, Any]:
    reason = None
    if source == "langfuse":
        try:
            traces, used = from_langfuse(), "langfuse"
        except Exception as exc:  # documented fallback: Langfuse temporarily unavailable
            reason = f"Langfuse unavailable ({type(exc).__name__}: {exc}); used the HW3 export"
            traces, used = from_export(), "export"
    else:
        traces, used = from_export(), "export"
    convs = build_conversations(traces, load_scenarios())
    cache = {"source": used, "fallback_reason": reason, "conversation_count": len(convs),
             "trace_count": sum(len(c["trace_ids"]) for c in convs),
             "conversations": convs, "graph": project(convs)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False))
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["langfuse", "export"], default="langfuse")
    args = ap.parse_args()
    cache = build_cache(args.source)
    print(f"source={cache['source']} conversations={cache['conversation_count']} traces={cache['trace_count']}"
          + (f"\nfallback: {cache['fallback_reason']}" if cache["fallback_reason"] else ""), file=sys.stderr)
    print(f"wrote {CACHE_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
