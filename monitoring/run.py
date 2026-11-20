"""Monitor one failure mode with a frozen Homework 5 judge.

Two ways to choose the traces:

    uv run python -m monitoring.run --period before      # a period from monitoring/config.json
    uv run python -m monitoring.run --last-hours 24      # the scheduled job

Steps: fetch the Langfuse traces in the window, group them by session into
conversation records, validate them, sample (random + risk groups), show the
number of judge calls, judge the union once with the frozen judge, correct the
random-sample flag rate, write Langfuse scores, and update the history and chart.

Add ``--dry-run`` to stop before the judge (no model calls, no scores written).
The judge only runs with ``--yes`` (the scheduled job passes it) or after an
interactive "y".

Judge input (student decision, HW7 Part A): the frozen judge
``policy_claim_without_citation-v2`` was validated on user turns and assistant
replies only, so the judge text leaves out tool calls and results. Tool names
are still recorded for each conversation, but only to build the risk groups.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from monitoring.sample import DEFAULT_RISK_GROUPS, select_traces

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "monitoring" / "config.json"
HISTORY = ROOT / "monitoring" / "history.jsonl"
CHART = ROOT / "monitoring" / "prevalence.svg"
OUTPUT_DIR = ROOT / "monitoring" / "output"  # local run output, uploaded by CI, not committed
SCENARIOS = ROOT / "scenarios" / "monitoring_scenarios.jsonl"
CONVERSATIONS_PER_PERIOD = 50


# ---------------------------------------------------------------- config and window


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(config["risk_groups"]) - set(DEFAULT_RISK_GROUPS)
    if unknown:
        raise SystemExit(f"unknown risk groups in config: {sorted(unknown)}")
    return config


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scenario_ids(path: Path = SCENARIOS) -> set[str]:
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


# ---------------------------------------------------------------- fetch


def fetch_window(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Every Langfuse trace whose timestamp falls in [start, end], normalized."""
    from analysis.helpers import langfuse_io
    from analysis.helpers.normalization import normalize_trace
    from observability.instrument import load_env

    load_env()
    if not langfuse_io.is_configured():
        raise SystemExit("Langfuse is not configured: set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST")
    from langfuse import get_client

    client = get_client()
    summaries: list[Any] = []
    page = 1
    while True:
        response = client.api.trace.list(
            from_timestamp=start, to_timestamp=end, page=page, limit=100
        )
        batch = list(response.data or [])
        summaries.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    seen: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        if str(summary.id) not in seen:
            seen[str(summary.id)] = normalize_trace(client.api.trace.get(summary.id))
    return list(seen.values())


# ---------------------------------------------------------------- conversations


def model_matches(observed: str, configured: str) -> bool:
    """The configured model, or a dated snapshot of it (gpt-4o-mini-2024-07-18)."""
    return observed == configured or re.fullmatch(
        re.escape(configured) + r"-\d{4}-\d{2}-\d{2}", observed
    ) is not None


def judge_text(turns: list[dict[str, Any]]) -> str:
    """The HW5 judge input: user turns and assistant replies only, in order.

    Same format as ``replay.rollout.judge_trace_text(..., include_tools=False)``,
    which HW6 checked against all 105 HW5 inputs: one ``user: ...`` and one
    ``assistant: ...`` line per turn, skipping empty text.
    """
    from replay.rollout import judge_trace_text

    return judge_trace_text(
        {"turns": [{"user": t["user"], "reply": t["reply"]} for t in turns]},
        include_tools=False,
    )


def build_conversations(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group traces by session into one record per conversation.

    Turns are split exactly as in the HW4 review app (the reply is what the
    user saw). The record id is the conversation's FINAL trace id, so the
    Langfuse scores attach to the last turn. ``tools`` and ``turn_count`` come
    from trace evidence and feed the risk groups only; ``text`` (the judge
    input) holds no labels, scenario metadata, or tool output.
    """
    from analysis.review_app.load_traces import _output_text, build_turn

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        session = trace.get("meta", {}).get("session_id") or trace["trace_id"]
        by_session[session].append(trace)

    conversations = []
    for session, group in by_session.items():
        group.sort(key=lambda t: (t.get("timestamp") or "", t["trace_id"]))
        turns = [build_turn(t, _output_text(t)) for t in group]
        conversations.append(
            {
                "id": group[-1]["trace_id"],
                "session_id": session,
                "scenario_id": group[0].get("meta", {}).get("scenario_id"),
                "trace_ids": [t["trace_id"] for t in group],
                "models": sorted({m for t in group for m in t.get("models", [])}),
                "tools": sorted({s["name"] for turn in turns for s in turn["steps"] if s["name"]}),
                "turn_count": len(turns),
                "text": judge_text(turns),
            }
        )
    conversations.sort(key=lambda c: (c["scenario_id"] or "", c["id"]))
    return conversations


def validate(
    conversations: list[dict[str, Any]], model: str, expected_ids: set[str] | None
) -> None:
    """Reject the window instead of producing a misleading estimate."""
    problems = []
    for conv in conversations:
        if not conv["models"]:
            problems.append(f"{conv['id']}: no model recorded")
        bad = [m for m in conv["models"] if not model_matches(m, model)]
        if bad:
            problems.append(f"{conv['scenario_id'] or conv['id']}: model {bad}, expected {model}")
        if not conv["text"].strip():
            problems.append(f"{conv['id']}: empty conversation text")
    if expected_ids is not None:
        found = [c["scenario_id"] for c in conversations]
        missing = sorted(expected_ids - set(found))
        repeated = sorted({sid for sid in found if found.count(sid) > 1})
        if missing:
            problems.append(f"missing scenario ids: {missing}")
        if repeated:
            problems.append(f"scenario ids in more than one session (retries?): {repeated}")
        if len(conversations) != CONVERSATIONS_PER_PERIOD:
            problems.append(f"{len(conversations)} conversations, expected {CONVERSATIONS_PER_PERIOD}")
    if problems:
        raise SystemExit("period rejected:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- scores


def anchor_period_score(
    records: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach the period's prevalence score to the window's final trace.

    Langfuse v3 rejects a score with no trace, session, or dataset run ("Provide
    exactly one of ... traceId, sessionId or datasetRunId"), and
    ``build_score_records`` leaves the period score's trace_id as None by
    contract. Student decision (HW7 Part C): anchor it to the latest trace in
    the window. The score_id is unchanged, so a rerun still updates it.
    """
    last = max(traces, key=lambda t: (t.get("timestamp") or "", t["trace_id"]))["trace_id"]
    return [
        {**record, "trace_id": last} if record["trace_id"] is None else record
        for record in records
    ]


def date_scores(
    records: list[dict[str, Any]], traces: list[dict[str, Any]], period_end: datetime
) -> list[dict[str, Any]]:
    """Give each score the time of the behavior it measures, not the write time.

    Student decision (HW7 Part D): a verdict score takes its trace's timestamp
    and the period's prevalence score takes the window end, so the Langfuse
    dashboard plots each period at its own date.
    """
    times = {t["trace_id"]: _utc(t["timestamp"]) for t in traces if t.get("timestamp")}
    dated = []
    for record in records:
        if record["name"].endswith("_corrected_prevalence"):
            dated.append({**record, "timestamp": period_end})
        else:
            dated.append({**record, "timestamp": times[record["trace_id"]]})
    return dated


# ---------------------------------------------------------------- history and chart


def update_history(record: dict[str, Any], path: Path = HISTORY) -> None:
    """One line per label: a rerun replaces that period's line."""
    rows = []
    if path.exists():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["label"] != record["label"]] + [record]
    rows.sort(key=lambda r: r["window"]["from"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_chart(config: dict[str, Any], path: Path = HISTORY, out: Path = CHART) -> None:
    from monitoring.chart import prevalence_chart

    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    points = [r for r in rows if r.get("corrected") is not None]
    if points:
        out.write_text(
            prevalence_chart(points, threshold=config["threshold"], mode=config["judge_mode"]),
            encoding="utf-8",
        )


# ---------------------------------------------------------------- main


def run(
    label: str,
    start: datetime,
    end: datetime,
    config: dict[str, Any],
    expected_ids: set[str] | None,
    *,
    dry_run: bool,
    assume_yes: bool,
) -> dict[str, Any] | None:
    mode, judge_id = config["judge_mode"], config["judge_id"]
    traces = fetch_window(start, end)
    if expected_ids is not None:
        # A period window can hold other scenarios' traces; keep the monitored ones.
        traces = [t for t in traces if t.get("meta", {}).get("scenario_id") in expected_ids]
    conversations = build_conversations(traces)
    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "label": label,
        "judge_id": judge_id,
        "mode": mode,
        "model": config["model"],
        "window": {"from": _iso(start), "to": _iso(end)},
        "trace_count": len(traces),
        "conversation_count": len(conversations),
    }
    print(f"[{label}] {_iso(start)} -> {_iso(end)}: {len(traces)} traces, {len(conversations)} conversations")

    if not conversations:
        record = {**base, "random_count": 0, "risk_count": 0, "judge_calls": 0, "corrected": None}
        (out_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print("no eligible conversations; nothing to judge")
        return record

    validate(conversations, config["model"], expected_ids)
    groups = {name: DEFAULT_RISK_GROUPS[name] for name in config["risk_groups"]}
    plan = select_traces(conversations, config["random_rate"], groups)
    random_ids = [c["id"] for c in plan["random"]]
    risk_ids = list(dict.fromkeys(c["id"] for g in plan["risk_groups"].values() for c in g))
    judge_ids = [c["id"] for c in plan["to_judge"]]
    print(f"  random sample: {len(random_ids)}")
    for name, members in plan["risk_groups"].items():
        print(f"  risk group {name}: {len(members)}")
    print(f"  in both: {len(set(random_ids) & set(risk_ids))}")
    print(f"  judge calls: {len(judge_ids)} ({judge_id})")
    (out_dir / "plan.json").write_text(
        json.dumps(
            {
                **base,
                "random": random_ids,
                "risk_groups": {n: [c["id"] for c in m] for n, m in plan["risk_groups"].items()},
                "to_judge": judge_ids,
                "conversations": [
                    {k: c[k] for k in ("id", "scenario_id", "session_id", "trace_ids", "tools", "turn_count")}
                    for c in conversations
                ],
            },
            indent=2,
        )
        + "\n"
    )
    if dry_run:
        print("dry run: stopped before the judge")
        return None
    if not assume_yes and input("run the judge? [y/N] ").strip().lower() != "y":
        print("stopped before the judge")
        return None

    from monitoring.correct import corrected_mode_prevalence
    from monitoring.run_judges import judge_sample, judge_test_data
    from monitoring.write_scores import build_score_records, post_scores

    verdicts = judge_sample(judge_id, plan["to_judge"])
    random_verdicts = {i: verdicts[i] for i in random_ids}
    risk_verdicts = {i: verdicts[i] for i in risk_ids}
    (out_dir / "verdicts.json").write_text(
        json.dumps({"random": random_verdicts, "risk": risk_verdicts}, indent=2) + "\n"
    )

    test_labels, test_preds = judge_test_data(judge_id)
    estimate = corrected_mode_prevalence(list(random_verdicts.values()), test_labels, test_preds)
    records = date_scores(
        anchor_period_score(
            build_score_records(mode, random_verdicts, risk_verdicts, estimate, label), traces
        ),
        traces,
        end,
    )
    written = post_scores(records)
    print(f"  wrote {written} Langfuse scores")

    record = {
        **base,
        "random_count": len(random_ids),
        "risk_count": len(risk_ids),
        "risk_groups": {n: len(m) for n, m in plan["risk_groups"].items()},
        "judge_calls": len(judge_ids),
        "random_flagged": sum(random_verdicts.values()),
        "risk_flagged": sum(risk_verdicts.values()),
        **{k: estimate[k] for k in ("raw", "corrected", "ci_low", "ci_high", "confidence",
                                    "failure_sensitivity", "pass_specificity", "n_sample")},
        "threshold": config["threshold"],
    }
    (out_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"  raw {estimate['raw']}, corrected {estimate['corrected']} "
        f"[{estimate['ci_low']}, {estimate['ci_high']}], threshold {config['threshold']}"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--period", help="a period label from monitoring/config.json")
    which.add_argument("--last-hours", type=float, help="monitor the last N hours (scheduled job)")
    parser.add_argument("--dry-run", action="store_true", help="stop before the judge")
    parser.add_argument("--yes", action="store_true", help="run the judge without asking")
    args = parser.parse_args(argv)
    config = load_config()

    if args.period:
        periods = {p["label"]: p for p in config["periods"]}
        if args.period not in periods:
            parser.error(f"unknown period {args.period!r}; config has {sorted(periods)}")
        period = periods[args.period]
        label, start, end = args.period, _utc(period["from"]), _utc(period["to"])
        expected: set[str] | None = scenario_ids()
    else:
        end = datetime.now(timezone.utc).replace(microsecond=0)
        start = end - timedelta(hours=args.last_hours)
        label, expected = f"last{args.last_hours:g}h-{_iso(end)}", None

    record = run(label, start, end, config, expected, dry_run=args.dry_run, assume_yes=args.yes)
    if record is not None:
        if args.period:
            update_history(record)
            write_chart(config)
        else:
            # The scheduled job keeps its result in the uploaded output, not in the committed history.
            update_history(record, OUTPUT_DIR / "history.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
