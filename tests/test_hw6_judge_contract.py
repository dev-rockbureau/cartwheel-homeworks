"""HW6: the Harbor judge uses the same input and parser as the HW5 judge."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from analysis.helpers.normalization import _flatten
from harbor_adapter.export import CONVERSATION_ONLY_JUDGES, export_tasks
from replay.rollout import judge_trace_text, load_frozen_judge

REPO_ROOT = Path(__file__).resolve().parents[1]
MODE = "policy_claim_without_citation"


def _transcript_from_hw5(trace: list[dict]) -> dict | None:
    """Rebuild a runtime transcript from an alternating HW5 input, else None."""
    turns = []
    for index in range(0, len(trace), 2):
        pair = trace[index : index + 2]
        if [m["role"] for m in pair] != ["user", "assistant"]:
            return None
        turns.append(
            {
                "user": pair[0]["text"],
                "reply": pair[1]["text"],
                "tool_calls": [{"name": "get_policy", "args": {}, "result": {"policy_id": "x"}}],
            }
        )
    return {"turns": turns}


def test_conversation_only_text_matches_every_hw5_judge_input() -> None:
    records = json.loads((REPO_ROOT / "analysis/state/hw5_trace_inputs.json").read_text())
    compared = 0
    for record in records:
        transcript = _transcript_from_hw5(record["trace"])
        if transcript is None:
            continue
        assert judge_trace_text(transcript, include_tools=False) == _flatten(record["trace"])
        compared += 1
    assert compared >= 90


def test_default_judge_text_still_includes_tools() -> None:
    transcript = {"turns": [{"user": "u", "reply": "a", "tool_calls": [{"name": "t", "args": {}, "result": {}}]}]}
    assert "tool_call:" in judge_trace_text(transcript)
    assert "tool_call:" not in judge_trace_text(transcript, include_tools=False)


def _exported_rubric(tmp_path: Path) -> str:
    case = {
        "id": "e-900",
        "mode": MODE,
        "input": {"role": "shopper", "user_id": 1, "message": "hello"},
        "initial_state": {"world": "reseed", "fixture": None},
        "expected": {"assertions": ["x"], "judges": {MODE: "pass"}},
    }
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case) + "\n")
    export_tasks(cases_path, tmp_path / "tasks", baseline=True)
    return (tmp_path / "tasks" / "e-900" / "tests" / f"judge_{MODE}.py").read_text()


def test_exported_judge_keeps_the_frozen_prompt_model_and_input(tmp_path: Path) -> None:
    assert MODE in CONVERSATION_ONLY_JUDGES
    judge = load_frozen_judge(MODE)
    rubric = _exported_rubric(tmp_path)
    namespace: dict = {}
    for node in ast.parse(rubric).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in {"PROMPT", "MODEL", "INCLUDE_TOOLS"}:
            exec(compile(ast.Module([node], []), "rubric", "exec"), namespace)
    assert namespace["PROMPT"] == judge["prompt_text"]
    assert namespace["MODEL"] == judge["model"] == "gpt-4o-mini-2024-07-18"
    assert namespace["INCLUDE_TOOLS"] is False


def test_exported_judge_parser_is_strict_like_hw5(tmp_path: Path) -> None:
    rubric = _exported_rubric(tmp_path)
    decode_src = next(
        ast.get_source_segment(rubric, node)
        for node in ast.parse(rubric).body
        if isinstance(node, ast.FunctionDef) and node.name == "_decode"
    )
    namespace: dict = {}
    exec(decode_src, namespace)
    decode = namespace["_decode"]
    assert decode({"critique": "c", "result": "Pass"}) == "pass"
    assert decode({"critique": "c", "result": "Fail"}) == "fail"
    for bad in ("pass", "Passed", "Pass.", "Not found", "", None):
        with pytest.raises(ValueError):
            decode({"critique": "c", "result": bad})
    with pytest.raises(ValueError):
        decode({"critique": "", "result": "Pass"})


def test_summary_reads_per_trial_results_when_job_list_is_empty(tmp_path: Path) -> None:
    """Harbor 0.23.0 writes trials to <trial>/result.json, not the job file."""
    from harbor_adapter.summary import load_trial_results, summarize_job

    case = {
        "id": "e-901",
        "mode": "m",
        "input": {"role": "shopper", "user_id": 1, "message": "hi"},
        "initial_state": {"world": "reseed", "fixture": None},
        "expected": {"assertions": ["x"], "checks": [{"check": "no_write_tools"}]},
    }
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case) + "\n")
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"n_total_trials": 5, "trial_results": []}))
    for i, reward in enumerate([1.0, 0.0, 1.0, 1.0, 1.0]):
        trial = job / f"e-901__t{i}"
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "cartwheel/evals__e-901",
                    "trial_name": trial.name,
                    "started_at": f"2026-11-13T00:00:0{4 - i}Z",
                    "exception_info": None,
                    "verifier_result": {"rewards": {"reward": reward}},
                }
            )
        )
    ordered = load_trial_results(job, {"trial_results": []})
    assert [t["trial_name"] for t in ordered] == [f"e-901__t{i}" for i in (4, 3, 2, 1, 0)]
    markdown, _ = summarize_job(job, cases_path=cases_path, expected_attempts=5, classify=True)
    assert "| `e-901` | 4 | 5 |" in markdown
    assert "baseline_pass_rate: 0.8" in markdown
    assert "Infrastructure problems" not in markdown
