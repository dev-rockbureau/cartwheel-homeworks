# HW7 monitor: `policy_claim_without_citation`

The monitor estimates how often Cartwheel's replies make a policy claim with no policy ID cited. It uses the frozen
HW5 judge `policy_claim_without_citation-v2` (`gpt-4o-mini-2024-07-18`), which reads user turns and assistant
replies only, as validated in HW5. Configuration is in [`config.json`](config.json), results are in
[`history.jsonl`](history.jsonl), and the chart is [`prevalence.svg`](prevalence.svg).

## Setup

| | |
|---|---|
| Scenarios | the 50 in `scenarios/monitoring_scenarios.jsonl` (HW3) |
| Cartwheel model | `gpt-4o-mini` in both periods |
| before | HW3 run, 2026-11-07 18:35:26Z to 18:57:13Z |
| after | new run, 2026-11-20 00:52:13Z to 00:56:18Z (`scenarios/hw7-after-results.jsonl`) |
| Agent code between periods | unchanged (`agent/`, `server/`, `seed/`, `facts.yaml` identical to HW3) |
| Random sample | 20% of the 50 conversations; seed 7 picks the same 10 scenarios in both periods |
| Risk groups | `policy_lookup`, `multi_turn` (inspection only, not used in the rate) |
| Threshold | 0.15 corrected failure rate, chosen before any judge result |
| Judge accuracy (HW5 test) | failure sensitivity 1.00 (15/15), pass specificity 0.963 (26/27) |

## Results

| Period | Traces | Conversations | Random flagged | Raw | Corrected | 95% interval | Risk flagged |
|---|---|---|---|---|---|---|---|
| before | 73 | 50 | 6 / 10 | 0.60 | 0.585 | 0.270 to 0.896 | 13 / 19 (`multi_turn` 19, `policy_lookup` 0) |
| after | 73 | 50 | 6 / 10 | 0.60 | 0.585 | 0.270 to 0.896 | 17 / 20 (`multi_turn` 19, `policy_lookup` 1) |

Observed details:

- All 50 conversation texts differ between the periods. Each of the 10 random scenarios got the same verdict in
  both periods. Three risk-group scenarios changed verdict: `support-0040`, `support-0235`, `support-0242`.
- The agent called `get_policy` or `search_help_center` in 0 of 50 conversations (before) and 1 of 50 (after),
  although 35 of the 50 scenarios involve a policy.

## Interpretation

### 1. Did the corrected failure estimate move between the two periods?

The corrected estimate did not move: 0.585 in both periods (6 of 10 random conversations flagged, raw 0.60). The
agent was unchanged, and the same 10 random scenarios received the same verdicts. The risk groups changed slightly
(policy_lookup 0 → 1, 20 risk-group conversations instead of 19), but they don't feed the estimate; see Q3.

### 2. Do the intervals support a conclusion, or is the result uncertain?

The intervals are too wide to detect a change: 0.27 to 0.90 in both periods, because the estimate rests on only 10
random conversations (each one moves the raw rate by 10 points). A real shift of 0.2 would fit inside both
intervals. They do support one conclusion: even the lower bound (0.27) is above the 0.15 threshold, so the failure
rate is very likely above the threshold in both periods.

### 3. What did the risk groups reveal that the random estimate did not?

The risk groups surfaced more failures to inspect: multi_turn flagged 13 of 19 conversations before and 17 of 20
after (30 flagged, against 12 from the random sample). Their rates are not estimates of the failure rate, because
the groups are chosen to fail more often. They also point to a likely cause: the agent rarely looks up policy.
policy_lookup matched 0 of 50 conversations before and 1 of 50 after, even though 35 of the 50 scenarios involve a
policy, so most policy claims are made without consulting a policy that could be cited.

### 4. What action should happen if the estimate crosses the threshold?

The estimate is above the threshold in both periods (0.585 against 0.15), so this applies now. Start error analysis
on the flagged traces (the dashboard's flagged-traces table): a person reviews them and confirms which are real
failures. Each confirmed failure becomes a new evaluation case in the HW6 suite, next to e-011 and e-012. Then test
a fix for the likely cause from Q3, such as requiring the agent to look up and cite the policy, and check it with
the HW6 cases and the next monitoring run.

## Running it

```bash
uv run python -m monitoring.run --period before          # a configured period (asks before judging)
uv run python -m monitoring.run --period after --dry-run # counts only, no judge calls
uv run python -m monitoring.run --last-hours 24 --yes    # the scheduled job
```

The daily job is [`.github/workflows/monitor.yml`](../.github/workflows/monitor.yml). It runs on a self-hosted
runner on the student's Mac, because Langfuse runs there in Docker. Scores go to Langfuse as
`policy_claim_without_citation_verdict` (random sample), `_risk_verdict` (risk groups), and
`_corrected_prevalence` (one per run, attached to the window's last trace). Each score is dated at the trace it
measures, or at the window end for the prevalence score.
