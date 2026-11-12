# HW4 review summary

Prepared from the state files on 2026-11-12; numbers are computed from `analysis/state/`. Section 4 was chosen and
section 11 written by the student.

## 1. Reviewed sample

- **Population:** the HW3 final run, `support-*` conversations in Langfuse: 250 conversations (346 traces).
  `pilot-*` and the smoke trace were excluded.
- **Unit of review:** the conversation (Langfuse session). Labels and scores are attached to the conversation's
  first trace id.
- **Reviewed:** **109 distinct conversations (160 traces)**, no conversation in two batches. support-0029 was used for
  practice and excluded from the sample.

| Batch | Method | Conversations | Roles (shopper / merchant / support) | Coverage / challenge | With a failure note |
|---|---|---|---|---|---|
| `uniform-1` | uniform random (seed 20261110) | 15 | 9 / 3 / 3 | 12 / 3 | 3 |
| `cluster-1` | k-means representatives (k = 8) | 15 | 10 / 3 / 2 | 11 / 4 | 8 |
| `dimension-2` | stratified by intent, 3 per intent | 30 | 16 / 9 / 5 | 22 / 8 | 14 |
| `depth-3` | depth searches on the setup of draft modes | 25 | 21 / 0 / 4 | 14 / 11 | 16 |
| `uniform-4` | uniform random (seed 20261111), the final 15 | 15 | 7 / 5 / 3 | 11 / 4 | 6 |
| `search-B` | Part D search v1 for `reply_contradicts_tool_result` | 7 | 7 / 0 / 0 | 4 / 3 | 6 |
| `search-B-v2` | Part D search v2 (after a definition revision) | 2 | 2 / 0 / 0 | 1 / 1 | 2 |
| **Total** | | **109** | **72 / 20 / 17** | **75 / 34** | **55** |

By intent: order_status 19, return_eligibility 18, refund 17, out_of_scope 10, policy_question 10, product_search 10,
account_change 8, cancellation 8, dispute 5, account_info 4. By length: 67 one-turn, 33 two-turn, 9 three-turn.

The sample is deliberately not representative: batches 2, 3 and the two searches steered toward situations where
failures were expected. All fractions below are **sample fractions, not prevalence estimates**.

## 2. Final taxonomy (8 modes) and labels

Every conversation in the sample has a Pass/Fail label for every final mode (109 x 8 = 872 labels), written to
`analysis/state/labels/<mode>.jsonl` and to Langfuse as scores (`hw4-<mode>-<trace id>`).

| Mode | Requirement | Evaluator | Positives / close negatives | Fail | Sample fraction |
|---|---|---|---|---|---|
| `eligibility_without_order_lookup` | PURPOSE-1, RESP-3 | LLM judge + code pre-check | 8 / 4 | 8 | 7.3% |
| `reply_contradicts_tool_result` | RESP-3 | LLM judge | 20 / 7 | 22 | 20.2% |
| `omitted_restocking_fee` | RESP-6 (new) | code | 7 / 0 * | 9 | 8.3% |
| `policy_claim_without_citation` | RESP-1 (PURPOSE-1) | LLM judge | 8 / 3 | 39 | 35.8% |
| `ambiguous_match_presented_as_unique` | RESP-3 | LLM judge | 4 / 3 | 4 | 3.7% |
| `required_escalation_not_made` | ESC-2, ESC-3 | LLM judge | 9 / 6 | 14 | 12.8% |
| `policy_question_answered_without_lookup` | PURPOSE-1, RESP-3 | LLM judge | 7 / 3 | 9 | 8.3% |
| `absence_claimed_after_bad_search_arguments` | RESP-3, TOOL-3, TOOL-6 | LLM judge | 3 / 3 | 4 | 3.7% |

\* No close negative exists in the data: every fee-store return-eligibility answer in all 250 conversations omits
the fee in its first answer (recorded under the handout's "when the reviewed data permits" exception).

Definitions, boundaries, nearest modes, product changes, originating annotations and the full revision history are
in `analysis/state/patterns.json`.

**How the labels were made.** Each cell started as a proposal in `analysis/state/label_proposals.json`:
- **group 1:** the student's Part D decisions;
- **group 2:** "the situation never comes up", by one student-approved rule per mode;
- **group 3:** the agent's reading, which the student verified.

Fails and UNSURE cells were labelled one at a time. High-confidence group-3 Passes were accepted in bulk only after
a 5-cell random spot check per column; all 30 spot-check cells agreed. Provenance per label (`accepted_via`):
81 individual, 88 group-1 bulk, 467 group-2 bulk, 148 group-3 bulk after a spot check, and 88 accepted before the
field existed. The student overrode one proposal and then restored it (D x 0063). Labelling rule (student decision):
each mode is judged on the trace as it reads, independently of other modes, so a failure caused by another mode's
failure still counts.

## 3. Stability: new modes in the final 15

The final batch (`uniform-4`, 15 uniform random conversations) produced 6 failures. Five fit existing modes
(B x3: 0040, 0039, 0181; F: 0080; A, widened: 0023). **One previously unseen consequential mode appeared**:
support-0202, where the agent searched "kitchen" as a product title, got nothing, and told the user the store had no
such items although three exist. It became `absence_claimed_after_bad_search_arguments` (3 positives after
rechecking the earlier traces: 0202, 0124, 0072). One new mode in the final 15 suggests the taxonomy is close to
stable; the handout's threshold for reviewing another batch ("several new modes") was not reached.

## 4. One taxonomy revision

**`reply_contradicts_tool_result`: the "listing" boundary, revised after a rejected search result.**
The Part D search for more B instances proposed support-0041 because the reply said "none of them", and an order
matching the user's product was in the results. The student rejected it: *"the reply indicates the correct order was
found and the ineligibility stated matches the order record."* The rejection exposed an unwritten line. 0041, 0013
and 0186 all list several orders as "related", but only 0013 and 0186 present a different product's order as the one
the user named. The boundary now reads: listing other results passes when the reply keeps them apart from the named
order (0041); presenting a different product's order as the named one fails (0013, 0186). The search was then
repeated (v2, with typo-tolerant title matching and a wrong-store signal), which added 0008.

Other revisions recorded in `patterns.json`:
- `eligibility_without_order_lookup` widened to a retrieved but unused delivery date (0023);
- `required_escalation_not_made` judged per reply, so an offer to escalate fails (0054, 0057);
- `policy_claim_without_citation` follows PURPOSE-1's "every policy claim", carried forward across turns;
- `ambiguous_match_presented_as_unique` dropped, then restored when 0244 supplied a third positive;
- the new mode was renamed after the AgentDebug comparison.

## 5. Specification revision

`SPEC.md` gains **RESP-6**: *"When a store opts in to a restocking fee, the first answer about returning an item from
that store states the fee (up to the store's maximum, opened items only) and cites the store's policy."* SPEC had no
requirement behind `omitted_restocking_fee`; the rule existed only in the `cw-restocking-fees` policy document.

**Motivating annotation:** support-0216 (batch 1): *"the restocking fee should have been mentioned here since it is
part of the store's return policy, missing citation."*

This revises the specification only. The running agent does not yet follow RESP-6: no code reads SPEC.md, and the
system prompt was not changed.

## 6. Search for additional instances (Part D)

- **Target:** `reply_contradicts_tool_result`, across all 250 conversations. The search used deterministic retrieval
  signals in `analysis/review_app/search_mode.py`: S1 wrong order, S2 false absence, S3 queued reported as done.
  Product titles were joined in from the seed database.
- **Recall check:** on the known positives, v1 caught 10 of 12 and v2 caught 16 of 18. Two shapes are missed: a wrong
  date with no order id (0178), and a request with no product named (0246).
- **v1:** 8 suggestions. 5 accepted, 1 revised, 2 rejected (0041, and 0243, whose reason pointed at an earlier T1
  failure that the student then noted as B).
- **v2:** 2 suggestions. 1 accepted (0008), 1 rejected (0244; its earlier failure became an
  `ambiguous_match_presented_as_unique` positive).
- **Outcome:** all decisions are in `analysis/state/suggestions.json`; 3 suggestions were rejected in total. The 9
  conversations retrieved by the search are part of the review set and carry labels for every mode.

## 7. Comparison with AgentDebug

Compared with the AgentErrorTaxonomy (Zhu et al., arXiv:2509.25370); the mapping is recorded per mode in
`patterns.json`.
- **Close matches:** B maps exactly to Reflection: Outcome Misinterpretation. A and G map to Memory: Retrieval
  Failure. The new mode maps to Action: Parameter Error followed by Reflection: Causal Misattribution, which motivated
  its rename.
- **Gaps in AgentDebug:** `omitted_restocking_fee`, `policy_claim_without_citation` and `required_escalation_not_made`
  have no AgentDebug category. They are specification-compliance failures (required content, citations, mandatory
  hand-offs), which a taxonomy built for task agents in simulated environments does not cover.
- **Considered and not added:**
  - Parameter Error as its own mode: the consequential cases are already the new mode.
  - Inefficient Planning: see section 8.
  - System errors: none were observed.

## 8. Observed but not modelled

- **Dropped for lack of positives:**
  - `out_of_scope_request_answered`: 1 positive; the agent declined correctly in 6 close negatives.
  - `correction_not_reconciled`: 1 positive.
  - `policy_rule_misapplied`: 2 positives, usually hidden behind an earlier A or F failure.
- **Out of scope, no requirement:** the agent asks for information it already has (0109 asked for the refund amount
  shown in the record; 0154 asked for a reason the user had given). AgentDebug calls this Inefficient Planning.
- **Notable severity inside B:** the agent moved money or cancelled on the wrong order (0033 refunded the wrong
  order; 0042 escalated a refund on the wrong order; 0243 cancelled the wrong order).

## 9. Preparing for Homework 5

HW5 needs at least 30 Pass and 30 Fail labels per mode. Every mode has at least 71 Passes, but **no mode reaches 30
Fails**:

| Mode | A | B | C | D | E | F | G | K |
|---|---|---|---|---|---|---|---|---|
| Fail labels | 8 | 22 | 9 | 39 | 4 | 14 | 9 | 4 |

The exception is D (39). The other modes will need synthetic scenarios targeting them. `omitted_restocking_fee` will
also need synthetic Pass cases near its boundary, because the data has no close negative.

## 10. Checks: offline and live

- **Offline:** sampling, the depth searches, the Part D searches, setup rules and label proposals, review-app tests
  (12 passing), and the label audit (872/872 labelled). These read the Langfuse export cache, `scenarios/`, the
  read-only seed database and the state files.
- **Live (Langfuse, no model calls):** trace export for the review cache, and one score write per label (872). One
  score was verified by reading it back.
- **No live model was run in HW4.** The agent's behaviour comes from the HW3 run (gpt-4o-mini).

## 11. Student reflection

What surprised me was the number of scenarios to review, and how the agent can miss data that is quite obvious if it
only looked at the available data closely.

It was also difficult to make an assessment when responses do not include critical information like store or product
names, which is why I added them to the review app.

What I would do differently is first understand the process and its steps, and what they are for.
