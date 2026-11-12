# Interface comparison

My review interface is in `analysis/review_app/`. I compared it with the reference interface
(`analysis/server.py`, `analysis/ui/index.html`) after reviewing seven conversations in the standard Langfuse
annotation view (support-0029, 0245, 0233, 0180, 0215, 0164, 0118). Traces were loaded from the live Langfuse
project. The offline export fallback was not needed.

## Retained from the reference interface

**Free-text margin notes placed beside the evidence, auto-saved through a small file-backed JSON API.** I select
text in a message (or use "+ note") and type an open code; it is saved to `analysis/state/annotations.json` on
every change. I kept this because open coding should be free text written where the evidence is, not a form or a
category picked from a list.

## Changed after inspecting my traces

**One conversation per screen, grouped by session, with each tool call and its result in one numbered step box.**
In Langfuse, each user turn is a separate trace, tool calls and their outputs sit in different spans, and the
`openai.response` input repeats the whole conversation. I had to click through spans to find a tool call, its
arguments, and its result, and a turn-2 trace did not show what the tools returned in turn 1. My interface shows
the turns in time order, puts the call (with arguments) and the result (one-line summary, full JSON on expand)
together, and labels the model's text between tool calls as "not shown to user", separate from the reply the
customer actually saw.

During review I also added store and product names next to the ids in every tool result. In 0178 and 0238-0242 the
wrong store's or wrong product's order went unnoticed, because the tools return ids only.

## Remaining limitation

**The taxonomy view is read-only.** Modes are edited in `analysis/state/patterns.json` (through my coding agent,
on my decisions) rather than in the interface, so renaming a mode or moving a note between modes cannot be done
by clicking.
